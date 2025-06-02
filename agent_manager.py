"""
实现父子Agent注册管理和任务分解机制
"""
from typing import Dict, List, Any, Type, Optional, Union, Callable, TypeVar, Generic, Set
import json
import os
from datetime import datetime
import inspect
import uuid
from pydantic import BaseModel, Field, ValidationError, create_model

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.callbacks import CallbackManagerForToolRun

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint import MemorySaver

from base import Executor, Planner, AgentRegistry
from langgraph_impl import LangGraphExecutor, LangGraphPlanner, GraphState
from executor_impl import ConcreteExecutor, ConcretePlanner, ToolRegistry
from validation import EnhancedExecutor, InteractiveValidator


class AgentConfig(BaseModel):
    """Agent配置"""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent名称")
    description: str = Field(..., description="Agent描述")
    type: str = Field(..., description="Agent类型，如'executor'或'planner'")
    parent_id: Optional[str] = Field(None, description="父Agent ID")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="输入参数模式")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出参数模式")
    tools: List[str] = Field(default_factory=list, description="工具列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class AgentManager:
    """
    Agent管理器，用于管理所有Agent的注册、查询和关系
    """
    def __init__(self):
        self.agents = {}  # id -> agent实例
        self.configs = {}  # id -> agent配置
        self.children = {}  # parent_id -> [child_id]
        self.registry = AgentRegistry()
        self.tool_registry = ToolRegistry()
    
    def register_agent(self, 
                       agent: Executor, 
                       config: AgentConfig) -> str:
        """
        注册Agent
        
        Args:
            agent: Agent实例
            config: Agent配置
            
        Returns:
            Agent ID
        """
        agent_id = config.id
        
        # 保存Agent和配置
        self.agents[agent_id] = agent
        self.configs[agent_id] = config
        
        # 注册到AgentRegistry
        self.registry.register(config.name, agent)
        
        # 如果有父Agent，建立父子关系
        if config.parent_id:
            if config.parent_id not in self.children:
                self.children[config.parent_id] = []
            self.children[config.parent_id].append(agent_id)
            
            # 将子Agent注册到父Agent
            parent_agent = self.agents.get(config.parent_id)
            if parent_agent:
                parent_agent.register_executor(agent)
        
        return agent_id
    
    def create_agent(self, 
                     config: AgentConfig, 
                     llm: Any, 
                     tools: List[BaseTool] = None) -> str:
        """
        创建并注册Agent
        
        Args:
            config: Agent配置
            llm: 用于驱动agent的模型
            tools: 工具列表
            
        Returns:
            Agent ID
        """
        # 创建输入模式
        input_schema = None
        if config.input_schema:
            input_schema = create_model(
                f"{config.name}Input",
                **{k: (v.get("type", Any), ... if v.get("required", False) else None) 
                   for k, v in config.input_schema.items()}
            )
        
        # 创建输出模式
        output_schema = None
        if config.output_schema:
            output_schema = create_model(
                f"{config.name}Output",
                **{k: (v.get("type", Any), ... if v.get("required", False) else None) 
                   for k, v in config.output_schema.items()}
            )
        
        # 获取工具
        agent_tools = []
        for tool_name in config.tools:
            tool = self.tool_registry.get(tool_name)
            if tool:
                agent_tools.append(tool)
        
        # 创建Agent
        if config.type.lower() == "planner":
            agent = ConcretePlanner(
                description=config.description,
                llm=llm,
                tools=agent_tools or tools,
                input_schema=input_schema,
                checkpoint_dir=f"./checkpoints/{config.id}"
            )
        else:
            agent = ConcreteExecutor(
                description=config.description,
                llm=llm,
                tools=agent_tools or tools,
                input_schema=input_schema,
                output_schema=output_schema,
                checkpoint_dir=f"./checkpoints/{config.id}"
            )
        
        # 注册Agent
        return self.register_agent(agent, config)
    
    def get_agent(self, agent_id: str) -> Optional[Executor]:
        """
        获取Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent实例
        """
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[Executor]:
        """
        通过名称获取Agent
        
        Args:
            name: Agent名称
            
        Returns:
            Agent实例
        """
        return self.registry.get(name)
    
    def get_children(self, parent_id: str) -> List[str]:
        """
        获取子Agent ID列表
        
        Args:
            parent_id: 父Agent ID
            
        Returns:
            子Agent ID列表
        """
        return self.children.get(parent_id, [])
    
    def get_config(self, agent_id: str) -> Optional[AgentConfig]:
        """
        获取Agent配置
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent配置
        """
        return self.configs.get(agent_id)
    
    def list_agents(self) -> List[AgentConfig]:
        """
        列出所有Agent
        
        Returns:
            Agent配置列表
        """
        return list(self.configs.values())
    
    def register_tool(self, name: str, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            name: 工具名称
            tool: 工具实例
        """
        self.tool_registry.register(name, tool)


class TaskManager:
    """
    任务管理器，用于管理任务的创建、分配和执行
    """
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.tasks = {}  # task_id -> task
        self.task_results = {}  # task_id -> result
        self.task_status = {}  # task_id -> status
        self.task_dependencies = {}  # task_id -> [dependency_task_id]
        self.task_children = {}  # task_id -> [child_task_id]
    
    def create_task(self, 
                    agent_id: str, 
                    input_data: Dict[str, Any],
                    parent_task_id: str = None) -> str:
        """
        创建任务
        
        Args:
            agent_id: Agent ID
            input_data: 输入数据
            parent_task_id: 父任务ID
            
        Returns:
            任务ID
        """
        # 生成任务ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # 创建任务
        task = {
            "id": task_id,
            "agent_id": agent_id,
            "input": input_data,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "parent_task_id": parent_task_id
        }
        
        # 保存任务
        self.tasks[task_id] = task
        self.task_status[task_id] = "created"
        
        # 如果有父任务，建立父子关系
        if parent_task_id:
            if parent_task_id not in self.task_children:
                self.task_children[parent_task_id] = []
            self.task_children[parent_task_id].append(task_id)
        
        return task_id
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            执行结果
        """
        # 获取任务
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        
        # 获取Agent
        agent_id = task["agent_id"]
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} 不存在")
        
        # 更新任务状态
        self.task_status[task_id] = "running"
        task["status"] = "running"
        task["started_at"] = datetime.now().isoformat()
        
        try:
            # 执行任务
            result = agent.execute(task["input"])
            
            # 更新任务状态和结果
            self.task_status[task_id] = "completed"
            self.task_results[task_id] = result
            task["status"] = "completed"
            task["completed_at"] = datetime.now().isoformat()
            task["result"] = result
            
            return result
        except Exception as e:
            # 更新任务状态和错误
            self.task_status[task_id] = "failed"
            task["status"] = "failed"
            task["error"] = str(e)
            task["completed_at"] = datetime.now().isoformat()
            
            raise
    
    def plan_task(self, task_id: str) -> Dict[str, str]:
        """
        规划任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            子任务ID映射
        """
        # 获取任务
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        
        # 获取Agent
        agent_id = task["agent_id"]
        agent = self.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} 不存在")
        
        # 检查Agent类型
        agent_config = self.agent_manager.get_config(agent_id)
        if not agent_config or agent_config.type.lower() != "planner":
            raise ValueError(f"Agent {agent_id} 不是计划器")
        
        # 更新任务状态
        self.task_status[task_id] = "planning"
        task["status"] = "planning"
        
        try:
            # 生成计划
            plan = agent.plan(task["input"])
            
            # 创建子任务
            subtasks = {}
            for subtask_name, executor in plan.items():
                # 获取执行器ID
                executor_id = None
                for eid, eagent in self.agent_manager.agents.items():
                    if eagent is executor:
                        executor_id = eid
                        break
                
                if not executor_id:
                    continue
                
                # 创建子任务输入
                subtask_input = {
                    "task_name": subtask_name,
                    "task_description": f"执行 {subtask_name} 任务",
                    **task["input"]
                }
                
                # 创建子任务
                subtask_id = self.create_task(executor_id, subtask_input, task_id)
                subtasks[subtask_name] = subtask_id
            
            # 更新任务状态和计划
            self.task_status[task_id] = "planned"
            task["status"] = "planned"
            task["plan"] = subtasks
            
            return subtasks
        except Exception as e:
            # 更新任务状态和错误
            self.task_status[task_id] = "plan_failed"
            task["status"] = "plan_failed"
            task["error"] = str(e)
            
            raise
    
    def execute_plan(self, task_id: str) -> Dict[str, Any]:
        """
        执行计划
        
        Args:
            task_id: 任务ID
            
        Returns:
            执行结果
        """
        # 获取任务
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        
        # 检查任务状态
        if task["status"] != "planned":
            raise ValueError(f"任务 {task_id} 未规划")
        
        # 获取子任务
        subtasks = task.get("plan", {})
        if not subtasks:
            raise ValueError(f"任务 {task_id} 没有子任务")
        
        # 执行子任务
        results = {}
        for subtask_name, subtask_id in subtasks.items():
            try:
                # 执行子任务
                result = self.execute_task(subtask_id)
                results[subtask_name] = result
            except Exception as e:
                # 记录错误
                results[subtask_name] = {"error": str(e)}
        
        # 更新任务状态和结果
        self.task_status[task_id] = "completed"
        self.task_results[task_id] = results
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        task["result"] = results
        
        return results
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息
        """
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务结果
        """
        return self.task_results.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态
        """
        return self.task_status.get(task_id)
    
    def get_subtasks(self, task_id: str) -> List[str]:
        """
        获取子任务ID列表
        
        Args:
            task_id: 任务ID
            
        Returns:
            子任务ID列表
        """
        return self.task_children.get(task_id, [])


class MultiAgentSystem:
    """
    多Agent系统，集成Agent管理器和任务管理器
    """
    def __init__(self, llm: Any):
        self.agent_manager = AgentManager()
        self.task_manager = TaskManager(self.agent_manager)
        self.llm = llm
    
    def create_agent(self, 
                     name: str, 
                     description: str, 
                     agent_type: str = "executor",
                     parent_name: str = None,
                     input_schema: Dict[str, Any] = None,
                     output_schema: Dict[str, Any] = None,
                     tools: List[str] = None) -> str:
        """
        创建Agent
        
        Args:
            name: Agent名称
            description: Agent描述
            agent_type: Agent类型，如'executor'或'planner'
            parent_name: 父Agent名称
            input_schema: 输入参数模式
            output_schema: 输出参数模式
            tools: 工具列表
            
        Returns:
            Agent ID
        """
        # 生成Agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # 获取父Agent ID
        parent_id = None
        if parent_name:
            parent_agent = self.agent_manager.get_agent_by_name(parent_name)
            if parent_agent:
                for aid, agent in self.agent_manager.agents.items():
                    if agent is parent_agent:
                        parent_id = aid
                        break
        
        # 创建Agent配置
        config = AgentConfig(
            id=agent_id,
            name=name,
            description=description,
            type=agent_type,
            parent_id=parent_id,
            input_schema=input_schema,
            output_schema=output_schema,
            tools=tools or []
        )
        
        # 创建Agent
        return self.agent_manager.create_agent(config, self.llm)
    
    def register_tool(self, name: str, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            name: 工具名称
            tool: 工具实例
        """
        self.agent_manager.register_tool(name, tool)
    
    def create_task(self, 
                    agent_name: str, 
                    input_data: Dict[str, Any]) -> str:
        """
        创建任务
        
        Args:
            agent_name: Agent名称
            input_data: 输入数据
            
        Returns:
            任务ID
        """
        # 获取Agent
        agent = self.agent_manager.get_agent_by_name(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} 不存在")
        
        # 获取Agent ID
        agent_id = None
        for aid, a in self.agent_manager.agents.items():
            if a is agent:
                agent_id = aid
                break
        
        if not agent_id:
            raise ValueError(f"无法获取Agent {agent_name} 的ID")
        
        # 创建任务
        return self.task_manager.create_task(agent_id, input_data)
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            执行结果
        """
        return self.task_manager.execute_task(task_id)
    
    def plan_and_execute(self, 
                         planner_name: str, 
                         input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        规划并执行任务
        
        Args:
            planner_name: 计划器名称
            input_data: 输入数据
            
        Returns:
            执行结果
        """
        # 创建任务
        task_id = self.create_task(planner_name, input_data)
        
        # 规划任务
        self.task_manager.plan_task(task_id)
        
        # 执行计划
        return self.task_manager.execute_plan(task_id)
    
    def get_agent(self, name: str) -> Optional[Executor]:
        """
        获取Agent
        
        Args:
            name: Agent名称
            
        Returns:
            Agent实例
        """
        return self.agent_manager.get_agent_by_name(name)
    
    def list_agents(self) -> List[AgentConfig]:
        """
        列出所有Agent
        
        Returns:
            Agent配置列表
        """
        return self.agent_manager.list_agents()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务信息
        """
        return self.task_manager.get_task(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务结果
        """
        return self.task_manager.get_task_result(task_id)
