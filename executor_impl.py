"""
实现Executor的关键成员变量和抽象方法的具体实现
"""
from typing import Dict, List, Any, Type, Optional, Union, Callable
import json
import os
from datetime import datetime
import inspect
from pydantic import BaseModel, Field, ValidationError

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


class ExecutorInput(BaseModel):
    """执行器输入的基础模型"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    task_description: str = Field(..., description="任务描述")
    
    class Config:
        extra = "allow"  # 允许额外字段


class ExecutorOutput(BaseModel):
    """执行器输出的基础模型"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    status: str = Field(..., description="任务状态，如'success'、'failed'等")
    result: Dict[str, Any] = Field(default_factory=dict, description="任务结果")
    error: Optional[str] = Field(None, description="错误信息，如果有的话")


class ConcreteExecutor(LangGraphExecutor):
    """
    具体执行器实现，扩展LangGraphExecutor
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[BaseTool] = None,
                 input_schema: Type[BaseModel] = ExecutorInput,
                 output_schema: Type[BaseModel] = ExecutorOutput,
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化具体执行器
        
        Args:
            description: 执行器的描述
            llm: 用于驱动agent和tools的模型
            tools: langchain的tools列表
            input_schema: 输入参数的Pydantic模型类
            output_schema: 输出参数的Pydantic模型类
            checkpoint_dir: 检查点保存目录
        """
        super().__init__(description, llm, tools, input_schema, checkpoint_dir)
        self.output_schema = output_schema
        self.execution_history = []
        self.current_task_id = None
        
        # 创建执行工作流
        self.execution_workflow = self._create_execution_workflow()
    
    def _create_execution_workflow(self) -> StateGraph:
        """
        创建执行工作流
        
        Returns:
            LangGraph工作流
        """
        # 创建工作流
        workflow = StateGraph(GraphState)
        
        # 添加节点
        
        # 准备执行节点
        def prepare_execution(state: GraphState) -> GraphState:
            # 记录当前任务ID
            if "task_id" in state["input"]:
                self.current_task_id = state["input"]["task_id"]
            else:
                # 生成一个任务ID
                self.current_task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                state["input"]["task_id"] = self.current_task_id
            
            # 记录任务开始
            self.execution_history.append({
                "task_id": self.current_task_id,
                "start_time": datetime.now().isoformat(),
                "status": "started",
                "input": state["input"]
            })
            
            return state
        
        # 执行任务节点
        def execute_task(state: GraphState) -> GraphState:
            try:
                # 使用agent执行器执行任务
                if self.agent_executor:
                    result = self.agent_executor.invoke({
                        "input": json.dumps(state["input"], ensure_ascii=False)
                    })
                    state["output"] = {
                        "task_id": self.current_task_id,
                        "task_name": state["input"].get("task_name", "未命名任务"),
                        "status": "success",
                        "result": {"output": result.get("output")}
                    }
                else:
                    # 如果没有agent执行器，返回默认结果
                    state["output"] = {
                        "task_id": self.current_task_id,
                        "task_name": state["input"].get("task_name", "未命名任务"),
                        "status": "success",
                        "result": {"message": "执行完成，但没有具体实现"}
                    }
            except Exception as e:
                # 记录错误
                state["output"] = {
                    "task_id": self.current_task_id,
                    "task_name": state["input"].get("task_name", "未命名任务"),
                    "status": "failed",
                    "error": str(e)
                }
            
            return state
        
        # 完成执行节点
        def complete_execution(state: GraphState) -> GraphState:
            # 记录任务完成
            self.execution_history.append({
                "task_id": self.current_task_id,
                "end_time": datetime.now().isoformat(),
                "status": state["output"].get("status", "completed"),
                "output": state["output"]
            })
            
            # 更新执行器的输出
            self.output = state["output"]
            
            return state
        
        # 添加节点到工作流
        workflow.add_node("prepare_execution", prepare_execution)
        workflow.add_node("execute_task", execute_task)
        workflow.add_node("complete_execution", complete_execution)
        
        # 添加边
        workflow.add_edge("prepare_execution", "execute_task")
        workflow.add_edge("execute_task", "complete_execution")
        workflow.add_edge("complete_execution", END)
        
        # 设置入口节点
        workflow.set_entry_point("prepare_execution")
        
        return workflow
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            input_data: 输入参数
            
        Returns:
            执行结果
        """
        self.input = input_data
        
        # 如果需要验证输入
        if self.input_schema:
            try:
                validated_input = self.validateInput(self.input_schema)
                self.input = validated_input
            except ValidationError as e:
                # 如果验证失败，返回错误
                return {
                    "task_id": input_data.get("task_id", f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                    "task_name": input_data.get("task_name", "未命名任务"),
                    "status": "failed",
                    "error": f"输入验证失败: {str(e)}"
                }
        
        # 初始状态
        initial_state = {
            "input": self.input,
            "output": {},
            "messages": [],
            "current_step": "execute",
            "validation_errors": None,
            "validated_input": None,
            "plan": None
        }
        
        # 运行工作流
        final_state = self.execution_workflow.invoke(initial_state)
        
        # 验证输出
        if self.output_schema:
            try:
                validated_output = self.output_schema(**final_state["output"])
                self.output = validated_output.model_dump()
            except ValidationError as e:
                # 如果验证失败，返回错误
                self.output = {
                    "task_id": self.current_task_id,
                    "task_name": self.input.get("task_name", "未命名任务"),
                    "status": "failed",
                    "error": f"输出验证失败: {str(e)}"
                }
        
        # 保存检查点
        self.saveCheckpoint()
        
        return self.output
    
    def plan(self, input_data: Dict[str, Any]) -> Dict[str, 'Executor']:
        """
        生成执行计划
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务计划和对应的执行器映射
        """
        # 创建提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一个专业的任务规划器。
            
            你的职责是: {self.description}
            
            你需要根据用户的输入，生成一个任务计划。每个任务应该包含以下信息:
            1. 任务名称
            2. 任务描述
            3. 任务输入参数
            
            请以JSON格式返回任务计划。
            """),
            ("human", f"""
            请根据以下输入生成任务计划:
            
            {json.dumps(input_data, ensure_ascii=False, indent=2)}
            """)
        ])
        
        # 调用LLM
        response = self.llm.invoke(prompt)
        
        # 解析响应
        try:
            # 尝试从响应中提取JSON
            content = response.content
            # 查找JSON部分
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                tasks = json.loads(json_str)
                
                # 为每个任务分配执行器
                plan = {}
                for task_name, task_info in tasks.items():
                    # 查找合适的执行器
                    executor = self._find_executor_for_task(task_name, task_info)
                    if executor:
                        plan[task_name] = executor
                
                self.plan = plan
                return plan
            else:
                self.plan = {}
                return {}
        except Exception as e:
            self.plan = {}
            return {}
    
    def _find_executor_for_task(self, task_name: str, task_info: Dict[str, Any]) -> Optional[Executor]:
        """
        为任务查找合适的执行器
        
        Args:
            task_name: 任务名称
            task_info: 任务信息
            
        Returns:
            合适的执行器，如果没有找到则返回None
        """
        # 这里可以实现更复杂的匹配逻辑
        # 默认实现是简单地遍历所有执行器，查找描述匹配的
        task_name_lower = task_name.lower()
        
        for executor in self.executors:
            executor_desc = executor.description.lower()
            if task_name_lower in executor_desc:
                return executor
        
        # 如果没有找到完全匹配的，返回第一个执行器（如果有）
        return self.executors[0] if self.executors else None
    
    def saveCheckpoint(self) -> str:
        """
        保存当前执行器的状态到检查点
        
        Returns:
            保存的检查点文件路径
        """
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "executor_type": self.__class__.__name__,
            "description": self.description,
            "input": self.input,
            "plan": {k: v.__class__.__name__ for k, v in self.plan.items()},
            "output": self.output,
            "execution_history": self.execution_history,
            "summary": self.summary()
        }
        
        # 生成检查点文件名
        filename = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 保存检查点
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def summary(self) -> str:
        """
        使用LLM生成执行总结
        
        Returns:
            总结文本
        """
        # 创建提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的总结生成器。请根据提供的执行器信息生成一个简洁明了的总结。"),
            ("human", """
            请总结以下执行器的执行情况:
            
            执行器: {executor_type}
            描述: {description}
            输入: {input}
            计划: {plan}
            输出: {output}
            执行历史: {execution_history}
            """)
        ])
        
        # 准备输入
        input_data = {
            "executor_type": self.__class__.__name__,
            "description": self.description,
            "input": json.dumps(self.input, ensure_ascii=False, indent=2),
            "plan": ", ".join(self.plan.keys()) if self.plan else "无计划",
            "output": json.dumps(self.output, ensure_ascii=False, indent=2),
            "execution_history": json.dumps(self.execution_history, ensure_ascii=False, indent=2)
        }
        
        # 调用LLM
        response = self.llm.invoke(prompt.format(**input_data))
        
        return response.content
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Returns:
            执行历史列表
        """
        return self.execution_history


class ConcretePlanner(LangGraphPlanner):
    """
    具体计划器实现，扩展LangGraphPlanner
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[BaseTool] = None,
                 input_schema: Type[BaseModel] = None,
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化具体计划器
        
        Args:
            description: 计划器的描述
            llm: 用于驱动agent和tools的模型
            tools: langchain的tools列表
            input_schema: 输入参数的Pydantic模型类
            checkpoint_dir: 检查点保存目录
        """
        super().__init__(description, llm, tools, input_schema, checkpoint_dir)
        self.planning_history = []
        
    def generate_tasks(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成任务列表
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务列表
        """
        # 记录规划开始
        planning_record = {
            "start_time": datetime.now().isoformat(),
            "input": input_data,
            "status": "started"
        }
        
        # 调用父类方法生成任务
        tasks = super().generate_tasks(input_data)
        
        # 记录规划完成
        planning_record.update({
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "tasks": tasks
        })
        self.planning_history.append(planning_record)
        
        return tasks
    
    def plan(self, input_data: Dict[str, Any]) -> Dict[str, Executor]:
        """
        生成执行计划
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务计划和对应的执行器映射
        """
        # 调用父类方法生成计划
        plan = super().plan(input_data)
        
        # 记录计划
        self.planning_history[-1].update({
            "plan": {k: v.__class__.__name__ for k, v in plan.items()}
        })
        
        return plan
    
    def saveCheckpoint(self) -> str:
        """
        保存当前计划器的状态到检查点
        
        Returns:
            保存的检查点文件路径
        """
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "planner_type": self.__class__.__name__,
            "description": self.description,
            "input": self.input,
            "plan": {k: v.__class__.__name__ for k, v in self.plan.items()},
            "output": self.output,
            "planning_history": self.planning_history,
            "summary": self.summary()
        }
        
        # 生成检查点文件名
        filename = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 保存检查点
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def summary(self) -> str:
        """
        使用LLM生成规划总结
        
        Returns:
            总结文本
        """
        # 创建提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的总结生成器。请根据提供的计划器信息生成一个简洁明了的总结。"),
            ("human", """
            请总结以下计划器的规划情况:
            
            计划器: {planner_type}
            描述: {description}
            输入: {input}
            生成的计划: {plan}
            输出: {output}
            规划历史: {planning_history}
            """)
        ])
        
        # 准备输入
        input_data = {
            "planner_type": self.__class__.__name__,
            "description": self.description,
            "input": json.dumps(self.input, ensure_ascii=False, indent=2),
            "plan": ", ".join(self.plan.keys()) if self.plan else "无计划",
            "output": json.dumps(self.output, ensure_ascii=False, indent=2),
            "planning_history": json.dumps(self.planning_history, ensure_ascii=False, indent=2)
        }
        
        # 调用LLM
        response = self.llm.invoke(prompt.format(**input_data))
        
        return response.content
    
    def get_planning_history(self) -> List[Dict[str, Any]]:
        """
        获取规划历史
        
        Returns:
            规划历史列表
        """
        return self.planning_history


# 自定义工具基类
class CustomTool(BaseTool):
    """自定义工具基类"""
    
    def __init__(self, executor: Executor = None, **kwargs):
        """
        初始化自定义工具
        
        Args:
            executor: 关联的执行器
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.executor = executor
    
    def _run(self, *args, **kwargs) -> Any:
        """
        运行工具
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            工具运行结果
        """
        raise NotImplementedError("子类必须实现_run方法")


# 工具注册器
class ToolRegistry:
    """工具注册表，用于管理所有注册的工具"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            name: 工具名称
            tool: 工具实例
        """
        self.tools[name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例，如果不存在则返回None
        """
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """
        列出所有注册的工具
        
        Returns:
            工具名称列表
        """
        return list(self.tools.keys())
    
    def get_tools_for_executor(self, executor_type: str) -> List[BaseTool]:
        """
        获取适用于特定执行器类型的工具
        
        Args:
            executor_type: 执行器类型
            
        Returns:
            工具列表
        """
        return [tool for name, tool in self.tools.items() if executor_type.lower() in name.lower()]
