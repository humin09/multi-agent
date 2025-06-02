"""
基于LangGraph和LangChain实现的Planner和Executor角色节点
"""
from typing import Dict, List, Any, Annotated, TypedDict, Callable, Type, Optional, Union
import json
import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint import MemorySaver

from base import Executor, Planner, AgentRegistry


class GraphState(TypedDict):
    """LangGraph状态类型定义"""
    input: Dict[str, Any]
    output: Dict[str, Any]
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    current_step: str
    validation_errors: Optional[List[str]]
    validated_input: Optional[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]


class LangGraphExecutor(Executor):
    """
    基于LangGraph实现的执行器
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[BaseTool] = None,
                 input_schema: Type[BaseModel] = None,
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化LangGraph执行器
        
        Args:
            description: 执行器的描述
            llm: 用于驱动agent和tools的模型
            tools: langchain的tools列表
            input_schema: 输入参数的Pydantic模型类
            checkpoint_dir: 检查点保存目录
        """
        super().__init__(description, llm, tools, checkpoint_dir)
        self.input_schema = input_schema
        self.agent_executor = None
        
        # 如果提供了工具和LLM，则创建agent执行器
        if self.tools and self.llm:
            self._create_agent_executor()
    
    def _create_agent_executor(self):
        """创建LangChain的Agent执行器"""
        # 创建系统提示
        system_prompt = f"""你是一个专业的执行器。
        
        你的职责是: {self.description}
        
        你需要根据用户的输入，使用提供的工具来完成任务。
        """
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # 创建agent
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # 创建agent执行器
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)
    
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
            validated_input = self.validateInput(self.input_schema)
            self.input = validated_input
        
        # 如果有agent执行器，则使用它执行任务
        if self.agent_executor:
            result = self.agent_executor.invoke({"input": json.dumps(self.input)})
            self.output = {"result": result.get("output")}
        else:
            # 默认实现，子类应该覆盖此方法
            self.output = {"message": "执行完成，但没有具体实现"}
        
        return self.output
    
    def plan(self, input_data: Dict[str, Any]) -> Dict[str, 'Executor']:
        """
        生成执行计划
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务计划和对应的执行器映射
        """
        # 默认实现，子类应该覆盖此方法
        self.plan = {}
        return self.plan
    
    def validateInput(self, input_schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        使用LangGraph实现的交互式输入验证
        
        Args:
            input_schema: 输入参数的Pydantic模型类
            
        Returns:
            验证后的输入参数
        """
        # 创建LangGraph工作流
        workflow = self._create_validation_workflow(input_schema)
        
        # 初始状态
        initial_state = {
            "input": self.input,
            "output": {},
            "messages": [],
            "current_step": "validate_input",
            "validation_errors": None,
            "validated_input": None,
            "plan": None
        }
        
        # 运行工作流
        final_state = workflow.invoke(initial_state)
        
        # 返回验证后的输入
        if final_state.get("validated_input"):
            return final_state["validated_input"]
        else:
            raise ValueError("输入验证失败")
    
    def _create_validation_workflow(self, input_schema: Type[BaseModel]) -> StateGraph:
        """
        创建输入验证工作流
        
        Args:
            input_schema: 输入参数的Pydantic模型类
            
        Returns:
            LangGraph工作流
        """
        # 创建工作流
        workflow = StateGraph(GraphState)
        
        # 添加节点
        
        # 验证输入节点
        def validate_input(state: GraphState) -> GraphState:
            try:
                # 尝试验证输入
                validated = input_schema(**state["input"])
                state["validated_input"] = validated.model_dump()
                state["validation_errors"] = None
            except ValidationError as e:
                # 记录验证错误
                state["validation_errors"] = [str(err) for err in e.errors()]
            
            return state
        
        # 生成提示节点
        def generate_prompt(state: GraphState) -> GraphState:
            if state["validation_errors"]:
                # 如果有验证错误，生成提示消息
                schema_json = json.dumps(input_schema.schema(), indent=2, ensure_ascii=False)
                error_msg = "\n".join(state["validation_errors"])
                
                prompt = f"""
                输入验证失败，请提供有效的输入参数。
                
                错误信息:
                {error_msg}
                
                请按照以下格式提供输入:
                {schema_json}
                
                当前输入:
                {json.dumps(state["input"], indent=2, ensure_ascii=False)}
                """
                
                state["messages"].append(HumanMessage(content=prompt))
            
            return state
        
        # LLM节点 - 使用LLM生成更好的输入建议
        def llm_suggest(state: GraphState) -> GraphState:
            if not state["messages"]:
                return state
                
            # 系统提示
            system_prompt = f"""你是一个输入验证助手。
            你的任务是帮助用户提供符合要求的输入参数。
            请根据验证错误和模式定义，生成有效的输入示例。
            """
            
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ])
            
            # 调用LLM
            response = self.llm.invoke(prompt.format(messages=state["messages"]))
            state["messages"].append(response)
            
            return state
        
        # 决策节点 - 决定是否需要用户输入
        def decide_next_step(state: GraphState) -> str:
            if state["validated_input"]:
                return "end"
            else:
                return "need_user_input"
        
        # 添加节点到工作流
        workflow.add_node("validate_input", validate_input)
        workflow.add_node("generate_prompt", generate_prompt)
        workflow.add_node("llm_suggest", llm_suggest)
        
        # 添加边
        workflow.add_edge("validate_input", "decide_next_step")
        workflow.add_conditional_edges(
            "decide_next_step",
            decide_next_step,
            {
                "end": END,
                "need_user_input": "generate_prompt"
            }
        )
        workflow.add_edge("generate_prompt", "llm_suggest")
        workflow.add_edge("llm_suggest", "validate_input")
        
        # 设置入口节点
        workflow.set_entry_point("validate_input")
        
        return workflow
    
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
            """)
        ])
        
        # 准备输入
        input_data = {
            "executor_type": self.__class__.__name__,
            "description": self.description,
            "input": json.dumps(self.input, ensure_ascii=False, indent=2),
            "plan": ", ".join(self.plan.keys()) if self.plan else "无计划",
            "output": json.dumps(self.output, ensure_ascii=False, indent=2)
        }
        
        # 调用LLM
        response = self.llm.invoke(prompt.format(**input_data))
        
        return response.content


class LangGraphPlanner(Planner):
    """
    基于LangGraph实现的计划器
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[BaseTool] = None,
                 input_schema: Type[BaseModel] = None,
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化LangGraph计划器
        
        Args:
            description: 计划器的描述
            llm: 用于驱动agent和tools的模型
            tools: langchain的tools列表
            input_schema: 输入参数的Pydantic模型类
            checkpoint_dir: 检查点保存目录
        """
        super().__init__(description, llm, tools, checkpoint_dir)
        self.input_schema = input_schema
        
        # 创建LangGraph工作流
        self.workflow = self._create_planning_workflow()
    
    def _create_planning_workflow(self) -> StateGraph:
        """
        创建计划工作流
        
        Returns:
            LangGraph工作流
        """
        # 创建工作流
        workflow = StateGraph(GraphState)
        
        # 添加节点
        
        # 验证输入节点
        def validate_input(state: GraphState) -> GraphState:
            if self.input_schema:
                try:
                    # 尝试验证输入
                    validated = self.input_schema(**state["input"])
                    state["validated_input"] = validated.model_dump()
                    state["validation_errors"] = None
                except ValidationError as e:
                    # 记录验证错误
                    state["validation_errors"] = [str(err) for err in e.errors()]
            else:
                # 如果没有提供模式，则直接使用输入
                state["validated_input"] = state["input"]
            
            return state
        
        # 生成计划节点
        def generate_plan(state: GraphState) -> GraphState:
            if state["validated_input"]:
                # 创建提示
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""你是一个专业的任务计划器。
                    
                    你的职责是: {self.description}
                    
                    你需要根据用户的输入，生成一个任务计划。每个任务应该包含以下信息:
                    1. 任务名称
                    2. 任务描述
                    3. 任务输入参数
                    
                    请以JSON格式返回任务计划。
                    """),
                    ("human", f"""
                    请根据以下输入生成任务计划:
                    
                    {json.dumps(state["validated_input"], ensure_ascii=False, indent=2)}
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
                        state["plan"] = tasks
                    else:
                        state["plan"] = {"error": "无法解析计划"}
                except Exception as e:
                    state["plan"] = {"error": f"计划生成失败: {str(e)}"}
            
            return state
        
        # 添加节点到工作流
        workflow.add_node("validate_input", validate_input)
        workflow.add_node("generate_plan", generate_plan)
        
        # 添加边
        workflow.add_edge("validate_input", "generate_plan")
        workflow.add_edge("generate_plan", END)
        
        # 设置入口节点
        workflow.set_entry_point("validate_input")
        
        return workflow
    
    def generate_tasks(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成任务列表
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务列表
        """
        # 初始状态
        initial_state = {
            "input": input_data,
            "output": {},
            "messages": [],
            "current_step": "generate_tasks",
            "validation_errors": None,
            "validated_input": None,
            "plan": None
        }
        
        # 运行工作流
        final_state = self.workflow.invoke(initial_state)
        
        # 从计划中提取任务
        if final_state.get("plan"):
            if isinstance(final_state["plan"], dict) and "error" not in final_state["plan"]:
                return [{"name": k, **v} for k, v in final_state["plan"].items()]
            elif isinstance(final_state["plan"], list):
                return final_state["plan"]
        
        return []
    
    def plan(self, input_data: Dict[str, Any]) -> Dict[str, Executor]:
        """
        生成执行计划
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务计划和对应的执行器映射
        """
        # 生成任务列表
        tasks = self.generate_tasks(input_data)
        
        # 为每个任务分配执行器
        plan = {}
        for task in tasks:
            task_name = task.get("name")
            if task_name:
                # 查找合适的执行器
                executor = self._find_executor_for_task(task)
                if executor:
                    plan[task_name] = executor
        
        self.plan = plan
        return plan
    
    def _find_executor_for_task(self, task: Dict[str, Any]) -> Optional[Executor]:
        """
        为任务查找合适的执行器
        
        Args:
            task: 任务信息
            
        Returns:
            合适的执行器，如果没有找到则返回None
        """
        # 这里可以实现更复杂的匹配逻辑
        # 默认实现是简单地遍历所有执行器，查找描述匹配的
        task_name = task.get("name", "").lower()
        
        for executor in self.executors:
            executor_desc = executor.description.lower()
            if task_name in executor_desc:
                return executor
        
        # 如果没有找到完全匹配的，返回第一个执行器（如果有）
        return self.executors[0] if self.executors else None
