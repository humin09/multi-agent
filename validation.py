"""
实现validateInput的人机交互流程
"""
from typing import Dict, List, Any, Type, Optional, Union, Callable, TypeVar, Generic
import json
import os
from datetime import datetime
import inspect
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


# 定义输入验证状态
class ValidationState(TypedDict):
    """输入验证状态"""
    input: Dict[str, Any]
    schema: Dict[str, Any]
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    validation_errors: Optional[List[str]]
    validated_input: Optional[Dict[str, Any]]
    missing_fields: List[str]
    invalid_fields: Dict[str, str]
    current_field: Optional[str]
    is_valid: bool
    user_message: Optional[str]
    llm_response: Optional[str]


# 泛型类型变量
T = TypeVar('T', bound=BaseModel)


class InputValidator(Generic[T]):
    """
    输入验证器，用于实现人机交互式的输入验证
    """
    def __init__(self, 
                 input_schema: Type[T], 
                 llm: Any,
                 max_retries: int = 3):
        """
        初始化输入验证器
        
        Args:
            input_schema: 输入参数的Pydantic模型类
            llm: 用于驱动交互的模型
            max_retries: 最大重试次数
        """
        self.input_schema = input_schema
        self.llm = llm
        self.max_retries = max_retries
        self.workflow = self._create_validation_workflow()
    
    def _create_validation_workflow(self) -> StateGraph:
        """
        创建输入验证工作流
        
        Returns:
            LangGraph工作流
        """
        # 创建工作流
        workflow = StateGraph(ValidationState)
        
        # 添加节点
        
        # 初始化验证状态
        def initialize_validation(state: ValidationState) -> ValidationState:
            # 获取模式信息
            schema = self.input_schema.schema()
            
            # 初始化状态
            state["schema"] = schema
            state["validation_errors"] = []
            state["validated_input"] = None
            state["missing_fields"] = []
            state["invalid_fields"] = {}
            state["current_field"] = None
            state["is_valid"] = False
            
            # 添加系统消息
            if not state.get("messages"):
                state["messages"] = [
                    SystemMessage(content=f"""你是一个输入验证助手。
                    你的任务是帮助用户提供符合要求的输入参数。
                    请根据验证错误和模式定义，引导用户提供有效的输入。
                    """)
                ]
            
            return state
        
        # 验证输入
        def validate_input(state: ValidationState) -> ValidationState:
            try:
                # 尝试验证输入
                validated = self.input_schema(**state["input"])
                state["validated_input"] = validated.model_dump()
                state["validation_errors"] = []
                state["is_valid"] = True
                state["missing_fields"] = []
                state["invalid_fields"] = {}
            except ValidationError as e:
                # 记录验证错误
                errors = e.errors()
                state["validation_errors"] = [str(err) for err in errors]
                state["is_valid"] = False
                
                # 分析错误
                missing_fields = []
                invalid_fields = {}
                
                for err in errors:
                    field = err["loc"][0] if err["loc"] else None
                    if field:
                        if err["type"] == "missing":
                            missing_fields.append(field)
                        else:
                            invalid_fields[field] = err["msg"]
                
                state["missing_fields"] = missing_fields
                state["invalid_fields"] = invalid_fields
                
                # 选择当前需要处理的字段
                if missing_fields:
                    state["current_field"] = missing_fields[0]
                elif invalid_fields:
                    state["current_field"] = list(invalid_fields.keys())[0]
            
            return state
        
        # 生成提示消息
        def generate_prompt(state: ValidationState) -> ValidationState:
            if not state["is_valid"]:
                # 获取模式信息
                schema = state["schema"]
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                # 生成字段描述
                field_descriptions = []
                for field_name, field_info in properties.items():
                    is_required = field_name in required
                    field_type = field_info.get("type", "any")
                    field_desc = field_info.get("description", "")
                    
                    status = ""
                    if field_name in state["missing_fields"]:
                        status = "【缺失】"
                    elif field_name in state["invalid_fields"]:
                        status = f"【无效: {state['invalid_fields'][field_name]}】"
                    
                    req_mark = "（必填）" if is_required else "（可选）"
                    field_descriptions.append(f"- {field_name}{req_mark}: {field_desc} {status}")
                
                # 生成当前需要的字段信息
                current_field = state["current_field"]
                current_field_info = ""
                if current_field:
                    field_info = properties.get(current_field, {})
                    field_type = field_info.get("type", "any")
                    field_desc = field_info.get("description", "")
                    is_required = current_field in required
                    
                    current_field_info = f"""
                    当前需要填写的字段: {current_field}
                    类型: {field_type}
                    描述: {field_desc}
                    是否必填: {"是" if is_required else "否"}
                    """
                
                # 生成提示消息
                prompt = f"""
                请提供有效的输入参数。
                
                字段列表:
                {chr(10).join(field_descriptions)}
                
                {current_field_info}
                
                当前输入:
                {json.dumps(state["input"], indent=2, ensure_ascii=False)}
                
                请提供完整的输入或补充缺失/无效的字段。
                """
                
                # 添加到消息列表
                state["messages"].append(HumanMessage(content=prompt))
            
            return state
        
        # LLM生成建议
        def llm_suggest(state: ValidationState) -> ValidationState:
            if not state["is_valid"] and state["messages"]:
                # 调用LLM
                response = self.llm.invoke(state["messages"])
                state["messages"].append(response)
                state["llm_response"] = response.content
            
            return state
        
        # 等待用户输入
        def wait_user_input(state: ValidationState) -> ValidationState:
            # 这个节点在实际应用中会等待用户输入
            # 在这里我们假设用户输入已经通过state["user_message"]提供
            if state.get("user_message"):
                # 尝试解析用户输入
                try:
                    # 首先尝试解析为JSON
                    user_input = json.loads(state["user_message"])
                    # 更新输入
                    state["input"].update(user_input)
                except json.JSONDecodeError:
                    # 如果不是JSON，尝试解析为单个字段的值
                    if state["current_field"]:
                        state["input"][state["current_field"]] = state["user_message"]
                
                # 添加用户消息
                state["messages"].append(HumanMessage(content=state["user_message"]))
                # 清除用户消息，避免重复处理
                state["user_message"] = None
            
            return state
        
        # 决策节点 - 决定下一步
        def decide_next_step(state: ValidationState) -> str:
            if state["is_valid"]:
                return "end"
            else:
                return "continue_validation"
        
        # 添加节点到工作流
        workflow.add_node("initialize", initialize_validation)
        workflow.add_node("validate", validate_input)
        workflow.add_node("generate_prompt", generate_prompt)
        workflow.add_node("llm_suggest", llm_suggest)
        workflow.add_node("wait_user_input", wait_user_input)
        
        # 添加边
        workflow.add_edge("initialize", "validate")
        workflow.add_conditional_edges(
            "validate",
            decide_next_step,
            {
                "end": END,
                "continue_validation": "generate_prompt"
            }
        )
        workflow.add_edge("generate_prompt", "llm_suggest")
        workflow.add_edge("llm_suggest", "wait_user_input")
        workflow.add_edge("wait_user_input", "validate")
        
        # 设置入口节点
        workflow.set_entry_point("initialize")
        
        return workflow
    
    def validate(self, input_data: Dict[str, Any], user_messages: List[str] = None) -> Dict[str, Any]:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            user_messages: 用户消息列表，用于模拟交互
            
        Returns:
            验证后的输入数据
        """
        # 初始状态
        state: ValidationState = {
            "input": input_data,
            "schema": {},
            "messages": [],
            "validation_errors": None,
            "validated_input": None,
            "missing_fields": [],
            "invalid_fields": {},
            "current_field": None,
            "is_valid": False,
            "user_message": None,
            "llm_response": None
        }
        
        # 如果没有用户消息，直接运行工作流
        if not user_messages:
            final_state = self.workflow.invoke(state)
            if final_state["is_valid"]:
                return final_state["validated_input"]
            else:
                raise ValidationError(final_state["validation_errors"])
        
        # 模拟交互
        current_state = self.workflow.invoke(state)
        
        # 最大重试次数
        retries = 0
        
        # 处理用户消息
        for msg in user_messages:
            if current_state["is_valid"] or retries >= self.max_retries:
                break
                
            # 更新用户消息
            current_state["user_message"] = msg
            
            # 继续工作流
            current_state = self.workflow.continue_from(current_state)
            
            retries += 1
        
        # 检查最终状态
        if current_state["is_valid"]:
            return current_state["validated_input"]
        else:
            raise ValidationError(current_state["validation_errors"])


class InteractiveValidator:
    """
    交互式验证器，用于实现完整的人机交互验证流程
    """
    def __init__(self, 
                 llm: Any,
                 max_retries: int = 3):
        """
        初始化交互式验证器
        
        Args:
            llm: 用于驱动交互的模型
            max_retries: 最大重试次数
        """
        self.llm = llm
        self.max_retries = max_retries
        self.validators = {}
    
    def register_schema(self, name: str, schema: Type[BaseModel]) -> None:
        """
        注册验证模式
        
        Args:
            name: 模式名称
            schema: Pydantic模型类
        """
        self.validators[name] = InputValidator(schema, self.llm, self.max_retries)
    
    def validate(self, schema_name: str, input_data: Dict[str, Any], user_messages: List[str] = None) -> Dict[str, Any]:
        """
        验证输入数据
        
        Args:
            schema_name: 模式名称
            input_data: 输入数据
            user_messages: 用户消息列表，用于模拟交互
            
        Returns:
            验证后的输入数据
        """
        validator = self.validators.get(schema_name)
        if not validator:
            raise ValueError(f"未找到名为 {schema_name} 的验证模式")
        
        return validator.validate(input_data, user_messages)
    
    def get_validation_prompt(self, schema_name: str, input_data: Dict[str, Any]) -> str:
        """
        获取验证提示
        
        Args:
            schema_name: 模式名称
            input_data: 输入数据
            
        Returns:
            验证提示
        """
        validator = self.validators.get(schema_name)
        if not validator:
            raise ValueError(f"未找到名为 {schema_name} 的验证模式")
        
        # 初始状态
        state: ValidationState = {
            "input": input_data,
            "schema": {},
            "messages": [],
            "validation_errors": None,
            "validated_input": None,
            "missing_fields": [],
            "invalid_fields": {},
            "current_field": None,
            "is_valid": False,
            "user_message": None,
            "llm_response": None
        }
        
        # 运行工作流的前几个节点
        workflow = validator.workflow
        
        # 手动运行节点
        state = workflow.get_node("initialize")(state)
        state = workflow.get_node("validate")(state)
        
        if not state["is_valid"]:
            state = workflow.get_node("generate_prompt")(state)
            state = workflow.get_node("llm_suggest")(state)
            
            # 返回LLM的建议
            return state["llm_response"] if state["llm_response"] else "请提供有效的输入参数。"
        else:
            return "输入验证通过。"


# 增强版执行器，集成交互式验证
class EnhancedExecutor(Executor):
    """
    增强版执行器，集成交互式验证
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[BaseTool] = None,
                 input_schema: Type[BaseModel] = None,
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化增强版执行器
        
        Args:
            description: 执行器的描述
            llm: 用于驱动agent和tools的模型
            tools: langchain的tools列表
            input_schema: 输入参数的Pydantic模型类
            checkpoint_dir: 检查点保存目录
        """
        super().__init__(description, llm, tools, checkpoint_dir)
        self.input_schema = input_schema
        self.validator = InteractiveValidator(llm) if llm else None
        
        # 注册验证模式
        if self.validator and self.input_schema:
            self.validator.register_schema(self.__class__.__name__, self.input_schema)
    
    def validateInput(self, input_schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        验证输入参数
        
        Args:
            input_schema: 输入参数的Pydantic模型类
            
        Returns:
            验证后的输入参数
        """
        if not self.validator:
            # 如果没有验证器，使用基本验证
            return super().validateInput(input_schema)
        
        try:
            # 使用交互式验证器验证输入
            return self.validator.validate(self.__class__.__name__, self.input)
        except ValidationError as e:
            # 如果验证失败，获取验证提示
            prompt = self.validator.get_validation_prompt(self.__class__.__name__, self.input)
            
            # 在实际应用中，这里应该与用户交互
            # 在这个示例中，我们只是抛出异常
            raise ValueError(f"输入验证失败: {prompt}")
    
    def interactive_validate(self, user_messages: List[str] = None) -> Dict[str, Any]:
        """
        交互式验证输入
        
        Args:
            user_messages: 用户消息列表，用于模拟交互
            
        Returns:
            验证后的输入参数
        """
        if not self.validator or not self.input_schema:
            raise ValueError("未配置验证器或输入模式")
        
        return self.validator.validate(self.__class__.__name__, self.input, user_messages)


# 人机交互工具
class UserInteractionTool(BaseTool):
    """用户交互工具，用于实现人机交互"""
    
    name = "user_interaction"
    description = "与用户交互，获取用户输入"
    
    def __init__(self, validator: InteractiveValidator = None, **kwargs):
        """
        初始化用户交互工具
        
        Args:
            validator: 交互式验证器
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.validator = validator
    
    def _run(self, schema_name: str, prompt: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        运行工具
        
        Args:
            schema_name: 模式名称
            prompt: 提示消息
            input_data: 输入数据
            
        Returns:
            用户输入
        """
        # 在实际应用中，这里应该与用户交互
        # 在这个示例中，我们只是返回一个模拟的用户输入
        return {"message": "这是模拟的用户输入"}
    
    async def _arun(self, schema_name: str, prompt: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        异步运行工具
        
        Args:
            schema_name: 模式名称
            prompt: 提示消息
            input_data: 输入数据
            
        Returns:
            用户输入
        """
        return self._run(schema_name, prompt, input_data)


# 创建人机交互工作流
def create_interactive_validation_workflow(
    input_schema: Type[BaseModel],
    llm: Any,
    max_retries: int = 3
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    创建人机交互验证工作流
    
    Args:
        input_schema: 输入参数的Pydantic模型类
        llm: 用于驱动交互的模型
        max_retries: 最大重试次数
        
    Returns:
        验证工作流函数
    """
    # 创建验证器
    validator = InputValidator(input_schema, llm, max_retries)
    
    # 创建工作流函数
    def validation_workflow(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证工作流
        
        Args:
            input_data: 输入数据
            
        Returns:
            验证后的输入数据
        """
        # 初始状态
        state: ValidationState = {
            "input": input_data,
            "schema": {},
            "messages": [],
            "validation_errors": None,
            "validated_input": None,
            "missing_fields": [],
            "invalid_fields": {},
            "current_field": None,
            "is_valid": False,
            "user_message": None,
            "llm_response": None
        }
        
        # 运行工作流
        final_state = validator.workflow.invoke(state)
        
        if final_state["is_valid"]:
            return final_state["validated_input"]
        else:
            # 在实际应用中，这里应该与用户交互
            # 在这个示例中，我们只是抛出异常
            raise ValidationError(final_state["validation_errors"])
    
    return validation_workflow
