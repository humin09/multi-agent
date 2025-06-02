"""
多Agent框架的核心抽象类和接口定义
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
from pydantic import BaseModel, Field
# 新增：将原方法内的导入提取到文件头部
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import os
from datetime import datetime

class Executor(ABC):
    """
    执行器基类，所有具体的执行器都应该继承这个类
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[Tool] = None,  # 修改：明确为LangChain的Tool列表
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化执行器
        
        Args:
            description: 执行器的描述，用于生成提示模板 
           llm: 用于驱动agent和tools的模型
            tools: LangChain的Tool列表（必须为Tool实例）
            checkpoint_dir: 检查点保存目录
        """
        # 新增：校验tools类型（如果传入非空列表）
        if tools is not None:
            for tool in tools:
                if not isinstance(tool, Tool):
                    raise ValueError(f"工具 {tool} 必须是 langchain.tools.Tool 的实例")
        
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.executors = []  # 注册的子执行器列表
        self.input = {}  # 输入参数
        self.output = {}  # 输出结果
        self.plan = {}  # 生成的计划和对应的执行器
        self.checkpoint_dir = checkpoint_dir
        
        # 确保检查点目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def register_executor(self, executor: 'Executor') -> None:
        """
        注册子执行器
        
        Args:
            executor: 要注册的子执行器
        """
        self.executors.append(executor)
    
    def register_executors(self, executors: List['Executor']) -> None:
        """
        批量注册子执行器
        
        Args:
            executors: 要注册的子执行器列表
        """
        self.executors.extend(executors)
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务的抽象方法，需要子类实现
        
        Args:
            input_data: 输入参数
            
        Returns:
            执行结果
        """
        pass
    

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
            "summary": self.summary()
        }
        
        # 生成检查点文件名
        filename = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 保存检查点
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
        return filepath
    
    def summary(self) -> str:
        """
        总结执行器的输入、计划和输出
        
        Returns:
            总结文本
        """
        # 这里可以使用LLM来生成总结
        # 默认实现是简单的字符串拼接
        summary = f"执行器: {self.__class__.__name__}\n"
        summary += f"描述: {self.description}\n"
        summary += f"输入: {json.dumps(self.input, ensure_ascii=False, indent=2)}\n"
        summary += f"计划: {', '.join(self.plan.keys())}\n"
        summary += f"输出: {json.dumps(self.output, ensure_ascii=False, indent=2)}\n"
        
        return summary
    
    def validateInput(self, input_schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        验证输入参数的基本实现，子类可以覆盖此方法提供更复杂的验证逻辑
        
        Args:
            input_schema: 输入参数的Pydantic模型类
            
        Returns:
            验证后的输入参数
        """
        # 这个方法将在子类中通过LangGraph实现更复杂的交互式验证
        # 基类提供一个简单的实现
        try:
            validated_input = input_schema(**self.input)
            return validated_input.model_dump()
        except Exception as e:
            raise ValueError(f"输入验证失败: {str(e)}")


class Planner(Executor):
    """
    计划器基类，负责生成任务列表和分配执行器
    """
    def __init__(self, 
                 description: str, 
                 llm: Any, 
                 tools: List[Any] = None,
                 checkpoint_dir: str = "./checkpoints"):
        """
        初始化计划器
        
        Args:
            description: 计划器的描述
            llm: 用于驱动agent和tools的模型
            tools: langchain的tools列表
            checkpoint_dir: 检查点保存目录
        """
        super().__init__(description, llm, tools, checkpoint_dir)
    
    @abstractmethod
    def generate_tasks(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成任务列表的抽象方法，需要子类实现
        
        Args:
            input_data: 输入参数
            
        Returns:
            任务列表
        """
        pass
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行计划器的方法，生成任务列表并分配给执行器执行
        
        Args:
            input_data: 输入参数
            
        Returns:
            执行结果
        """
        self.input = input_data
        
        # 生成任务列表
        tasks = self.generate_tasks(input_data)
        
        # 生成执行计划
        self.plan = self.plan(input_data)
        
        # 执行每个任务
        results = {}
        for task_name, executor in self.plan.items():
            # 找到对应任务的输入参数
            task_input = next((t for t in tasks if t.get('name') == task_name), {})
            # 执行任务
            results[task_name] = executor.execute(task_input)
        
        self.output = results
        return results


class AgentRegistry:
    """
    Agent注册表，用于管理所有注册的Agent
    """
    def __init__(self):
        self.agents = {}
    
    def register(self, name: str, agent: Executor) -> None:
        """
        注册Agent
        
        Args:
            name: Agent的名称
            agent: Agent实例
        """
        self.agents[name] = agent
    
    def get(self, name: str) -> Optional[Executor]:
        """
        获取Agent
        
        Args:
            name: Agent的名称
            
        Returns:
            Agent实例，如果不存在则返回None
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        列出所有注册的Agent
        
        Returns:
            Agent名称列表
        """
        return list(self.agents.keys())

    def generate_input_prompt(self, input_schema: Type[BaseModel]) -> str:
        """
        基于Pydantic模型生成用户输入提示（例如：需要提供哪些参数及其描述）
        
        Args:
            input_schema: Pydantic输入模型类
            
        Returns:
            用户输入提示文本
        """
        # 提取模型字段的描述（假设模型字段有`description`参数）
        fields = input_schema.model_fields
        field_descriptions = [
            f"- {name}: {field.description}" 
            for name, field in fields.items() 
            if field.description
        ]
        return (
            "请提供以下参数信息（用自然语言描述即可）：\n" + 
            "\n".join(field_descriptions) + 
            "\n输入后我会帮您提取并校验参数，若需要修改请直接说明，输入'忽略'可跳过当前步骤。"
        )

    def extract_parameters(self, user_message: str, input_schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        使用LLM从用户消息中提取符合Pydantic模型的参数
        
        Args:
            user_message: 用户输入的自然语言消息
            input_schema: Pydantic输入模型类
            
        Returns:
            提取的参数字典（可能未完全校验）
        """
        # 定义LLM提取参数的提示模板
        prompt = PromptTemplate(
            template="用户消息：{user_message}\n请从上述消息中提取以下参数（JSON格式）：{fields}\n提取结果：",
            input_variables=["user_message", "fields"]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # 获取模型字段列表（用于提示LLM）
        fields = list(input_schema.model_fields.keys())
        response = chain.run({
            "user_message": user_message,
            "fields": fields
        })
        
        # 解析LLM返回的JSON参数（简单示例，实际需处理解析异常）
        return json.loads(response)

    def interactive_validate(self, input_schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        交互式输入验证流程（用户输入→LLM提取→校验→用户确认）
        
        Args:
            input_schema: Pydantic输入模型类
            
        Returns:
            最终确认的参数字典
        """
        while True:
            # 步骤1：生成并展示输入提示
            prompt = self.generate_input_prompt(input_schema)
            print(prompt)
            
            # 步骤2：获取用户输入（模拟用户输入，实际需替换为真实输入获取逻辑）
            user_message = input("请输入参数信息：")  # 实际可对接聊天界面
            
            # 步骤3：用户选择忽略
            if user_message.strip().lower() == "忽略":
                return {}  # 或根据需求返回默认值/空值
            
            # 步骤4：LLM提取参数
            try:
                raw_params = self.extract_parameters(user_message, input_schema)
            except Exception as e:
                print(f"参数提取失败（{str(e)}），请重新描述信息。")
                continue
            
            # 步骤5：校验参数（复用现有validateInput方法）
            try:
                validated_params = self.validateInput(input_schema)
            except ValueError as e:
                print(f"参数校验失败：{str(e)}，请补充或修正信息。")
                continue
            
            # 步骤6：用户确认参数
            print("提取的参数如下，请确认（输入'确认'继续，其他内容重新输入）：")
            print(json.dumps(validated_params, indent=2, ensure_ascii=False))
            confirm = input("请输入确认信息：")
            
            if confirm.strip().lower() == "确认":
                return validated_params
            else:
                print("参数未确认，重新开始输入流程...")
