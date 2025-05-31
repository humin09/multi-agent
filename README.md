# 多Agent框架使用说明文档

## 1. 框架概述

本框架基于LangGraph和LangChain实现了一个灵活的多Agent系统，支持父子Agent注册、任务分解、人机交互式输入验证和工具集成。框架的核心特性包括：

- **角色节点**：支持Planner和Executor两种角色节点
- **父子Agent注册**：子Agent可以注册到父Agent上，形成层次结构
- **任务分解**：Planner负责生成任务列表，并分配给对应的Executor执行
- **人机交互**：支持交互式输入验证，引导用户提供完整有效的输入
- **工具集成**：支持集成LangChain工具，如TavilySearch等
- **状态管理**：支持任务状态跟踪和检查点保存

## 2. 架构设计

框架采用了分层设计，主要包括以下几个层次：

1. **抽象层**：定义了Executor和Planner的抽象接口
2. **实现层**：基于LangGraph和LangChain实现了具体的执行器和计划器
3. **管理层**：提供了Agent管理和任务管理的功能
4. **应用层**：基于框架实现具体的应用，如旅游Agent系统

核心组件包括：

- **Executor**：执行器基类，定义了execute、plan等抽象方法
- **Planner**：计划器基类，继承自Executor，增加了generate_tasks方法
- **LangGraphExecutor/Planner**：基于LangGraph实现的执行器和计划器
- **ConcreteExecutor/Planner**：具体的执行器和计划器实现
- **InputValidator**：输入验证器，支持交互式输入验证
- **AgentManager**：Agent管理器，负责Agent的注册和查询
- **TaskManager**：任务管理器，负责任务的创建、分配和执行
- **MultiAgentSystem**：多Agent系统，集成了Agent管理和任务管理

## 3. 快速开始

### 3.1 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install langchain langchain-core langchain-community langgraph pydantic tavily-python langchain-openai
```

### 3.2 基本用法

```python
from langchain_openai import ChatOpenAI
from travel_agents import initialize_travel_agent_system

# 初始化LLM
llm = ChatOpenAI(temperature=0.7)

# 初始化旅游Agent系统
system = initialize_travel_agent_system(llm)

# 创建旅游任务
input_data = {
    "task_id": "travel_task_001",
    "task_name": "东京旅行规划",
    "destination": "东京",
    "start_date": "2023-10-01",
    "end_date": "2023-10-07",
    "travelers": 2,
    "budget": "中档",
    "interests": ["历史", "美食", "购物"]
}

# 规划并执行任务
result = system.plan_and_execute("TravelAgent", input_data)
print(result)
```

### 3.3 运行测试

```bash
# 设置OpenAI API密钥
export OPENAI_API_KEY=your_api_key

# 运行测试
python test_system.py
```

## 4. 核心API

### 4.1 Executor

```python
class Executor(ABC):
    def __init__(self, description: str, llm: Any, tools: List[Any] = None, checkpoint_dir: str = "./checkpoints"):
        # 初始化执行器
        
    def register_executor(self, executor: 'Executor') -> None:
        # 注册子执行器
        
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 执行任务的抽象方法
        
    @abstractmethod
    def plan(self, input_data: Dict[str, Any]) -> Dict[str, 'Executor']:
        # 生成执行计划的抽象方法
        
    def saveCheckpoint(self) -> str:
        # 保存当前执行器的状态到检查点
        
    def summary(self) -> str:
        # 总结执行器的输入、计划和输出
        
    def validateInput(self, input_schema: Type[BaseModel]) -> Dict[str, Any]:
        # 验证输入参数
```

### 4.2 Planner

```python
class Planner(Executor):
    def __init__(self, description: str, llm: Any, tools: List[Any] = None, checkpoint_dir: str = "./checkpoints"):
        # 初始化计划器
        
    @abstractmethod
    def generate_tasks(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 生成任务列表的抽象方法
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 执行计划器的方法
```

### 4.3 MultiAgentSystem

```python
class MultiAgentSystem:
    def __init__(self, llm: Any):
        # 初始化多Agent系统
        
    def create_agent(self, name: str, description: str, agent_type: str = "executor", parent_name: str = None,
                     input_schema: Dict[str, Any] = None, output_schema: Dict[str, Any] = None,
                     tools: List[str] = None) -> str:
        # 创建Agent
        
    def register_tool(self, name: str, tool: BaseTool) -> None:
        # 注册工具
        
    def create_task(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        # 创建任务
        
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        # 执行任务
        
    def plan_and_execute(self, planner_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 规划并执行任务
```

## 5. 自定义Agent

### 5.1 创建自定义Agent

```python
# 定义输入模型
class CustomAgentInput(BaseModel):
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    # 添加自定义字段
    custom_field: str = Field(..., description="自定义字段")

# 创建自定义Agent
def create_custom_agent(system: MultiAgentSystem, llm: Any) -> str:
    agent_id = system.create_agent(
        name="CustomAgent",
        description="自定义Agent，负责处理特定任务",
        agent_type="executor",
        input_schema={
            "custom_field": {"type": "string", "description": "自定义字段", "required": True}
        },
        tools=["tavily_search"]
    )
    return agent_id
```

### 5.2 创建自定义工具

```python
# 创建自定义工具
@tool
def custom_tool(param1: str, param2: int) -> str:
    """
    自定义工具
    
    Args:
        param1: 参数1
        param2: 参数2
        
    Returns:
        工具执行结果
    """
    # 工具实现
    return f"处理 {param1} 和 {param2} 的结果"

# 注册工具
system.register_tool("custom_tool", custom_tool)
```

## 6. 示例：旅游Agent系统

旅游Agent系统是一个基于本框架实现的示例应用，包括以下组件：

- **TravelAgent**：旅游规划Agent，负责规划旅游行程
- **FlightAgent**：机票预订Agent，负责搜索和推荐航班
- **HotelAgent**：酒店预订Agent，负责搜索和推荐酒店
- **TicketAgent**：门票预订Agent，负责搜索和推荐景点门票

### 6.1 使用示例

```python
# 初始化旅游Agent系统
system = initialize_travel_agent_system(llm)

# 创建旅游任务
input_data = {
    "task_id": "travel_task_001",
    "task_name": "东京旅行规划",
    "destination": "东京",
    "start_date": "2023-10-01",
    "end_date": "2023-10-07",
    "travelers": 2,
    "budget": "中档",
    "interests": ["历史", "美食", "购物"]
}

# 规划并执行任务
result = system.plan_and_execute("TravelAgent", input_data)
```

### 6.2 输出示例

```json
{
  "机票预订": {
    "task_id": "task_1234abcd",
    "task_name": "机票预订",
    "status": "success",
    "result": {
      "flights": [
        {
          "airline": "全日空航空",
          "flight_number": "NH123",
          "departure": "北京",
          "destination": "东京",
          "departure_time": "2023-10-01 08:30",
          "arrival_time": "2023-10-01 13:45",
          "price": "¥3,500"
        }
      ]
    }
  },
  "酒店预订": {
    "task_id": "task_5678efgh",
    "task_name": "酒店预订",
    "status": "success",
    "result": {
      "hotels": [
        {
          "name": "东京新大谷酒店",
          "address": "东京都千代田区纪尾井町4-1",
          "check_in": "2023-10-01",
          "check_out": "2023-10-07",
          "price": "¥1,200/晚",
          "rating": "4.5星"
        }
      ]
    }
  },
  "门票预订": {
    "task_id": "task_9012ijkl",
    "task_name": "门票预订",
    "status": "success",
    "result": {
      "tickets": [
        {
          "attraction": "东京迪士尼乐园",
          "date": "2023-10-03",
          "price": "¥500/人",
          "opening_hours": "09:00-21:00"
        },
        {
          "attraction": "浅草寺",
          "date": "2023-10-04",
          "price": "免费",
          "opening_hours": "06:00-17:00"
        }
      ]
    }
  }
}
```

## 7. 扩展与定制

### 7.1 添加新的Agent类型

1. 创建新的Agent类，继承自Executor或Planner
2. 实现必要的抽象方法
3. 在MultiAgentSystem中注册新的Agent类型

### 7.2 集成新的工具

1. 创建新的工具类，继承自BaseTool或使用@tool装饰器
2. 实现_run和_arun方法
3. 在MultiAgentSystem中注册新工具

### 7.3 自定义输入验证

1. 创建新的Pydantic模型类，定义输入字段和验证规则
2. 在创建Agent时指定input_schema参数
3. 使用validateInput方法进行输入验证

## 8. 最佳实践

1. **合理设计Agent层次结构**：将复杂任务分解为多个子任务，每个子任务由专门的Agent处理
2. **明确定义输入输出模式**：使用Pydantic模型明确定义每个Agent的输入和输出格式
3. **使用适当的工具**：为每个Agent配置合适的工具，提高任务处理能力
4. **保存检查点**：定期保存Agent状态，便于恢复和调试
5. **处理异常情况**：妥善处理输入验证失败、任务执行错误等异常情况
6. **优化提示模板**：为LLM提供清晰的指令和上下文，提高任务执行质量

## 9. 故障排除

1. **输入验证失败**：检查输入数据是否符合模式定义，确保必填字段已提供
2. **任务执行失败**：查看日志文件，检查错误信息，确保工具配置正确
3. **LLM响应不符合预期**：优化提示模板，提供更明确的指令和示例
4. **Agent注册失败**：检查Agent配置，确保父Agent存在且类型正确
5. **工具调用失败**：检查工具实现，确保参数类型和返回值符合预期

## 10. 参考资料

- [LangChain文档](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph文档](https://python.langchain.com/docs/langgraph)
- [Pydantic文档](https://docs.pydantic.dev/)
- [Tavily API文档](https://docs.tavily.com/)
# multi-agent
