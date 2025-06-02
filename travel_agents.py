"""
初始化旅游Agent及其子Agent，并集成TavilySearch工具
"""
from typing import Dict, List, Any, Type, Optional, Union
import json
import os
from datetime import datetime
import inspect

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from base import Executor, Planner
from executor_impl import ConcreteExecutor, ConcretePlanner
from agent_manager import MultiAgentSystem, AgentConfig


# 机票Agent输入模型
class FlightAgentInput(BaseModel):
    """机票Agent的输入参数模型"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    departure: str = Field(..., description="出发地")
    destination: str = Field(..., description="目的地")
    departure_date: str = Field(..., description="出发日期，格式为YYYY-MM-DD")
    return_date: Optional[str] = Field(None, description="返回日期，格式为YYYY-MM-DD，单程票可不填")
    passengers: int = Field(1, description="乘客人数")
    cabin_class: str = Field("Economy", description="舱位等级，如Economy、Business、First")


# 酒店Agent输入模型
class HotelAgentInput(BaseModel):
    """酒店Agent的输入参数模型"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    location: str = Field(..., description="入住地址")
    check_in_date: str = Field(..., description="入住日期，格式为YYYY-MM-DD")
    check_out_date: str = Field(..., description="退房日期，格式为YYYY-MM-DD")
    guests: int = Field(1, description="入住人数")
    rooms: int = Field(1, description="房间数量")
    hotel_class: Optional[str] = Field(None, description="酒店星级，如3星级、4星级、5星级")


# 门票Agent输入模型
class TicketAgentInput(BaseModel):
    """门票Agent的输入参数模型"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    attraction: str = Field(..., description="旅游景点")
    visit_date: str = Field(..., description="游玩日期，格式为YYYY-MM-DD")
    visitors: int = Field(1, description="游玩人数")
    ticket_type: Optional[str] = Field("Standard", description="门票类型，如Standard、VIP")


# 旅游Agent输入模型
class TravelAgentInput(BaseModel):
    """旅游Agent的输入参数模型"""
    task_id: str = Field(..., description="任务ID")
    task_name: str = Field(..., description="任务名称")
    destination: str = Field(..., description="旅游目的地")
    start_date: str = Field(..., description="开始日期，格式为YYYY-MM-DD")
    end_date: str = Field(..., description="结束日期，格式为YYYY-MM-DD")
    travelers: int = Field(1, description="旅行人数")
    budget: Optional[str] = Field(None, description="预算范围，如经济型、中档、豪华")
    interests: Optional[List[str]] = Field(None, description="兴趣爱好，如历史、自然、美食等")


# TavilySearch工具
class TavilySearchTool(BaseTool):
    """TavilySearch工具，用于检索外部网站信息"""
    
    name = "tavily_search"
    description = "使用Tavily搜索引擎检索外部网站信息"
    
    def __init__(self, api_key: str = None):
        """
        初始化TavilySearch工具
        
        Args:
            api_key: Tavily API密钥
        """
        super().__init__()
        self.search = TavilySearchAPIWrapper(tavily_api_key=api_key)
    
    def _run(self, query: str) -> str:
        """
        运行工具
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        return self.search.results(query)
    
    async def _arun(self, query: str) -> str:
        """
        异步运行工具
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        return await self.search.aresults(query)


# 机票搜索工具
@tool
def search_flights(departure: str, destination: str, departure_date: str, return_date: Optional[str] = None, passengers: int = 1, cabin_class: str = "Economy") -> str:
    """
    搜索航班信息
    
    Args:
        departure: 出发地
        destination: 目的地
        departure_date: 出发日期，格式为YYYY-MM-DD
        return_date: 返回日期，格式为YYYY-MM-DD，单程票可不填
        passengers: 乘客人数
        cabin_class: 舱位等级，如Economy、Business、First
        
    Returns:
        航班搜索结果
    """
    # 构建搜索查询
    query = f"从{departure}到{destination}的航班，出发日期{departure_date}"
    if return_date:
        query += f"，返回日期{return_date}"
    query += f"，{passengers}名乘客，{cabin_class}舱"
    
    # 使用TavilySearch搜索
    search = TavilySearchAPIWrapper()
    results = search.results(query)
    
    return json.dumps(results, ensure_ascii=False, indent=2)


# 酒店搜索工具
@tool
def search_hotels(location: str, check_in_date: str, check_out_date: str, guests: int = 1, rooms: int = 1, hotel_class: Optional[str] = None) -> str:
    """
    搜索酒店信息
    
    Args:
        location: 入住地址
        check_in_date: 入住日期，格式为YYYY-MM-DD
        check_out_date: 退房日期，格式为YYYY-MM-DD
        guests: 入住人数
        rooms: 房间数量
        hotel_class: 酒店星级，如3星级、4星级、5星级
        
    Returns:
        酒店搜索结果
    """
    # 构建搜索查询
    query = f"{location}的酒店，入住日期{check_in_date}，退房日期{check_out_date}，{guests}名客人，{rooms}间房"
    if hotel_class:
        query += f"，{hotel_class}"
    
    # 使用TavilySearch搜索
    search = TavilySearchAPIWrapper()
    results = search.results(query)
    
    return json.dumps(results, ensure_ascii=False, indent=2)


# 门票搜索工具
@tool
def search_tickets(attraction: str, visit_date: str, visitors: int = 1, ticket_type: str = "Standard") -> str:
    """
    搜索景点门票信息
    
    Args:
        attraction: 旅游景点
        visit_date: 游玩日期，格式为YYYY-MM-DD
        visitors: 游玩人数
        ticket_type: 门票类型，如Standard、VIP
        
    Returns:
        门票搜索结果
    """
    # 构建搜索查询
    query = f"{attraction}的门票，游玩日期{visit_date}，{visitors}名游客，{ticket_type}票"
    
    # 使用TavilySearch搜索
    search = TavilySearchAPIWrapper()
    results = search.results(query)
    
    return json.dumps(results, ensure_ascii=False, indent=2)


# 初始化旅游Agent系统
def initialize_travel_agent_system(llm: Any) -> MultiAgentSystem:
    """
    初始化旅游Agent系统
    
    Args:
        llm: 用于驱动agent的模型
        
    Returns:
        多Agent系统
    """
    # 创建多Agent系统
    system = MultiAgentSystem(llm)
    
    # 注册工具
    tavily_search = TavilySearchTool()
    system.register_tool("tavily_search", tavily_search)
    system.register_tool("search_flights", search_flights)
    system.register_tool("search_hotels", search_hotels)
    system.register_tool("search_tickets", search_tickets)
    
    # 创建旅游Agent（Planner）
    travel_agent_id = system.create_agent(
        name="TravelAgent",
        description="旅游规划Agent，负责规划旅游行程，包括机票、酒店和景点门票",
        agent_type="planner",
        input_schema={
            "destination": {"type": "string", "description": "旅游目的地", "required": True},
            "start_date": {"type": "string", "description": "开始日期，格式为YYYY-MM-DD", "required": True},
            "end_date": {"type": "string", "description": "结束日期，格式为YYYY-MM-DD", "required": True},
            "travelers": {"type": "integer", "description": "旅行人数", "required": True},
            "budget": {"type": "string", "description": "预算范围，如经济型、中档、豪华", "required": False},
            "interests": {"type": "array", "description": "兴趣爱好，如历史、自然、美食等", "required": False}
        },
        tools=["tavily_search"]
    )
    
    # 创建机票Agent（Executor）
    flight_agent_id = system.create_agent(
        name="FlightAgent",
        description="机票预订Agent，负责搜索和推荐航班",
        agent_type="executor",
        parent_name="TravelAgent",
        input_schema={
            "departure": {"type": "string", "description": "出发地", "required": True},
            "destination": {"type": "string", "description": "目的地", "required": True},
            "departure_date": {"type": "string", "description": "出发日期，格式为YYYY-MM-DD", "required": True},
            "return_date": {"type": "string", "description": "返回日期，格式为YYYY-MM-DD", "required": False},
            "passengers": {"type": "integer", "description": "乘客人数", "required": True},
            "cabin_class": {"type": "string", "description": "舱位等级，如Economy、Business、First", "required": False}
        },
        tools=["tavily_search", "search_flights"]
    )
    
    # 创建酒店Agent（Executor）
    hotel_agent_id = system.create_agent(
        name="HotelAgent",
        description="酒店预订Agent，负责搜索和推荐酒店",
        agent_type="executor",
        parent_name="TravelAgent",
        input_schema={
            "location": {"type": "string", "description": "入住地址", "required": True},
            "check_in_date": {"type": "string", "description": "入住日期，格式为YYYY-MM-DD", "required": True},
            "check_out_date": {"type": "string", "description": "退房日期，格式为YYYY-MM-DD", "required": True},
            "guests": {"type": "integer", "description": "入住人数", "required": True},
            "rooms": {"type": "integer", "description": "房间数量", "required": True},
            "hotel_class": {"type": "string", "description": "酒店星级，如3星级、4星级、5星级", "required": False}
        },
        tools=["tavily_search", "search_hotels"]
    )
    
    # 创建门票Agent（Executor）
    ticket_agent_id = system.create_agent(
        name="TicketAgent",
        description="门票预订Agent，负责搜索和推荐景点门票",
        agent_type="executor",
        parent_name="TravelAgent",
        input_schema={
            "attraction": {"type": "string", "description": "旅游景点", "required": True},
            "visit_date": {"type": "string", "description": "游玩日期，格式为YYYY-MM-DD", "required": True},
            "visitors": {"type": "integer", "description": "游玩人数", "required": True},
            "ticket_type": {"type": "string", "description": "门票类型，如Standard、VIP", "required": False}
        },
        tools=["tavily_search", "search_tickets"]
    )
    
    return system


# 示例使用
def run_travel_agent_example(llm: Any) -> Dict[str, Any]:
    """
    运行旅游Agent示例
    
    Args:
        llm: 用于驱动agent的模型
        
    Returns:
        执行结果
    """
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
    
    return result
