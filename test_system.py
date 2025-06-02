"""
验证整体功能流程的主程序
"""
import os
import json
from typing import Dict, Any, List
import logging

from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from base import Executor, Planner, AgentRegistry
from langgraph_impl import LangGraphExecutor, LangGraphPlanner
from executor_impl import ConcreteExecutor, ConcretePlanner
from validation import EnhancedExecutor, InteractiveValidator
from agent_manager import MultiAgentSystem, AgentConfig
from travel_agents import initialize_travel_agent_system, run_travel_agent_example


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_multi_agent_system():
    """测试多Agent系统的功能"""
    logger.info("开始测试多Agent系统")
    
    # 初始化LLM
    # 注意：实际使用时需要设置环境变量OPENAI_API_KEY
    llm = ChatOpenAI(temperature=0.7)
    
    try:
        # 初始化旅游Agent系统
        logger.info("初始化旅游Agent系统")
        system = initialize_travel_agent_system(llm)
        
        # 列出所有Agent
        agents = system.list_agents()
        logger.info(f"系统中的Agent数量: {len(agents)}")
        for agent_config in agents:
            logger.info(f"Agent: {agent_config.name}, 类型: {agent_config.type}, 描述: {agent_config.description}")
        
        # 创建旅游任务
        logger.info("创建旅游任务")
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
        
        # 获取旅游Agent
        travel_agent = system.get_agent("TravelAgent")
        if not travel_agent:
            logger.error("未找到TravelAgent")
            return
        
        # 测试输入验证
        logger.info("测试输入验证")
        try:
            # 故意提供不完整的输入
            incomplete_input = {
                "task_id": "travel_task_002",
                "task_name": "不完整的旅行规划"
                # 缺少必填字段
            }
            
            # 这里应该会抛出异常
            if hasattr(travel_agent, "validateInput") and travel_agent.input_schema:
                travel_agent.input = incomplete_input
                travel_agent.validateInput(travel_agent.input_schema)
        except Exception as e:
            logger.info(f"预期的输入验证错误: {str(e)}")
        
        # 测试任务规划
        logger.info("测试任务规划")
        try:
            # 使用完整的输入
            task_id = system.create_task("TravelAgent", input_data)
            logger.info(f"创建的任务ID: {task_id}")
            
            # 规划任务
            system.task_manager.plan_task(task_id)
            
            # 获取任务信息
            task = system.task_manager.get_task(task_id)
            logger.info(f"任务状态: {task['status']}")
            logger.info(f"任务计划: {task.get('plan', {})}")
            
            # 执行计划
            logger.info("执行任务计划")
            results = system.task_manager.execute_plan(task_id)
            
            # 输出结果
            logger.info("任务执行完成")
            logger.info(f"结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
            
            return results
        except Exception as e:
            logger.error(f"任务执行错误: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        raise


def main():
    """主函数"""
    # 确保检查点目录存在
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 测试多Agent系统
    try:
        results = test_multi_agent_system()
        if results:
            # 保存结果到文件
            with open("test_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info("测试结果已保存到test_results.json")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")


if __name__ == "__main__":
    main()
