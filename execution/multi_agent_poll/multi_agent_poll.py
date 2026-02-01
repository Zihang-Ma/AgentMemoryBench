from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml

from ..base import ExecutionEngine
from ..single_agent.single_agent import SingleAgentExecutionEngine, SingleAgentConfig


@dataclass
class MultiAgentPollConfig:
    """
    Multi-agent poll 执行配置。
    
    支持：
    - agents: 多个 agent 名称列表
    - polling.interval: 每完成多少个样本切换一次 agent（轮询间隔）
    - polling.independent: 是否每个 agent 使用独立的 memory（False 则共享 memory）
    """
    agent_names: List[str]
    interval: int = 1
    independent: bool = False


class MultiAgentPollExecutionEngine(ExecutionEngine):
    """
    Multi-agent poll 执行引擎：多个 agent 轮询执行任务。
    
    实现参考 stream-bench 的 round-robin 机制：
    - 每完成 interval 个样本，切换到下一个 agent
    - 支持共享 memory（independent=False）或独立 memory（independent=True）
    - 每个 agent 使用 SingleAgentExecutionEngine 执行
    """
    
    def __init__(self, config: MultiAgentPollConfig | None = None) -> None:
        self.config = config or MultiAgentPollConfig(agent_names=["gpt-4o-mini"])
        self.K = len(self.config.agent_names)  # 多个 agent
        self.cur_arm = 0  # 当前使用的 agent 索引
        self.sample_count = 0  # 已完成的样本计数（用于轮询）
        
        # 为每个 agent 创建 SingleAgentExecutionEngine
        self.agent_engines: List[SingleAgentExecutionEngine] = []
        for agent_name in self.config.agent_names:
            agent_cfg = SingleAgentConfig(agent_name=agent_name)
            engine = SingleAgentExecutionEngine(agent_cfg)
            self.agent_engines.append(engine)
    
    def _get_current_agent_engine(self) -> SingleAgentExecutionEngine:
        """获取当前应该使用的 agent engine（根据轮询机制）"""
        return self.agent_engines[self.cur_arm]
    
    def _update_polling(self) -> None:
        """
        更新轮询状态：每完成 interval 个样本，切换到下一个 agent。
        参考 stream-bench 的 pull_rr() 方法。
        """
        self.sample_count += 1
        if self.sample_count % self.config.interval == 0:
            # 切换到下一个 agent（round-robin）
            self.cur_arm = (self.cur_arm + 1) % self.K
    
    def run_sample(
        self,
        task: str,
        index: int,
        session_id: int,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        agent_pool: Any,
        backend_client: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行一条样本：使用当前轮询到的 agent 执行。
        
        注意：agent_pool 在这里应该是一个字典，包含每个 agent_name 对应的 SimpleHTTPChatAgent 实例。
        如果 agent_pool 是 None 或不是字典，则使用当前 agent_engine 的 agent_name 从 agent_pool 获取。
        """
        # 获取当前应该使用的 agent engine
        current_engine = self._get_current_agent_engine()
        current_agent_name = self.config.agent_names[self.cur_arm]
        
        # 如果 agent_pool 是字典，获取对应的 agent
        if isinstance(agent_pool, dict):
            if current_agent_name not in agent_pool:
                raise RuntimeError(
                    f"Agent '{current_agent_name}' not found in agent_pool. "
                    f"Available agents: {list(agent_pool.keys())}"
                )
            current_agent = agent_pool[current_agent_name]
        else:
            # 如果 agent_pool 不是字典，假设它是单个 agent（向后兼容）
            current_agent = agent_pool
        
        # 保存当前的 agent_index（在更新之前）
        current_agent_index = self.cur_arm
        
        # 使用当前 agent engine 执行
        history, result = current_engine.run_sample(
            task=task,
            index=index,
            session_id=session_id,
            messages=messages,
            tools=tools,
            agent_pool=current_agent,
            backend_client=backend_client,
        )
        
        # 更新轮询状态（在样本执行完成后）
        self._update_polling()
        
        # 在 result 中记录使用的 agent（使用更新前的 agent_index，但使用更新后的 sample_count）
        if isinstance(result, dict):
            result["agent_name"] = current_agent_name
            result["agent_index"] = current_agent_index
            result["sample_count"] = self.sample_count
        
        return history, result


def load_multi_agent_poll_engine_from_yaml(config_path: str) -> MultiAgentPollExecutionEngine:
    """
    从 execution/multi_agent_poll/multi_agent_poll.yaml 读取配置，构造 MultiAgentPollExecutionEngine。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    map_cfg = raw.get("multi_agent_poll", {}) or {}
    agents_cfg = map_cfg.get("agents", []) or []
    polling_cfg = map_cfg.get("polling", {}) or {}
    
    agent_names = [agent.get("name") for agent in agents_cfg if "name" in agent]
    if not agent_names:
        raise ValueError("No agents specified in multi_agent_poll configuration")
    
    interval = polling_cfg.get("interval", 1)
    independent = polling_cfg.get("independent", False)
    
    cfg = MultiAgentPollConfig(
        agent_names=agent_names,
        interval=interval,
        independent=independent,
    )
    return MultiAgentPollExecutionEngine(cfg)

