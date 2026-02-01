from __future__ import annotations

import json
import random
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from ..base import ExecutionEngine
from ..single_agent.single_agent import SingleAgentExecutionEngine, SingleAgentConfig


@dataclass
class MultiAgentBanditConfig:
    """
    Multi-agent bandit 执行配置。
    
    支持：
    - agents: 多个 agent 名称列表
    - bandit.method: bandit 算法（decaying_epsilon_greedy, upper_confidence_bound, thompson_sampling）
    - bandit.probability: epsilon-greedy 的初始探索概率
    - bandit.coefficient: UCB 的探索系数
    - memory: independent 或 shared
    """
    agent_names: List[str]
    bandit_method: str = "thompson_sampling"
    epsilon: float = 0.05  # 用于 decaying_epsilon_greedy
    ucb_coefficient: float = 1.0  # 用于 upper_confidence_bound
    memory_independent: bool = False
    warmup_steps: int = 0  # 前 warmup_steps 个样本使用 round-robin


class MultiAgentBanditExecutionEngine(ExecutionEngine):
    """
    Multi-agent bandit 执行引擎：使用多臂老虎机（MAB）算法动态选择最优 agent。
    
    实现参考 stream-bench 的 multi-agent bandit 机制：
    - 使用 bandit 算法（epsilon-greedy, UCB, Thompson Sampling）选择 agent
    - 根据样本执行结果（reward）更新每个 agent 的统计信息
    - 支持共享 memory（memory_independent=False）或独立 memory（memory_independent=True）
    - 每个 agent 使用 SingleAgentExecutionEngine 执行
    """
    
    def __init__(self, config: MultiAgentBanditConfig | None = None) -> None:
        self.config = config or MultiAgentBanditConfig(agent_names=["gpt-4o-mini"])
        self.K = len(self.config.agent_names)  # 多个 agent
        
        # Bandit 统计信息：S[i] = agent i 的成功次数，F[i] = agent i 的失败次数
        # 使用乐观初始化：给每个 agent 初始 S=1, F=0，鼓励早期探索
        self.S = np.ones(self.K, dtype=float)  # 成功次数（reward=1），乐观初始化：初始为 1
        self.F = np.zeros(self.K, dtype=float)  # 失败次数（reward=0），乐观初始化：初始为 0
        self.total_steps = 0  # 总样本数
        
        # 为每个 agent 创建 SingleAgentExecutionEngine
        self.agent_engines: List[SingleAgentExecutionEngine] = []
        for agent_name in self.config.agent_names:
            agent_cfg = SingleAgentConfig(agent_name=agent_name)
            engine = SingleAgentExecutionEngine(agent_cfg)
            self.agent_engines.append(engine)
    
    def _pull_agent(self) -> int:
        """
        使用 bandit 算法选择 agent。
        
        Returns:
            选中的 agent 索引
        """
        # Warmup 阶段：使用 round-robin
        if self.total_steps < self.config.warmup_steps:
            return self.total_steps % self.K
        
        # 根据 bandit 方法选择
        if self.config.bandit_method == "decaying_epsilon_greedy":
            return self._pull_epsilon_greedy()
        elif self.config.bandit_method == "upper_confidence_bound":
            return self._pull_ucb()
        elif self.config.bandit_method == "thompson_sampling":
            return self._pull_thompson_sampling()
        else:
            raise ValueError(f"Unknown bandit method: {self.config.bandit_method}")
    
    def _predict_agent(self) -> int:
        """
        预测性地选择 agent（不更新 total_steps），用于在 run_sample 之前获取选中的 agent。
        这个方法与 _pull_agent() 相同，但不会更新 total_steps（在 run_sample 中会更新）。
        
        Returns:
            选中的 agent 索引
        """
        # Warmup 阶段：使用 round-robin
        if self.total_steps < self.config.warmup_steps:
            return self.total_steps % self.K
        
        # 根据 bandit 方法选择（与 _pull_agent 相同）
        if self.config.bandit_method == "decaying_epsilon_greedy":
            return self._pull_epsilon_greedy()
        elif self.config.bandit_method == "upper_confidence_bound":
            return self._pull_ucb()
        elif self.config.bandit_method == "thompson_sampling":
            return self._pull_thompson_sampling()
        else:
            raise ValueError(f"Unknown bandit method: {self.config.bandit_method}")
    
    def _pull_epsilon_greedy(self) -> int:
        """
        Epsilon-greedy 算法（带衰减）。
        
        以概率 epsilon 探索（随机选择），以概率 1-epsilon 利用（选择平均 reward 最高的 agent）。
        epsilon 会随时间衰减：epsilon = initial_epsilon / sqrt(total_steps + 1)
        """
        # 计算衰减的 epsilon
        if self.total_steps == 0:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.epsilon / math.sqrt(self.total_steps + 1)
        
        if random.random() < epsilon:
            # 探索：随机选择
            return random.randint(0, self.K - 1)
        else:
            # 利用：选择平均 reward 最高的 agent
            # 由于使用了乐观初始化（S=1, F=0），所有 agent 初始 avg_reward = 1.0
            avg_rewards = []
            for i in range(self.K):
                total_trials = self.S[i] + self.F[i]
                avg_reward = self.S[i] / total_trials if total_trials > 0 else 1.0
                avg_rewards.append(avg_reward)
            
            # 如果有多个 agent 的平均 reward 相同，随机选择一个
            max_reward = max(avg_rewards)
            best_agents = [i for i, r in enumerate(avg_rewards) if r == max_reward]
            return random.choice(best_agents)
    
    def _pull_ucb(self) -> int:
        """
        Upper Confidence Bound (UCB) 算法。
        
        选择 UCB 值最大的 agent：
        UCB(i) = avg_reward(i) + coefficient * sqrt(ln(total_steps) / trials(i))
        """
        ucb_values = []
        for i in range(self.K):
            total_trials = self.S[i] + self.F[i]
            avg_reward = self.S[i] / total_trials if total_trials > 0 else 1.0
            # 避免 ln(0) 的情况
            if self.total_steps == 0:
                exploration = 0
            else:
                # 由于使用了乐观初始化，total_trials 至少为 1，不会出现除零错误
                exploration = self.config.ucb_coefficient * math.sqrt(
                    math.log(self.total_steps + 1) / total_trials
                )
            ucb_value = avg_reward + exploration
            ucb_values.append(ucb_value)
        
        # 选择 UCB 值最大的 agent
        max_ucb = max(ucb_values)
        best_agents = [i for i, ucb in enumerate(ucb_values) if ucb == max_ucb]
        return random.choice(best_agents)
    
    def _pull_thompson_sampling(self) -> int:
        """
        Thompson Sampling 算法。
        
        为每个 agent 维护 Beta 分布，从分布中采样，选择采样值最大的 agent。
        Beta(alpha, beta) = Beta(S[i] + 1, F[i] + 1)
        """
        # 从每个 agent 的 Beta 分布中采样
        theta_samples = []
        for i in range(self.K):
            # Beta(S[i] + 1, F[i] + 1)
            theta = np.random.beta(self.S[i] + 1, self.F[i] + 1)
            theta_samples.append(theta)
        
        # 选择采样值最大的 agent
        max_theta = max(theta_samples)
        best_agents = [i for i, theta in enumerate(theta_samples) if theta == max_theta]
        return random.choice(best_agents)
    
    def _update_bandit_stats(self, agent_index: int, reward: float) -> None:
        """
        更新 bandit 统计信息。
        
        Args:
            agent_index: 执行样本的 agent 索引
            reward: 样本的奖励（0 或 1，或 0-1 之间的值）
        """
        self.total_steps += 1
        
        # 将 reward 转换为成功/失败计数
        # reward = 1 表示成功，reward = 0 表示失败
        # 如果 reward 是 0-1 之间的值，按比例更新
        if reward >= 1.0:
            self.S[agent_index] += 1
        elif reward <= 0.0:
            self.F[agent_index] += 1
        else:
            # 部分奖励：按比例更新
            self.S[agent_index] += reward
            self.F[agent_index] += (1 - reward)
    
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
        执行一条样本：使用 bandit 算法选择 agent 执行。
        
        注意：agent_pool 在这里应该是一个字典，包含每个 agent_name 对应的 SimpleHTTPChatAgent 实例。
        """
        # 使用 bandit 算法选择 agent
        selected_agent_index = self._pull_agent()
        selected_agent_name = self.config.agent_names[selected_agent_index]
        selected_engine = self.agent_engines[selected_agent_index]
        
        # 显示 bandit 选择信息
        is_warmup = self.total_steps < self.config.warmup_steps
        warmup_info = f" (warmup: {self.total_steps}/{self.config.warmup_steps})" if is_warmup else ""
        print(f"  -> Bandit selection (method={self.config.bandit_method}, step={self.total_steps}{warmup_info}):")
        print(f"     Selected agent: {selected_agent_name} (index={selected_agent_index})")
        
        # 显示所有 agent 的统计信息
        print(f"     Agent statistics:")
        for i, agent_name in enumerate(self.config.agent_names):
            total_trials = self.S[i] + self.F[i]
            avg_reward = self.S[i] / total_trials if total_trials > 0 else 1.0
            # 判断是否已经实际尝试过（total_trials > 1 表示已经尝试过，因为乐观初始化是 S=1, F=0）
            if total_trials > 1:
                print(f"       - {agent_name}: S={self.S[i]:.1f}, F={self.F[i]:.1f}, trials={total_trials:.1f}, avg_reward={avg_reward:.3f}")
            else:
                print(f"       - {agent_name}: S={self.S[i]:.1f}, F={self.F[i]:.1f}, trials={total_trials:.1f}, avg_reward={avg_reward:.3f} (optimistic init, not tried yet)")
        
        # 如果 agent_pool 是字典，获取对应的 agent
        if isinstance(agent_pool, dict):
            if selected_agent_name not in agent_pool:
                raise RuntimeError(
                    f"Agent '{selected_agent_name}' not found in agent_pool. "
                    f"Available agents: {list(agent_pool.keys())}"
                )
            selected_agent = agent_pool[selected_agent_name]
        else:
            # 如果 agent_pool 不是字典，假设它是单个 agent（向后兼容）
            selected_agent = agent_pool
        
        # 使用选中的 agent engine 执行
        history, result = selected_engine.run_sample(
            task=task,
            index=index,
            session_id=session_id,
            messages=messages,
            tools=tools,
            agent_pool=selected_agent,
            backend_client=backend_client,
        )
        
        # 从 result 中提取 reward（0 或 1）
        reward = 0.0
        if isinstance(result, dict):
            # 优先使用 reward 字段
            if "reward" in result:
                reward = float(result.get("reward", 0))
            # 如果没有 reward，根据 status 判断
            elif "status" in result:
                status = result.get("status", "")
                if status == "completed" or status == "success":
                    reward = 1.0
                else:
                    reward = 0.0
            # 如果都没有，默认为 0
        
        # 更新 bandit 统计信息
        self._update_bandit_stats(selected_agent_index, reward)
        
        # 显示更新后的统计信息
        total_trials = self.S[selected_agent_index] + self.F[selected_agent_index]
        avg_reward = self.S[selected_agent_index] / total_trials if total_trials > 0 else 0.0
        print(f"  -> Bandit update: {selected_agent_name} received reward={reward:.1f}")
        print(f"     Updated stats: S={self.S[selected_agent_index]:.1f}, F={self.F[selected_agent_index]:.1f}, trials={total_trials:.1f}, avg_reward={avg_reward:.3f}")
        print(f"     Total steps: {self.total_steps}")
        
        # 在 result 中记录使用的 agent 和 bandit 信息
        if isinstance(result, dict):
            result["agent_name"] = selected_agent_name
            result["agent_index"] = selected_agent_index
            result["bandit_method"] = self.config.bandit_method
            result["bandit_stats"] = {
                "S": float(self.S[selected_agent_index]),
                "F": float(self.F[selected_agent_index]),
                "total_trials": float(total_trials),
                "avg_reward": float(avg_reward),
            }
            result["total_steps"] = self.total_steps
        
        return history, result


def load_multi_agent_bandit_engine_from_yaml(config_path: str) -> MultiAgentBanditExecutionEngine:
    """
    从 execution/multi_agent_bandit/multi_agent_bandit.yaml 读取配置，构造 MultiAgentBanditExecutionEngine。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    mab_cfg = raw.get("multi_agent_bandit", {}) or {}
    agents_cfg = mab_cfg.get("agents", []) or []
    bandit_cfg = mab_cfg.get("bandit", {}) or {}
    
    agent_names = [agent.get("name") for agent in agents_cfg if "name" in agent]
    if not agent_names:
        raise ValueError("No agents specified in multi_agent_bandit configuration")
    
    bandit_method = bandit_cfg.get("method", "thompson_sampling")
    # 处理 method 名称中的连字符
    if bandit_method == "decaying epsilon-greedy" or bandit_method == "decaying_epsilon_greedy":
        bandit_method = "decaying_epsilon_greedy"
    elif bandit_method == "upper confidence bound" or bandit_method == "upper_confidence_bound":
        bandit_method = "upper_confidence_bound"
    elif bandit_method == "thompson sampling" or bandit_method == "thompson_sampling":
        bandit_method = "thompson_sampling"
    
    epsilon = bandit_cfg.get("probability", 0.05)
    ucb_coefficient = bandit_cfg.get("coefficient", 1.0)
    memory_independent = mab_cfg.get("memory", "shared") == "independent"
    warmup_steps = bandit_cfg.get("warmup_steps", 0)
    
    cfg = MultiAgentBanditConfig(
        agent_names=agent_names,
        bandit_method=bandit_method,
        epsilon=epsilon,
        ucb_coefficient=ucb_coefficient,
        memory_independent=memory_independent,
        warmup_steps=warmup_steps,
    )
    return MultiAgentBanditExecutionEngine(cfg)

