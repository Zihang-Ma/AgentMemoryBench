"""
Execution engines for the lifelong-learning benchmark.

当前阶段实现：
- single_agent: 单一 LLM agent 跑完整个样本
- multi_agent_poll: 多个 agent 轮询执行任务（round-robin）
- multi_agent_vote: 多个 agent 同时执行同一样本，通过投票机制选择最终答案
- multi_agent_bandit: 多个 agent 使用多臂老虎机（MAB）算法动态选择最优 agent
"""

from .base import ExecutionEngine  # noqa: F401
# single_agent 是一个子包，实现位于 execution/single_agent/single_agent.py
from .single_agent.single_agent import SingleAgentExecutionEngine
# multi_agent_poll 是一个子包，实现位于 execution/multi_agent_poll/multi_agent_poll.py
from .multi_agent_poll.multi_agent_poll import MultiAgentPollExecutionEngine  # noqa: F401
# multi_agent_vote 是一个子包，实现位于 execution/multi_agent_vote/multi_agent_vote.py
from .multi_agent_vote.multi_agent_vote import MultiAgentVoteExecutionEngine  # noqa: F401
# multi_agent_bandit 是一个子包，实现位于 execution/multi_agent_bandit/multi_agent_bandit.py
from .multi_agent_bandit.multi_agent_bandit import MultiAgentBanditExecutionEngine  # noqa: F401


