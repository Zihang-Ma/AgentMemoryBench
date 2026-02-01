from __future__ import annotations

from typing import Protocol, List, Dict, Any, Tuple


class ExecutionEngine(Protocol):
    """
    抽象的执行引擎接口。

    负责在「已经构造好的 messages + tools」基础上：
    - 选择合适的 agent（single / multi-agent）
    - 调用 LLM
    - 与 backend 进行多轮 /interact 交互
    - 返回完整的 history 和最终 result（reward / status / metric 等）
    """

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
        执行一条样本的完整交互流程。

        返回值：
        - history: OpenAI Chat 格式的完整对话轨迹（system/user/assistant/tool/...）
        - result: 由 backend 返回或汇总的最终结果（reward/status/metric 等）
        """


