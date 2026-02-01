from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml

from ..base import ExecutionEngine


@dataclass
class SingleAgentConfig:
    """
    单 agent 执行配置。

    当前阶段支持：
    - agent_name: 在 configs/llmapi/agent.yaml 中登记的名称（如 gpt-4o-mini）
    """

    agent_name: str = "gpt-4o-mini"


class SingleAgentExecutionEngine(ExecutionEngine):
    """
    最简单的执行引擎：对每条样本只用一个 agent，从头跑到尾。

    目前先实现一个占位版本：
    - 不与真实 LLM / /interact 打通，只返回初始 messages 和一个占位 result。
    - 后续会替换为：
        - 使用 agent_pool 调 LLM（支持 tools / tool_calls）
        - 使用 backend_client 与 /interact 多轮交互
    """

    def __init__(self, config: SingleAgentConfig | None = None) -> None:
        self.config = config or SingleAgentConfig()

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
        多轮执行流程（单 agent）：
        - 使用 agent_pool（LLM）基于当前 history + tools 生成 assistant 消息（含 tool_calls）
        - 将该 assistant 作为 agent_response 发给 backend_client.interact
        - 将 env_out 中的 messages（通常是 tool 消息）追加到 history
        - 根据 env_out.finish/status 判断是否继续

        目前不额外限制前端轮数，真正的 max_round 由后端 Task 中的 round_limit / max_step 控制。
        """
        history = list(messages) if messages is not None else []
        current_tools = list(tools) if tools is not None else []

        if agent_pool is None or not hasattr(agent_pool, "inference"):
            raise RuntimeError("SingleAgentExecutionEngine requires an agent_pool with an 'inference' method.")

        # 如果没有任何初始消息，直接返回错误，避免向 LLM 发送空 messages 触发 400/500
        if not history:
            result: Dict[str, Any] = {
                "task": task,
                "index": index,
                "agent_name": self.config.agent_name,
                "error": "No initial messages from backend; aborting to avoid invalid LLM request.",
                "note": "SingleAgentExecutionEngine: backend returned empty messages.",
            }
            return history, result

        while True:
            # 1) 规范化 history 用于 LLM（确保消息格式符合 OpenAI schema）
            llm_history = self._normalize_history_for_llm(history)

            try:
                # 2) 调用 LLM，获取 assistant 消息
                assistant_msg = agent_pool.inference(llm_history, current_tools)
            except Exception as e:
                result: Dict[str, Any] = {
                    "task": task,
                    "index": index,
                    "agent_name": self.config.agent_name,
                    "error": f"LLM inference failed: {e}",
                    "note": "SingleAgentExecutionEngine: LLM error during multi-turn execution.",
                }
                return history, result

            # 3) 简单规范化 assistant 消息（确保有 role 和 content）
            if not isinstance(assistant_msg, dict):
                assistant_msg = {"role": "assistant", "content": str(assistant_msg)}
            if "role" not in assistant_msg:
                assistant_msg["role"] = "assistant"
            if "content" not in assistant_msg:
                assistant_msg["content"] = ""

            # 4) 追加 assistant 消息到 history
            history.append(assistant_msg)

            # 5) 无论如何都发送到后端（让后端处理 turn 计数和错误提示）
            # 参考 AgentBench 和 LifelongAgentBench：每次 LLM 推理后都调用 /interact
            env_out = backend_client.interact(session_id, [assistant_msg])
            env_messages = env_out.get("messages", []) or []
            current_tools = env_out.get("tools", current_tools) or current_tools

            # 6) 追加后端返回的消息（通常是 tool 结果或 user 提示）
            for m in env_messages:
                history.append(m)

            # 7) 检查是否结束
            status = env_out.get("status")
            finish = env_out.get("finish", status != "RUNNING")
            reward = env_out.get("reward", 0)
            metric = env_out.get("metric", {})

            if finish:
                result: Dict[str, Any] = {
                    "task": task,
                    "index": index,
                    "agent_name": self.config.agent_name,
                    "status": status,
                    "reward": reward,
                    "metric": metric,
                    "note": "SingleAgentExecutionEngine: completed multi-turn execution via backend /interact.",
                }
                return history, result

    def _normalize_history_for_llm(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        确保发送给 LLM 的消息满足 OpenAI schema，避免因缺 content/role 报 400。
        - 补充缺失的 role 和 content 字段
        - 校验 tool_calls 的 JSON 格式
        - 确保 tool 消息对应正确的 tool_call_id
        """
        normalized: List[Dict[str, Any]] = []
        last_assistant_call_ids: set[str] = set()

        for m in history:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role is None:
                continue

            nm = dict(m)

            # tool 消息必须有 content，且必须对应上一条 assistant 的 tool_call_id
            if role == "tool":
                tcid = nm.get("tool_call_id")
                if tcid is None or (last_assistant_call_ids and tcid not in last_assistant_call_ids):
                    # 丢弃没有对应 tool_call 的 tool 消息
                    continue
                if nm.get("content") is None:
                    nm["content"] = ""
                normalized.append(nm)
                continue

            # assistant: content 为空时补空串；tool_calls 做基本的 JSON 校验
            if role == "assistant":
                if nm.get("content") is None:
                    nm["content"] = ""
                tool_calls = nm.get("tool_calls") or []
                cleaned_calls: List[Dict[str, Any]] = []
                for i, tc in enumerate(tool_calls):
                    func = tc.get("function") or {}
                    args_str = func.get("arguments")
                    name = func.get("name")
                    if not name:
                        continue
                    # 校验 arguments 是否为合法 JSON
                    try:
                        json.loads(args_str)
                    except Exception:
                        continue
                    tc_id = tc.get("id") or tc.get("tool_call_id") or f"call_hist_{i}"
                    cleaned_calls.append(
                        {
                            "id": tc_id,
                            "type": tc.get("type", "function"),
                            "function": {"name": name, "arguments": args_str},
                        }
                    )
                nm["tool_calls"] = cleaned_calls
                if not cleaned_calls and "tool_calls" in nm:
                    nm.pop("tool_calls", None)
                last_assistant_call_ids = {c["id"] for c in cleaned_calls}
            else:
                # 非 assistant 消息重置
                last_assistant_call_ids = set()

            normalized.append(nm)
        return normalized


def load_single_agent_engine_from_yaml(config_path: str) -> SingleAgentExecutionEngine:
    """
    从 execution/single_agent/single_agent.yaml 读取配置，构造 SingleAgentExecutionEngine。

    期望 YAML 结构示例：

    single_agent:
      agent:
        name: gpt-4o-mini
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sa_cfg = raw.get("single_agent", {}) or {}
    agent_cfg = sa_cfg.get("agent", {}) or {}
    agent_name = agent_cfg.get("name", "gpt-4o-mini")

    cfg = SingleAgentConfig(agent_name=agent_name)
    return SingleAgentExecutionEngine(cfg)


