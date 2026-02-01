"""
后端客户端模块
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import requests

from src.client.scheduler import SampleIndex


class BackendClient:
    """
    极简后端客户端，只封装 /start_sample 与 /list_workers。
    后续可以在这里继续封装 /interact /calculate_overall 等。
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def list_workers(self) -> Dict[str, Any]:
        url = f"{self.base_url}/list_workers"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_indices(self, task_name: str) -> List[SampleIndex]:
        url = f"{self.base_url}/get_indices"
        resp = requests.get(url, params={"name": task_name}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected indices response for task {task_name}: {data}")
        return list(data)

    def start_sample(self, task_name: str, index: int) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        调用 /start_sample，返回 (session_id, messages, tools)。
        当前后端实现返回的 TaskOutput 是一个 dict，包含 messages/tools。
        """
        url = f"{self.base_url}/start_sample"
        payload = {"name": task_name, "index": index}
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        # session_id 在响应头中
        session_id_header = resp.headers.get("session_id")
        if session_id_header is None:
            raise RuntimeError("Backend /start_sample did not return session_id in headers")
        try:
            session_id = int(session_id_header)
        except ValueError:
            raise RuntimeError(f"Invalid session_id header: {session_id_header}")

        data = resp.json()
        messages = data.get("messages", []) or []
        tools = data.get("tools", []) or []
        return session_id, messages, tools

    def interact(self, session_id: int, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        调用 /interact，将 agent 的回复发给后端环境控制器。

        按 AgentBench client 的约定：
        - session_id 通过 HTTP Header 传递；
        - JSON Body 只包含 {"messages": [...]}；
        - 返回值直接是 env_result（包含 finish/status/reward/messages 等字段）。
        """
        url = f"{self.base_url}/interact"
        headers = {"session_id": str(session_id)}
        payload = {"messages": messages}
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data
