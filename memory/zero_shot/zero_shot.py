from __future__ import annotations

from typing import List, Dict, Any

import yaml

from ..base import MemoryMechanism


class ZeroShotMemory(MemoryMechanism):
    """
    Zero-shot 记忆机制：完全不利用历史经验，也不更新任何记忆文件。

    - use_memory: 原样透传后端返回的 messages
    - update_memory:  不做任何操作

    之所以仍然实现这个类，是为了和后续 few_shot 等机制
    共享同一套接口，便于在 runner 中通过配置文件动态切换。
    """

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 为避免外部误改原列表，这里返回一个浅拷贝
        return list(messages) if messages is not None else []

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        # Zero-shot 不记录任何经验，这里刻意留空
        return


def load_zero_shot_from_yaml(config_path: str) -> ZeroShotMemory:
    """
    从 memory/zero_shot/zero_shot.yaml 读取配置，构造 ZeroShotMemory。

    目前 zero_shot.yaml 只包含描述信息，因此这里只是验证文件存在 / 结构合法，
    返回一个简单的 ZeroShotMemory 实例，为后续扩展（例如为不同任务增加静态提示）预留位置。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        _ = yaml.safe_load(f) or {}
    return ZeroShotMemory()