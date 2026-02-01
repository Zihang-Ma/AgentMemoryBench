"""
Memory mechanisms for the lifelong-learning benchmark.

当前阶段只实现 zero_shot，用于占位和统一接口，后续会在此目录下扩展：
- stream_icl
- previous_sample_utilization
- agent_workflow_memory
- mem0
- context_compression
等。
"""

from .base import MemoryMechanism  # noqa: F401
# zero_shot 是子包，实现位于 memory/zero_shot/zero_shot.py
from .zero_shot.zero_shot import ZeroShotMemory  # noqa: F401
# context_compression 是子包，实现位于 memory/context_compression/context_compression.py
# from .context_compression.context_compression import ConversationCompressionMemory  # noqa: F401  # TODO: 未实现


