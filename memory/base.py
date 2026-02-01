from __future__ import annotations

import json
import logging
import re
from typing import Protocol, List, Dict, Any, Optional, Type, TypeVar

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    logging.warning("json_repair not installed. Install with: pip install json-repair")

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    logging.warning("pydantic not installed. Install with: pip install pydantic")


T = TypeVar('T', bound='BaseModel')


def parse_llm_json_response(
    response_text: str,
    schema: Optional[Type[T]] = None,
    logger_prefix: str = "Memory"
) -> Optional[Dict[str, Any] | T]:
    """
    通用的 LLM JSON 响应解析器，支持 3 层渐进式容错：

    1. 输出清洗：移除 markdown 代码块、注释等，只保留 JSON 内容
    2. 智能修复：使用 json_repair 包自动修复常见格式错误（抢救 90% 残次品）
    3. Schema 校验：使用 Pydantic 验证数据类型和业务逻辑（可选）

    参数：
        response_text: LLM 返回的原始文本
        schema: Pydantic 模型类（可选），用于 Schema 校验和类型转换
        logger_prefix: 日志前缀（默认 "Memory"）

    返回：
        - 如果提供了 schema：返回 Pydantic 模型实例
        - 如果没有 schema：返回 Dict
        - 解析失败：返回 None

    示例：
        # 基础用法（只解析 JSON）
        result = parse_llm_json_response(llm_output)

        # 带 Schema 校验
        class MySchema(BaseModel):
            id: str
            event: str
            text: str

        result = parse_llm_json_response(llm_output, schema=MySchema)
    """
    logger = logging.getLogger(logger_prefix)

    # ========== 第 1 步：输出清洗 ==========
    cleaned = _clean_llm_output(response_text)
    logger.debug(f"[{logger_prefix}] Step 1: Output cleaning completed, length: {len(cleaned)}")

    # ========== 第 2 步：智能修复（json_repair）==========
    repaired = _smart_repair(cleaned, logger_prefix)
    if repaired is None:
        logger.warning(f"[{logger_prefix}] Step 2: Smart repair failed, trying fallback methods")
        # 如果 json_repair 失败，尝试传统方法（括号匹配）
        repaired = _extract_json_by_bracket_matching(cleaned, logger_prefix)

    if repaired is None:
        logger.error(f"[{logger_prefix}] Step 2: All repair methods failed")
        return None

    # ========== 第 3 步：Schema 校验（Pydantic）==========
    if schema is not None:
        validated = _validate_schema(repaired, schema, logger_prefix)
        if validated is None:
            logger.warning(f"[{logger_prefix}] Step 3: Schema validation failed")
            return None
        logger.debug(f"[{logger_prefix}] Step 3: Schema validation passed")
        return validated

    # 没有 schema，直接返回 dict
    logger.debug(f"[{logger_prefix}] Parsing completed (no schema validation)")
    return repaired


def _clean_llm_output(text: str) -> str:
    """第 1 步：输出清洗 - 移除 markdown、注释等"""
    # 移除 markdown 代码块标记
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*$', '', cleaned)

    # 移除 HTML 注释
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)

    # 移除 JavaScript 风格的单行注释（但保留 URL 中的 //）
    cleaned = re.sub(r'(?<!:)//[^\n]*', '', cleaned)

    # 移除 C 风格的多行注释
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

    return cleaned.strip()


def _smart_repair(text: str, logger_prefix: str) -> Optional[Dict[str, Any]]:
    """第 2 步：智能修复 - 使用 json_repair 抢救 90% 残次品"""
    logger = logging.getLogger(logger_prefix)

    if not HAS_JSON_REPAIR:
        logger.debug(f"[{logger_prefix}] json_repair not available, skipping smart repair")
        return None

    try:
        # 使用 json_repair 修复 JSON
        repaired_str = repair_json(text)
        # 尝试解析修复后的 JSON
        result = json.loads(repaired_str)
        logger.debug(f"[{logger_prefix}] Smart repair succeeded")
        return result
    except Exception as e:
        logger.debug(f"[{logger_prefix}] Smart repair failed: {e}")
        return None


def _extract_json_by_bracket_matching(text: str, logger_prefix: str) -> Optional[Dict[str, Any]]:
    """传统方法：使用括号匹配提取 JSON 对象"""
    logger = logging.getLogger(logger_prefix)

    try:
        # 查找第一个 { 的位置
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # 使用括号匹配找到对应的结束 }
        brace_count = 0
        in_string = False
        escape_next = False
        end_idx = start_idx

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

        if end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            result = json.loads(json_str)
            logger.debug(f"[{logger_prefix}] Bracket matching extraction succeeded")
            return result
    except Exception as e:
        logger.debug(f"[{logger_prefix}] Bracket matching extraction failed: {e}")

    return None


def _validate_schema(data: Dict[str, Any], schema: Type[T], logger_prefix: str) -> Optional[T]:
    """第 3 步：Schema 校验 - 使用 Pydantic 验证类型和业务逻辑"""
    logger = logging.getLogger(logger_prefix)

    if not HAS_PYDANTIC:
        logger.warning(f"[{logger_prefix}] pydantic not available, skipping schema validation")
        return data  # 返回原始 dict

    try:
        validated = schema(**data)
        return validated
    except ValidationError as e:
        logger.debug(f"[{logger_prefix}] Schema validation failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"[{logger_prefix}] Schema validation error: {e}")
        return None


class MemoryMechanism(Protocol):
    """
    抽象的记忆机制接口。

    - use_memory: 在调用 LLM 之前，用记忆改写 messages（例如插入 few-shot 或经验）
    - update_memory:  在样本结束后，根据完整 history 与结果更新记忆存储

    注意：这里不关心具体任务（DBBench / OS / KG / ALFWorld），只看统一的
    OpenAI Chat 格式：[{role, content, ...}, ...]。
    """

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于当前任务名和原始 messages，使用记忆返回增强后的 messages。

        对于 zero_shot，这里通常是直接透传。
        """

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """
        在单个样本执行结束后调用，用于把新的轨迹/结果写入记忆。

        对于 zero_shot，这里通常是 no-op。
        """


