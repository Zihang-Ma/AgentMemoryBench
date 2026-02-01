#!/usr/bin/env python3
"""
启发式估算每种任务平均需要的 token 数。

使用方法:
    python scripts/estimate_task_tokens.py [--output-dir OUTPUT_DIR]

参数:
    --output-dir: 输出目录路径（包含已执行的样本，包含完整的 history）
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    启发式估算 token 数。
    使用字符数 / 4（与 AWM 和 context_compression 一致）
    """
    return len(text) // 4


def format_messages_to_text(messages: List[Dict[str, Any]]) -> str:
    """将消息列表格式化为文本（用于估算 token）"""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts)


def estimate_sample_tokens_from_history(history: List[Dict[str, Any]]) -> int:
    """从 history 估算样本的 token 数"""
    text = format_messages_to_text(history)
    return estimate_tokens(text)


def load_samples_from_output_dir(output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    从输出目录加载已执行的样本（包含完整的 history）。
    这是最准确的方法，因为包含了实际的对话历史。
    """
    task_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    if not output_dir.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return task_samples
    
    # 遍历所有子目录（每个任务一个目录）
    for task_dir in output_dir.iterdir():
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        
        # 查找所有 JSON 文件（样本文件）
        json_files = sorted(
            [f for f in task_dir.glob("*.json")],
            key=lambda x: int(x.stem) if x.stem.isdigit() else 999999
        )
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "history" in data:
                        task_samples[task_name].append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        if task_samples[task_name]:
            logger.info(f"Loaded {len(task_samples[task_name])} samples for task: {task_name}")
    
    return task_samples


def calculate_task_statistics(task_samples: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """计算每个任务的 token 统计信息"""
    statistics = {}
    
    for task_name, samples in task_samples.items():
        if not samples:
            continue
        
        token_counts = []
        for sample in samples:
            history = sample.get("history", [])
            if history:
                tokens = estimate_sample_tokens_from_history(history)
                token_counts.append(tokens)
        
        if not token_counts:
            logger.warning(f"No valid samples for task: {task_name}")
            continue
        
        # 计算统计信息
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        median_tokens = sorted(token_counts)[len(token_counts) // 2]
        
        statistics[task_name] = {
            "avg": avg_tokens,
            "min": min_tokens,
            "max": max_tokens,
            "median": median_tokens,
            "count": len(token_counts)
        }
    
    return statistics


def print_statistics(statistics: Dict[str, Dict[str, float]]):
    """打印统计信息"""
    if not statistics:
        print("No statistics available.")
        return
    
    print("\n" + "=" * 80)
    print("Task Token Statistics (启发式估算: 字符数 / 4)")
    print("=" * 80)
    print(f"{'Task Name':<20} {'Count':<8} {'Avg':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
    print("-" * 80)
    
    # 按平均 token 数排序
    sorted_stats = sorted(statistics.items(), key=lambda x: x[1]["avg"], reverse=True)
    
    for task_name, stats in sorted_stats:
        print(f"{task_name:<20} {stats['count']:<8} {stats['avg']:<12.0f} {stats['min']:<12.0f} "
              f"{stats['max']:<12.0f} {stats['median']:<12.0f}")
    
    print("-" * 80)
    
    # 计算总体统计
    all_avgs = [s["avg"] for s in statistics.values()]
    overall_avg = sum(all_avgs) / len(all_avgs) if all_avgs else 0
    
    print(f"\nOverall Average: {overall_avg:.0f} tokens per sample")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="估算每种任务平均需要的 token 数（从输出目录）")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径（包含已执行的样本，包含完整的 history）"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="项目根目录（默认：脚本所在目录的父目录）"
    )
    
    args = parser.parse_args()
    
    # 确定项目根目录
    if args.root_dir:
        root_dir = Path(args.root_dir)
    else:
        # 假设脚本在 scripts/ 目录下
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent
    
    # 处理输出目录路径
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    
    logger.info(f"Loading samples from output directory: {output_dir}")
    
    # 加载样本
    task_samples = load_samples_from_output_dir(output_dir)
    
    if not task_samples:
        logger.error(f"No samples found in {output_dir}. Please check the directory path.")
        return
    
    # 计算统计信息
    statistics = calculate_task_statistics(task_samples)
    
    # 打印结果
    print_statistics(statistics)
    
    # 保存到文件
    output_file = root_dir / "task_token_statistics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    logger.info(f"\nStatistics saved to: {output_file}")


if __name__ == "__main__":
    main()

