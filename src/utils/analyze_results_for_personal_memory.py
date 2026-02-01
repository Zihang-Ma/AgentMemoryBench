"""
分析 personal memory (locomo) benchmark 运行结果的脚本。

用法：
    python -m src.utils.analyze_results_for_personal_memory outputs/2025-12-28_12-58-09/locomo-0
"""

import json
import sys
import io
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# 设置 Windows 控制台编码为 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def analyze_results(result_dir: Path) -> Dict[str, Any]:
    """分析结果目录中的所有 JSON 文件"""
    # 获取所有 JSON 文件（排除 .error.json 文件）
    all_json_files = list(result_dir.glob("*.json"))
    
    # 分离普通 JSON 文件和 .error.json 文件
    normal_files = []
    error_files = []
    
    for p in all_json_files:
        if p.stem.endswith(".error"):
            # 提取数字部分（例如 "251.error" -> 251）
            try:
                num = int(p.stem.split(".")[0])
                error_files.append((num, p))
            except ValueError:
                continue
        else:
            # 普通 JSON 文件
            try:
                num = int(p.stem)
                normal_files.append((num, p))
            except ValueError:
                continue
    
    # 按数字排序
    json_files = [p for _, p in sorted(normal_files, key=lambda x: x[0])]
    
    if not json_files:
        print(f"[ERROR] 未找到 JSON 文件在: {result_dir}")
        return {}
    
    # 统计数据结构
    stats = {
        "total_samples": len(json_files),
        "failed_samples": len(error_files),
        "task_name": None,
        "agent_name": None,
        # 总体指标
        "f1_scores": [],
        "bleu_scores": [],
        "llm_scores": [],
        # 按 category 分组
        "by_category": defaultdict(lambda: {
            "count": 0,
            "f1_scores": [],
            "bleu_scores": [],
            "llm_scores": [],
        }),
    }
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARNING] 无法读取 {json_file.name}: {e}")
            continue
        
        result = data.get("result", {})
        index = result.get("index", data.get("index", json_file.stem))
        
        # 提取基本信息
        if stats["task_name"] is None:
            stats["task_name"] = result.get("task", data.get("task", "unknown"))
        if stats["agent_name"] is None:
            stats["agent_name"] = result.get("agent_name", "unknown")
        
        # 检查状态（status 可能在 result 中，也可能在顶层）
        status = result.get("status") or data.get("status", "")
        if status != "completed":
            continue
        
        # 提取 metrics
        metrics = result.get("metrics", {})
        if not metrics:
            continue
        
        f1_score = metrics.get("f1_score")
        bleu_score = metrics.get("bleu_score")
        llm_score = metrics.get("llm_score")
        
        # 只统计有效的分数（0-1 之间的数）
        if f1_score is not None and 0 <= f1_score <= 1:
            stats["f1_scores"].append(f1_score)
        if bleu_score is not None and 0 <= bleu_score <= 1:
            stats["bleu_scores"].append(bleu_score)
        if llm_score is not None and 0 <= llm_score <= 1:
            stats["llm_scores"].append(llm_score)
        
        # 按 category 分组统计
        category = result.get("category")
        if category is not None:
            cat_key = f"category_{category}"
            if f1_score is not None and 0 <= f1_score <= 1:
                stats["by_category"][cat_key]["f1_scores"].append(f1_score)
            if bleu_score is not None and 0 <= bleu_score <= 1:
                stats["by_category"][cat_key]["bleu_scores"].append(bleu_score)
            if llm_score is not None and 0 <= llm_score <= 1:
                stats["by_category"][cat_key]["llm_scores"].append(llm_score)
            stats["by_category"][cat_key]["count"] += 1
    
    # 计算总体平均值
    stats["avg_f1_score"] = sum(stats["f1_scores"]) / len(stats["f1_scores"]) if stats["f1_scores"] else 0.0
    stats["avg_bleu_score"] = sum(stats["bleu_scores"]) / len(stats["bleu_scores"]) if stats["bleu_scores"] else 0.0
    stats["avg_llm_score"] = sum(stats["llm_scores"]) / len(stats["llm_scores"]) if stats["llm_scores"] else 0.0
    
    # 计算按 category 的平均值
    for cat_key, cat_stats in stats["by_category"].items():
        cat_stats["avg_f1_score"] = sum(cat_stats["f1_scores"]) / len(cat_stats["f1_scores"]) if cat_stats["f1_scores"] else 0.0
        cat_stats["avg_bleu_score"] = sum(cat_stats["bleu_scores"]) / len(cat_stats["bleu_scores"]) if cat_stats["bleu_scores"] else 0.0
        cat_stats["avg_llm_score"] = sum(cat_stats["llm_scores"]) / len(cat_stats["llm_scores"]) if cat_stats["llm_scores"] else 0.0
    
    return stats


def print_report(stats: Dict[str, Any], result_dir: Path):
    """打印分析报告"""
    print("=" * 80)
    print("Personal Memory (Locomo) Benchmark 结果分析报告")
    print("=" * 80)
    print(f"\n结果目录: {result_dir}")
    print(f"任务: {stats.get('task_name', 'unknown')}")
    print(f"Agent: {stats.get('agent_name', 'unknown')}")
    
    print(f"\n{'─' * 80}")
    print("总体统计")
    print(f"{'─' * 80}")
    total = stats["total_samples"]
    failed = stats["failed_samples"]
    print(f"总样本数: {total}")
    print(f"失败样本数: {failed}")
    print(f"有效样本数: {total - failed}")
    
    print(f"\n{'─' * 80}")
    print("总体平均指标")
    print(f"{'─' * 80}")
    print(f"F1 Score 平均值: {stats['avg_f1_score']:.4f}")
    print(f"BLEU Score 平均值: {stats['avg_bleu_score']:.4f}")
    print(f"LLM Score 平均值: {stats['avg_llm_score']:.4f}")
    
    # 按 category 分组统计
    if stats["by_category"]:
        print(f"\n{'─' * 80}")
        print("按 Category 分组统计")
        print(f"{'─' * 80}")
        
        # 按 category 数字排序
        sorted_categories = sorted(
            stats["by_category"].items(),
            key=lambda x: int(x[0].split("_")[1]) if x[0].startswith("category_") else 0
        )
        
        for cat_key, cat_stats in sorted_categories:
            category_num = cat_key.split("_")[1] if cat_key.startswith("category_") else "?"
            print(f"\nCategory {category_num}:")
            print(f"  样本数: {cat_stats['count']}")
            print(f"  F1 Score 平均值: {cat_stats['avg_f1_score']:.4f}")
            print(f"  BLEU Score 平均值: {cat_stats['avg_bleu_score']:.4f}")
            print(f"  LLM Score 平均值: {cat_stats['avg_llm_score']:.4f}")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("用法: python -m src.utils.analyze_results_for_personal_memory <结果目录>")
        print("示例: python -m src.utils.analyze_results_for_personal_memory outputs/2025-12-28_12-58-09/locomo-0")
        sys.exit(1)
    
    result_dir = Path(sys.argv[1])
    if not result_dir.exists():
        print(f"[ERROR] 目录不存在: {result_dir}")
        sys.exit(1)
    
    stats = analyze_results(result_dir)
    if stats:
        print_report(stats, result_dir)
    else:
        print("[ERROR] 未能分析任何结果")


if __name__ == "__main__":
    main()

