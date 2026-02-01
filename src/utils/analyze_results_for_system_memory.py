"""
分析 benchmark 运行结果的脚本。

用法：
    python -m src.utils.analyze_results outputs/2025-12-05_19-45-36/dbbench-std
"""

import json
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

# 设置 Windows 控制台编码为 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def count_turns(history: List[Dict[str, Any]]) -> int:
    """计算对话轮数（assistant 消息数）"""
    return sum(1 for msg in history if msg.get("role") == "assistant")


def extract_error_type(error_msg: str) -> str:
    """从错误消息中提取错误类型"""
    if not error_msg:
        return "None"
    error_lower = error_msg.lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        return "Timeout"
    elif "400" in error_msg or "bad request" in error_lower:
        return "400 Bad Request"
    elif "429" in error_msg or "too many requests" in error_lower:
        return "429 Rate Limit"
    elif "500" in error_msg or "internal server error" in error_lower or "upstream error" in error_lower:
        return "500 Server Error"
    elif "connection" in error_lower or "connection aborted" in error_lower or "connectionreset" in error_lower:
        return "Connection Error"
    else:
        return "Other Error"


def analyze_error_cause(error_msg: str, history: List[Dict[str, Any]]) -> str:
    """分析错误原因"""
    error_lower = error_msg.lower()
    turn_count = count_turns(history)
    
    if "timeout" in error_lower or "timed out" in error_lower:
        if turn_count == 0:
            return "首次调用超时 - LLM API响应时间超过阈值（可能是服务端负载高或首次请求处理慢）"
        else:
            return f"第{turn_count+1}轮调用超时 - 可能是对话历史过长或服务端响应慢"
    elif "connection" in error_lower or "connection aborted" in error_lower:
        if "max retries exceeded" in error_lower:
            return "连接失败 - 无法建立到API服务器的连接（网络问题、DNS解析失败或服务不可用）"
        elif "connection aborted" in error_lower or "connectionreset" in error_lower:
            return "连接中断 - 远程主机强制关闭连接（可能是服务端主动断开或网络不稳定）"
        else:
            return "连接错误 - 网络连接问题"
    elif "500" in error_msg or "upstream error" in error_lower:
        return "服务端错误 - API服务端内部故障（upstream服务失败，已实现重试但服务端持续故障仍会失败）"
    elif "400" in error_msg:
        return "请求错误 - 客户端请求格式问题（已实现验证，但边缘情况可能仍存在）"
    else:
        return "未知错误"


def analyze_results(result_dir: Path) -> Dict[str, Any]:
    """分析结果目录中的所有 JSON 文件"""
    # 获取所有 JSON 文件，包括 .error.json 文件
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
                # 如果无法提取数字，跳过
                continue
        else:
            # 普通 JSON 文件
            try:
                num = int(p.stem)
                normal_files.append((num, p))
            except ValueError:
                # 如果无法转换为整数，跳过
                continue
    
    # 按数字排序
    json_files = [p for _, p in sorted(normal_files + error_files, key=lambda x: x[0])]
    
    if not json_files:
        print(f"[ERROR] 未找到 JSON 文件在: {result_dir}")
        return {}
    
    stats = {
        "total_samples": len(json_files),
        "completed": 0,
        "failed": 0,
        "reward_1": 0,
        "reward_0": 0,
        "no_reward": 0,
        "error_types": Counter(),
        "turn_counts": [],
        "error_samples": [],
        "success_samples": [],
        "task_name": None,
        "agent_name": None,
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
        
        # 统计状态
        status = result.get("status", "")
        if status == "completed":
            stats["completed"] += 1
        elif result.get("error"):
            stats["failed"] += 1
        
        # 统计 reward
        reward = result.get("reward")
        if reward == 1:
            stats["reward_1"] += 1
            stats["success_samples"].append(index)
        elif reward == 0:
            stats["reward_0"] += 1
        else:
            stats["no_reward"] += 1
        
        # 统计错误类型
        error = result.get("error")
        if error:
            error_type = extract_error_type(error)
            error_cause = analyze_error_cause(error, history)
            stats["error_types"][error_type] += 1
            stats["error_samples"].append({
                "index": index,
                "error_type": error_type,
                "error_cause": error_cause,
                "turn_count": count_turns(history),
                "error": error[:200] + "..." if len(error) > 200 else error,
            })
        
        # 统计轮数
        history = data.get("history", [])
        if history:
            turns = count_turns(history)
            stats["turn_counts"].append(turns)
    
    # 计算平均轮数
    if stats["turn_counts"]:
        stats["avg_turns"] = sum(stats["turn_counts"]) / len(stats["turn_counts"])
        stats["min_turns"] = min(stats["turn_counts"])
        stats["max_turns"] = max(stats["turn_counts"])
    else:
        stats["avg_turns"] = 0
        stats["min_turns"] = 0
        stats["max_turns"] = 0
    
    return stats


def print_report(stats: Dict[str, Any], result_dir: Path):
    """打印分析报告"""
    print("=" * 80)
    print("Benchmark 结果分析报告")
    print("=" * 80)
    print(f"\n结果目录: {result_dir}")
    print(f"任务: {stats.get('task_name', 'unknown')}")
    print(f"Agent: {stats.get('agent_name', 'unknown')}")
    
    print(f"\n{'─' * 80}")
    print("总体统计")
    print(f"{'─' * 80}")
    total = stats["total_samples"]
    print(f"总样本数: {total}")
    print(f"已完成: {stats['completed']} ({stats['completed']/total*100:.1f}%)")
    print(f"失败: {stats['failed']} ({stats['failed']/total*100:.1f}%)")
    
    print(f"\n{'─' * 80}")
    print("Reward 分布")
    print(f"{'─' * 80}")
    print(f"Reward = 1 (成功): {stats['reward_1']} ({stats['reward_1']/total*100:.1f}%)")
    print(f"Reward = 0 (失败): {stats['reward_0']} ({stats['reward_0']/total*100:.1f}%)")
    print(f"无 Reward: {stats['no_reward']} ({stats['no_reward']/total*100:.1f}%)")
    
    if stats["turn_counts"]:
        print(f"\n{'─' * 80}")
        print("对话轮数统计")
        print(f"{'─' * 80}")
        print(f"平均轮数: {stats['avg_turns']:.2f}")
        print(f"最少轮数: {stats['min_turns']}")
        print(f"最多轮数: {stats['max_turns']}")
    
    if stats["error_types"]:
        print(f"\n{'─' * 80}")
        print("错误类型分布")
        print(f"{'─' * 80}")
        for error_type, count in stats["error_types"].most_common():
            print(f"  {error_type}: {count} ({count/total*100:.1f}%)")
    
    if stats["error_samples"]:
        print(f"\n{'─' * 80}")
        print("错误样本详情（前 10 个）")
        print(f"{'─' * 80}")
        for i, sample in enumerate(stats["error_samples"][:10], 1):
            print(f"\n  {i}. Index {sample['index']} - {sample['error_type']}")
            print(f"     对话轮数: {sample.get('turn_count', 0)}")
            print(f"     错误原因: {sample.get('error_cause', '未知')}")
            print(f"     错误详情: {sample['error']}")
        if len(stats["error_samples"]) > 10:
            print(f"\n  ... 还有 {len(stats['error_samples']) - 10} 个错误样本")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("用法: python -m src.utils.analyze_results <结果目录>")
        print("示例: python -m src.utils.analyze_results outputs/2025-12-05_19-45-36/dbbench-std")
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

