#!/usr/bin/env python3
"""
计算 replay 模式下每次 replay 时 train 和 test 的 reward>0 的占比，并绘制散点图。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def extract_reward_from_json(json_path: Path) -> float:
    """从 JSON 文件中提取 reward 值。
    
    支持两种格式：
    1. result.reward (db 格式)
    2. history[-1].reward (locomo 格式)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 首先尝试从 result.reward 获取（db 格式）
            reward = data.get('result', {}).get('reward', None)
            if reward is not None:
                return float(reward)
            
            # 如果 result.reward 不存在，尝试从 history 的最后一个元素获取（locomo 格式）
            history = data.get('history', [])
            if history and isinstance(history[-1], dict):
                reward = history[-1].get('reward', None)
                if reward is not None:
                    return float(reward)
            
            # 如果都没有，返回 0.0
            return 0.0
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}")
        return 0.0


def calculate_reward_ratio(directory: Path) -> float:
    """计算目录下所有 JSON 文件中 reward > 0 的占比。
    
    支持嵌套目录结构（如 train/locomo-0/*.json 或 train/dbbench-std/*.json）。
    自动过滤掉 .error.json 文件。
    """
    if not directory.exists():
        return 0.0
    
    # 递归查找所有 JSON 文件，但排除错误文件
    json_files = [
        f for f in directory.glob('**/*.json')
        if not f.name.endswith('.error.json')
    ]
    if not json_files:
        return 0.0
    
    rewards = []
    for json_file in json_files:
        reward = extract_reward_from_json(json_file)
        rewards.append(reward)
    
    if not rewards:
        return 0.0
    
    positive_count = sum(1 for r in rewards if r > 0)
    ratio = positive_count / len(rewards)
    return ratio


def collect_replay_data(output_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    """收集所有 replay 的数据。
    
    Returns:
        (replay_ids, train_ratios, test_ratios)
    """
    replay_ids = []
    train_ratios = []
    test_ratios = []
    
    # 找到所有 replay 目录
    replay_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('replay')],
        key=lambda x: int(x.name.replace('replay', ''))
    )
    
    for replay_dir in replay_dirs:
        try:
            replay_id = int(replay_dir.name.replace('replay', ''))
        except ValueError:
            print(f"Warning: Skipping invalid replay directory: {replay_dir.name}")
            continue
        
        # 计算 train 和 test 的 reward > 0 占比
        train_dir = replay_dir / 'train'
        test_dir = replay_dir / 'test'
        
        train_ratio = calculate_reward_ratio(train_dir)
        test_ratio = calculate_reward_ratio(test_dir)
        
        replay_ids.append(replay_id)
        train_ratios.append(train_ratio)
        test_ratios.append(test_ratio)
        
        print(f"Replay {replay_id}: train={train_ratio:.3f}, test={test_ratio:.3f}")
    
    return replay_ids, train_ratios, test_ratios


def plot_replay_reward_line(
    replay_ids: List[int],
    train_ratios: List[float],
    test_ratios: List[float],
    output_path: Path
):
    """绘制折线图。"""
    plt.figure(figsize=(10, 6))
    
    # 使用清新的颜色
    train_color = '#6BC4A6'  # 薄荷绿
    test_color = '#FF9A9E'   # 浅珊瑚粉
    
    # 绘制折线，带标记点
    plt.plot(
        replay_ids, train_ratios,
        label='Train', color=train_color,
        marker='o', markersize=8, linewidth=2.5,
        alpha=0.8, markerfacecolor=train_color,
        markeredgecolor='white', markeredgewidth=1.5
    )
    plt.plot(
        replay_ids, test_ratios,
        label='Test', color=test_color,
        marker='s', markersize=8, linewidth=2.5,
        alpha=0.8, markerfacecolor=test_color,
        markeredgecolor='white', markeredgewidth=1.5
    )
    
    # 设置标签和标题
    plt.xlabel('Replay ID', fontsize=12, fontweight='bold')
    plt.ylabel('Reward > 0 Ratio', fontsize=12, fontweight='bold')
    plt.title('Reward > 0 Ratio Across Replays', fontsize=14, fontweight='bold')
    
    # 设置坐标轴
    plt.xlim(min(replay_ids) - 0.5, max(replay_ids) + 0.5)
    plt.ylim(-0.05, 1.05)
    plt.xticks(replay_ids)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # 显示图片
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate and plot reward > 0 ratio for replay mode'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the replay output directory (e.g., outputs/replay/zero-shot-db-epoch10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the plot (default: {output_dir}/replay_reward_plot.png)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        return
    
    # 收集数据
    print(f"Collecting data from: {output_dir}\n")
    replay_ids, train_ratios, test_ratios = collect_replay_data(output_dir)
    
    if not replay_ids:
        print("Error: No replay data found!")
        return
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / 'replay_reward_line_plot.png'
    
    # 绘制折线图
    plot_replay_reward_line(replay_ids, train_ratios, test_ratios, output_path)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    print(f"Total replays: {len(replay_ids)}")
    print(f"Train - Mean: {np.mean(train_ratios):.3f}, Std: {np.std(train_ratios):.3f}")
    print(f"Test  - Mean: {np.mean(test_ratios):.3f}, Std: {np.std(test_ratios):.3f}")


if __name__ == '__main__':
    main()

