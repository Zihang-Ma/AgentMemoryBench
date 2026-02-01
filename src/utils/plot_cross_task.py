#!/usr/bin/env python3
"""
绘制 cross task 的累积成功率曲线图
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# 尝试导入 seaborn 优化样式
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using default matplotlib style")

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
cross_dir = Path("outputs/cross")

# 方法名称和对应的颜色（优化后的专业配色方案）
methods_config = {
    "zero-shot": {"color": "#7F8C8D", "label": "Zero-Shot", "linestyle": "-"},  # 灰色，基线
    "streamICL": {"color": "#E67E22", "label": "StreamICL", "linestyle": "-"},  # 橙色
    "context-compression": {"color": "#3498DB", "label": "Context-Compression", "linestyle": "-"},  # 蓝色
    "mem0": {"color": "#2ECC71", "label": "Mem0", "linestyle": "-"},  # 绿色
    "MEMs": {"color": "#FF0000", "label": "MEMs", "linestyle": "-"},  # 亮红色，性能最好
    "awm": {"color": "#9B59B6", "label": "AWM", "linestyle": "-"},  # 紫色
}

# 读取所有 JSON 文件
data = {}
for json_file in cross_dir.glob("*.json"):
    method_name = json_file.stem
    if method_name in methods_config:
        with open(json_file, 'r', encoding='utf-8') as f:
            data[method_name] = json.load(f)

# 创建图形，使用 seaborn 优化样式（参考图的样式）
if HAS_SEABORN:
    sns.set_style("whitegrid", {
        'grid.color': '#E0E0E0',  # 浅灰色网格线
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'axes.edgecolor': '#CCCCCC',  # 浅灰色坐标轴边框
        'axes.linewidth': 1.0,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,  # 隐藏顶部边框
        'axes.spines.right': False,  # 隐藏右侧边框
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')  # 确保背景是白色
ax.set_facecolor('white')  # 确保坐标轴背景是白色

# 绘制每条曲线，按性能排序（从高到低）
# 根据最终性能排序：MEMs > StreamICL > AWM > Context-Compression > Mem0 > Zero-Shot
plot_order = ["MEMs", "streamICL", "awm", "context-compression", "mem0", "zero-shot"]

for method_name in plot_order:
    if method_name in data and method_name in methods_config:
        # 使用 overall 字段的数据，而不是 system_memory
        overall = data[method_name].get("overall", {})
        cumulative_rates = overall.get("cumulative_rates", [])
        total_counts = overall.get("total_counts", [])
        if cumulative_rates and total_counts:
            # 使用 total_counts 作为 x 轴（样本数量），而不是时间步索引
            config = methods_config[method_name]
            ax.plot(
                total_counts,
                cumulative_rates,
                color=config["color"],
                label=config["label"],
                linewidth=2.5,  # 线条粗细，参考图约2-2.5
                alpha=1.0,  # 完全不透明，参考图线条很清晰
                linestyle=config.get("linestyle", "-"),
                marker='',  # 不使用标记点，让线条更平滑
                antialiased=True,  # 抗锯齿，让线条更平滑
                zorder=3  # 确保线条在网格上方
            )

# 设置坐标轴标签（加粗，使用更明确的字体设置）
ax.set_xlabel("Number of Samples", fontsize=14, fontweight='bold', color='#2C3E50', 
              fontfamily='sans-serif')
ax.set_ylabel("Success rate", fontsize=14, fontweight='bold', color='#2C3E50',
              fontfamily='sans-serif')

# 设置标题（加粗，使用更明确的字体设置）
ax.set_title("Cumulative success rate (cross task on db and locomo-0)", 
             fontsize=16, fontweight='bold', pad=20, color='#2C3E50',
             fontfamily='sans-serif')

# 设置网格（参考图的浅灰色网格风格）
if HAS_SEABORN:
    # seaborn whitegrid 会自动设置网格，这里微调
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='#E0E0E0', zorder=0)
else:
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#E0E0E0', zorder=0)
ax.set_axisbelow(True)  # 将网格放在数据线下面

# 设置图例（参考图的样式：清晰、加粗、白色背景）
from matplotlib import font_manager
legend = ax.legend(loc='best', fontsize=12, framealpha=1.0,  # 完全不透明
                   edgecolor='#CCCCCC', facecolor='white', 
                   frameon=True, shadow=False, fancybox=False,  # 参考图没有阴影和圆角
                   prop=font_manager.FontProperties(weight='bold', size=12),  # 图例字体加粗
                   handlelength=2.5,  # 图例线条长度
                   handletextpad=0.5)  # 图例文字和线条的间距
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_edgecolor('#CCCCCC')

# 设置坐标轴范围
ax.set_xlim(left=0)
ax.set_ylim(bottom=0.2, top=0.8)  # 纵轴范围 0.2 到 0.8

# 设置坐标轴刻度（根据实际数据范围动态调整）
# x 轴：根据 overall.total_counts 的最大值设置刻度
max_samples_list = []
for method in plot_order:
    if method in data:
        overall = data[method].get("overall", {})
        total_counts = overall.get("total_counts", [])
        if total_counts:
            max_samples_list.append(max(total_counts))
max_samples = max(max_samples_list) if max_samples_list else 0
if max_samples > 0:
    # 设置 x 轴刻度，每 50 个样本一个刻度
    ax.set_xticks(np.arange(0, max_samples + 1, 50))
ax.set_yticks(np.arange(0.2, 0.81, 0.1))

# 美化坐标轴（参考图样式：浅灰色边框，清晰的刻度）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
# 刻度标签加粗
ax.tick_params(colors='#333333', labelsize=11, width=1.0, length=5)  # 刻度颜色更深，更清晰
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# 调整布局
plt.tight_layout()

# 保存图片
output_path = "outputs/cross/cumulative_success_rate.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已保存到: {output_path}")

# 显示图片
plt.show()

