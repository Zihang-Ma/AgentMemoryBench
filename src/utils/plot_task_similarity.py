#!/usr/bin/env python3
"""
绘制任务相似度热力图
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# 尝试导入 seaborn 优化样式
try:
    import seaborn as sns
    sns.set_style("white")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using default matplotlib style")

# 设置字体样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path = Path("data/task_similarity.json")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取数据集列表
datasets = data.get("datasets", [])
within_similarity = data.get("within_dataset_similarity", {})
cross_similarity = data.get("cross_dataset_similarity", {})

# 创建相似度矩阵
n_datasets = len(datasets)
similarity_matrix = np.zeros((n_datasets, n_datasets))

# 填充矩阵
for i, dataset1 in enumerate(datasets):
    for j, dataset2 in enumerate(datasets):
        if i == j:
            # 对角线：within_dataset_similarity
            similarity_matrix[i, j] = within_similarity.get(dataset1, 0.0)
        elif i < j:
            # 上三角：cross_dataset_similarity
            if dataset1 in cross_similarity and dataset2 in cross_similarity[dataset1]:
                similarity_matrix[i, j] = cross_similarity[dataset1][dataset2].get("dataset_level", 0.0)
        else:
            # 下三角：对称复制
            if dataset2 in cross_similarity and dataset1 in cross_similarity[dataset2]:
                similarity_matrix[i, j] = cross_similarity[dataset2][dataset1].get("dataset_level", 0.0)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 绘制热力图（越小越相关，所以反转颜色映射）
if HAS_SEABORN:
    # 使用 seaborn 绘制热力图
    # 反转颜色映射：使用 reversed colormap，小值（高相似度）用深色，大值（低相似度）用浅色
    sns.heatmap(
        similarity_matrix,
        annot=False,  # 不显示数值
        cmap='YlOrRd_r',  # 反转的颜色方案：红-橙-黄（小值深色，大值浅色）
        square=True,  # 正方形格子
        linewidths=1.5,  # 格子之间的线宽
        cbar_kws={'label': 'Similarity (lower = more similar)', 'shrink': 0.8},  # 颜色条标签
        xticklabels=datasets,
        yticklabels=datasets,
        ax=ax,
        vmin=0,  # 最小值
        vmax=0.5,  # 最大值（根据数据范围调整）
    )
else:
    # 使用 matplotlib 绘制热力图
    # 反转颜色映射
    im = ax.imshow(similarity_matrix, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=0.5)
    
    # 设置刻度
    ax.set_xticks(np.arange(n_datasets))
    ax.set_yticks(np.arange(n_datasets))
    ax.set_xticklabels(datasets)
    ax.set_yticklabels(datasets)
    
    # 不添加数值标注（用户要求）
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Similarity (lower = more similar)', fontsize=12, fontweight='bold')

# 设置标题（加粗）
ax.set_title("Task Similarity Heatmap", fontsize=16, fontweight='bold', pad=20)

# 设置坐标轴标签（加粗）
ax.set_xlabel("Dataset", fontsize=14, fontweight='bold')
ax.set_ylabel("Dataset", fontsize=14, fontweight='bold')

# 旋转 x 轴标签
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

# 调整布局
plt.tight_layout()

# 保存图片
output_path = "data/task_similarity_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"热力图已保存到: {output_path}")

# 显示图片
plt.show()

