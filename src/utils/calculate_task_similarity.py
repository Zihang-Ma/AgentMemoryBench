"""
任务相似度计算工具

根据论文方法计算：
1. 数据集内任务相似度（如 db 数据集内）
2. 跨数据集任务相似度（如 db 和 os 之间）

使用方法：
    python -m src.utils.calculate_task_similarity
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]

# 数据目录
DATA_DIR = ROOT_DIR / "outputs" / "data_for_calculate_similarity"

# Embedding 模型路径（本地）
EMBEDDING_MODEL_PATH = Path(r"B:\desktop\python\agent\lifeLongLearning (ACL2026\bge-base-en-v1.5")

# 数据集映射（任务名称前缀 -> 数据集名称）
DATASET_MAPPING = {
    "dbbench": "db",
    "os": "os",
    "kg": "kg",
    "alfworld": "alf",
    "webshop": "webshop",
    "locomo": "locomo",
}


def extract_task_name_from_path(file_path: Path) -> str:
    """从文件路径提取任务名称"""
    # 例如：outputs/data_for_calculate_similarity/db/0.json -> dbbench-std
    # 或者：outputs/data_for_calculate_similarity/locomo-0/0.json -> locomo-0
    parent_dir = file_path.parent.name
    
    # 如果是 locomo-0, locomo-4，直接返回
    if parent_dir.startswith("locomo-"):
        return parent_dir
    
    # 否则根据目录名映射到任务名称前缀
    if parent_dir == "db":
        return "dbbench-std"
    elif parent_dir == "os":
        return "os-std"
    elif parent_dir == "kg":
        return "kg-std"
    elif parent_dir == "alf":
        return "alfworld-std"
    elif parent_dir == "webshop":
        return "webshop-std"
    else:
        return parent_dir


def get_dataset_name(task_name: str) -> str:
    """从任务名称获取数据集名称"""
    if task_name.startswith("locomo-"):
        return "locomo"
    elif task_name.startswith("dbbench"):
        return "db"
    elif task_name.startswith("os"):
        return "os"
    elif task_name.startswith("kg"):
        return "kg"
    elif task_name.startswith("alfworld"):
        return "alf"
    elif task_name.startswith("webshop"):
        return "webshop"
    else:
        # 默认使用任务名称的前缀部分
        parts = task_name.split("-")
        return parts[0] if parts else task_name


def load_task_samples(data_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Path]]]:
    """
    加载所有任务的样本数据
    
    Returns:
        ({task_name: [sample1, sample2, ...]}, {task_name: [file_path1, file_path2, ...]})
    """
    task_samples: Dict[str, List[Dict[str, Any]]] = {}
    task_file_paths: Dict[str, List[Path]] = {}
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # 遍历所有子目录
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # 跳过 metrics.txt 等文件
        json_files = sorted([f for f in subdir.glob("*.json")], key=lambda x: int(x.stem) if x.stem.isdigit() else 999999)
        
        if not json_files:
            logger.warning(f"No JSON files found in {subdir}")
            continue
        
        # 从第一个文件提取任务名称
        first_file = json_files[0]
        task_name = extract_task_name_from_path(first_file)
        
        samples = []
        file_paths = []
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 如果加载的是列表，跳过（可能是格式错误的文件）
                    if isinstance(data, list):
                        logger.warning(f"File {json_file} contains a list instead of dict, skipping")
                        continue
                    # 确保是字典格式
                    if isinstance(data, dict):
                        samples.append(data)
                        file_paths.append(json_file)
                    else:
                        logger.warning(f"File {json_file} contains unexpected type {type(data)}, skipping")
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        if samples:
            task_samples[task_name] = samples
            task_file_paths[task_name] = file_paths
            logger.info(f"Loaded {len(samples)} samples for task: {task_name}")
    
    return task_samples, task_file_paths


def extract_task_description(sample: Dict[str, Any], file_path: Path = None) -> Tuple[str, bool]:
    """
    从样本中提取任务描述（第一个 system + user 消息）
    
    对于 db、os、kg、alf、webshop：使用 history[0] (system) + history[1] (user)
    对于 locomo-0、locomo-4：使用 history[0] (system) + history[1] (user)
    """
    """
    从样本中提取任务描述（第一个 system + user 消息）
    
    Returns:
        (description, is_valid): 描述文本和是否有效
    """
    # 处理 sample 可能是列表的情况（某些文件格式可能不同）
    if isinstance(sample, list):
        file_info = f" ({file_path})" if file_path else ""
        logger.warning(f"Sample is a list, skipping{file_info}")
        return "", False
    
    if not isinstance(sample, dict):
        file_info = f" ({file_path})" if file_path else ""
        logger.warning(f"Sample is not a dict (type: {type(sample)}), skipping{file_info}")
        return "", False
    
    history = sample.get("history", [])
    
    if len(history) < 2:
        file_info = f" ({file_path})" if file_path else ""
        logger.warning(f"Sample has less than 2 messages in history (got {len(history)}), skipping{file_info}")
        return "", False
    
    # 提取第一个 system 和 user 消息
    system_msg = None
    user_msg = None
    
    for msg in history:
        role = msg.get("role", "")
        if role == "system" and system_msg is None:
            system_msg = msg.get("content", "")
        elif role == "user" and user_msg is None:
            user_msg = msg.get("content", "")
            break  # 找到第一个 user 消息后停止
    
    if not system_msg or not user_msg:
        # 提供更详细的警告信息
        roles_found = [msg.get("role", "unknown") for msg in history[:5]]
        file_info = f" ({file_path})" if file_path else ""
        logger.warning(f"Missing system or user message in sample. Found roles: {roles_found}{file_info}")
        return "", False
    
    # 拼接 system + user 作为任务描述
    description = f"{system_msg}\n{user_msg}"
    return description.strip(), True


def aggregate_task_descriptions(
    samples: List[Dict[str, Any]], 
    file_paths: List[Path],
    max_samples: int = None
) -> str:
    """
    聚合多个样本的任务描述
    
    Args:
        samples: 样本列表
        file_paths: 对应的文件路径列表
        max_samples: 最多使用多少个样本（None 表示使用所有）
    
    Returns:
        聚合后的任务描述文本
    """
    if max_samples:
        samples = samples[:max_samples]
        file_paths = file_paths[:max_samples]
    
    descriptions = []
    skipped_files = []
    
    for i, (sample, file_path) in enumerate(zip(samples, file_paths)):
        desc, is_valid = extract_task_description(sample, file_path)
        if is_valid:
            descriptions.append(desc)
        else:
            skipped_files.append(file_path)
    
    # 记录被跳过的文件
    if skipped_files:
        logger.info(f"  Skipped {len(skipped_files)} invalid samples:")
        for file_path in skipped_files:
            logger.info(f"    - {file_path}")
    
    # 将所有描述用换行符连接
    aggregated = "\n\n".join(descriptions)
    return aggregated


def calculate_within_dataset_similarity(
    sample_embeddings: List[np.ndarray]
) -> float:
    """
    计算数据集内任务相似度（基于样本级别）
    
    步骤（完全按照论文方法）：
    1. 计算所有样本 embedding 的聚类中心（均值向量）
    2. 计算每个样本 embedding 与聚类中心的余弦距离
    3. 取所有距离的平均值
    
    Args:
        sample_embeddings: 该数据集内所有样本的embedding向量列表
    
    Returns:
        平均余弦距离（越小表示相似度越高）
    """
    if len(sample_embeddings) < 1:
        logger.warning(f"No samples found for dataset")
        return float('inf')
    
    # 如果只有一个样本，自己与自己的相似度为0
    if len(sample_embeddings) == 1:
        return 0.0
    
    # 转换为 numpy 数组
    embeddings = np.array(sample_embeddings)
    
    # 计算聚类中心（均值向量）
    cluster_center = np.mean(embeddings, axis=0)
    
    # 计算每个样本 embedding 与聚类中心的余弦距离
    # 余弦距离 = 1 - 余弦相似度
    distances = []
    for emb in embeddings:
        # 归一化向量
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        center_norm = cluster_center / (np.linalg.norm(cluster_center) + 1e-8)
        
        # 计算余弦相似度
        cosine_sim = np.dot(emb_norm, center_norm)
        
        # 转换为余弦距离
        cosine_dist = 1 - cosine_sim
        distances.append(cosine_dist)
    
    # 返回平均距离
    avg_distance = np.mean(distances)
    return float(avg_distance)


def calculate_cross_dataset_similarity_dataset_level(
    task_embeddings: Dict[str, np.ndarray],
    dataset1_name: str,
    dataset2_name: str
) -> float:
    """
    计算跨数据集相似度（数据集级别方法）
    
    步骤：
    1. 分别计算两个数据集的聚类中心
    2. 计算两个聚类中心的余弦距离
    
    Args:
        task_embeddings: {task_name: embedding_vector}
        dataset1_name: 第一个数据集名称
        dataset2_name: 第二个数据集名称
    
    Returns:
        余弦距离（越小表示相似度越高）
    """
    # 筛选出属于两个数据集的任务
    dataset1_tasks = {
        task_name: emb for task_name, emb in task_embeddings.items()
        if get_dataset_name(task_name) == dataset1_name
    }
    dataset2_tasks = {
        task_name: emb for task_name, emb in task_embeddings.items()
        if get_dataset_name(task_name) == dataset2_name
    }
    
    if len(dataset1_tasks) == 0 or len(dataset2_tasks) == 0:
        logger.warning(f"One of the datasets is empty: {dataset1_name} ({len(dataset1_tasks)}), {dataset2_name} ({len(dataset2_tasks)})")
        return float('inf')
    
    # 计算两个数据集的聚类中心
    embeddings1 = np.array(list(dataset1_tasks.values()))
    embeddings2 = np.array(list(dataset2_tasks.values()))
    
    center1 = np.mean(embeddings1, axis=0)
    center2 = np.mean(embeddings2, axis=0)
    
    # 归一化
    center1_norm = center1 / (np.linalg.norm(center1) + 1e-8)
    center2_norm = center2 / (np.linalg.norm(center2) + 1e-8)
    
    # 计算余弦相似度
    cosine_sim = np.dot(center1_norm, center2_norm)
    
    # 转换为余弦距离
    cosine_dist = 1 - cosine_sim
    return float(cosine_dist)


def calculate_cross_dataset_similarity_task_pair_level(
    task_embeddings: Dict[str, np.ndarray],
    dataset1_name: str,
    dataset2_name: str
) -> float:
    """
    计算跨数据集相似度（任务对级别方法）
    
    步骤：
    1. 计算 dataset1 中每个任务与 dataset2 中每个任务的余弦距离
    2. 取所有距离的平均值
    
    Args:
        task_embeddings: {task_name: embedding_vector}
        dataset1_name: 第一个数据集名称
        dataset2_name: 第二个数据集名称
    
    Returns:
        平均余弦距离（越小表示相似度越高）
    """
    # 筛选出属于两个数据集的任务
    dataset1_tasks = {
        task_name: emb for task_name, emb in task_embeddings.items()
        if get_dataset_name(task_name) == dataset1_name
    }
    dataset2_tasks = {
        task_name: emb for task_name, emb in task_embeddings.items()
        if get_dataset_name(task_name) == dataset2_name
    }
    
    if len(dataset1_tasks) == 0 or len(dataset2_tasks) == 0:
        logger.warning(f"One of the datasets is empty: {dataset1_name} ({len(dataset1_tasks)}), {dataset2_name} ({len(dataset2_tasks)})")
        return float('inf')
    
    # 计算所有任务对的余弦距离
    distances = []
    embeddings1 = np.array(list(dataset1_tasks.values()))
    embeddings2 = np.array(list(dataset2_tasks.values()))
    
    for emb1 in embeddings1:
        for emb2 in embeddings2:
            # 归一化
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # 计算余弦相似度
            cosine_sim = np.dot(emb1_norm, emb2_norm)
            
            # 转换为余弦距离
            cosine_dist = 1 - cosine_sim
            distances.append(cosine_dist)
    
    # 返回平均距离
    avg_distance = np.mean(distances)
    return float(avg_distance)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Task Similarity Calculation")
    logger.info("=" * 60)
    
    # 1. 加载数据
    logger.info(f"Loading task samples from: {DATA_DIR}")
    task_samples, task_file_paths = load_task_samples(DATA_DIR)
    
    if not task_samples:
        logger.error("No task samples loaded!")
        return
    
    logger.info(f"Loaded {len(task_samples)} tasks: {list(task_samples.keys())}")
    
    # 2. 提取任务描述并聚合（用于跨数据集相似度计算）
    logger.info("Extracting and aggregating task descriptions...")
    task_descriptions: Dict[str, str] = {}
    for task_name, samples in task_samples.items():
        # 使用所有样本聚合任务描述
        file_paths = task_file_paths[task_name]
        description = aggregate_task_descriptions(samples, file_paths, max_samples=None)
        task_descriptions[task_name] = description
        logger.info(f"  {task_name}: {len(description)} characters")
    
    # 3. 生成 Embedding
    logger.info(f"Loading embedding model from: {EMBEDDING_MODEL_PATH}")
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path not found: {EMBEDDING_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(EMBEDDING_MODEL_PATH))
    model = AutoModel.from_pretrained(str(EMBEDDING_MODEL_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    logger.info("Generating sample-level embeddings...")
    
    def encode_text(text: str) -> np.ndarray:
        """使用 transformers 生成 embedding"""
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            # Expand attention mask to match embeddings shape
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            # Sum embeddings and divide by sum of attention mask
            sum_embeddings = torch.sum(embeddings * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize
        embedding_np = mean_embeddings.cpu().numpy()[0]
        embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)
        return embedding_np
    
    # 3.1 为每个样本生成embedding（用于数据集内相似度计算，按照论文方法）
    logger.info("Generating sample-level embeddings for within-dataset similarity...")
    dataset_sample_embeddings: Dict[str, List[np.ndarray]] = {}
    total_samples = 0
    skipped_counts: Dict[str, int] = {}
    
    for task_name, samples in task_samples.items():
        dataset_name = get_dataset_name(task_name)
        
        # locomo-0 和 locomo-4 分别作为独立任务处理
        if dataset_name == "locomo":
            # 使用任务名称作为key，而不是数据集名称
            task_key = task_name
        else:
            # 其他数据集使用数据集名称作为key
            task_key = dataset_name
        
        if task_key not in dataset_sample_embeddings:
            dataset_sample_embeddings[task_key] = []
            skipped_counts[task_key] = 0
        
        file_paths = task_file_paths[task_name]
        for sample, file_path in zip(samples, file_paths):
            desc, is_valid = extract_task_description(sample, file_path)
            if is_valid:
                embedding = encode_text(desc)
                dataset_sample_embeddings[task_key].append(embedding)
                total_samples += 1
            else:
                skipped_counts[task_key] += 1
    
    logger.info(f"Generated {total_samples} sample embeddings")
    for task_key in sorted(dataset_sample_embeddings.keys()):
        valid_count = len(dataset_sample_embeddings[task_key])
        skipped_count = skipped_counts.get(task_key, 0)
        logger.info(f"  {task_key}: {valid_count} valid samples, {skipped_count} skipped")
    
    # 3.2 为每个任务生成embedding（用于跨数据集相似度计算）
    logger.info("Generating task-level embeddings for cross-dataset similarity...")
    task_embeddings: Dict[str, np.ndarray] = {}
    for task_name, description in task_descriptions.items():
        embedding = encode_text(description)
        task_embeddings[task_name] = embedding
        logger.info(f"  {task_name}: embedding shape {embedding.shape}")
    
    # 4. 计算数据集内相似度（按照论文方法：基于样本级别）
    logger.info("Calculating within-dataset similarity...")
    within_similarities: Dict[str, float] = {}
    
    for dataset_name in sorted(dataset_sample_embeddings.keys()):
        sample_embeddings = dataset_sample_embeddings[dataset_name]
        logger.info(f"  Calculating similarity for {dataset_name}: {len(sample_embeddings)} samples")
        similarity = calculate_within_dataset_similarity(sample_embeddings)
        within_similarities[dataset_name] = similarity
        logger.info(f"  {dataset_name}: {similarity:.4f} (based on {len(sample_embeddings)} samples)")
    
    # 5. 计算跨数据集相似度
    logger.info("Calculating cross-dataset similarity...")
    cross_similarities: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    # 构建任务/数据集列表：locomo-0 和 locomo-4 作为独立任务，其他使用数据集名称
    task_or_dataset_list = []
    seen_datasets = set()
    
    for task_name in sorted(task_embeddings.keys()):
        dataset_name = get_dataset_name(task_name)
        if dataset_name == "locomo":
            # locomo-0 和 locomo-4 作为独立任务
            task_or_dataset_list.append(task_name)
        else:
            # 其他数据集只添加一次
            if dataset_name not in seen_datasets:
                task_or_dataset_list.append(dataset_name)
                seen_datasets.add(dataset_name)
    
    # 计算所有任务/数据集对之间的相似度
    for i, item1 in enumerate(task_or_dataset_list):
        for item2 in task_or_dataset_list[i+1:]:
            # 判断是任务名称还是数据集名称
            if item1.startswith("locomo-"):
                # item1 是 locomo 任务，直接使用任务 embedding
                emb1_list = [task_embeddings[item1]]
            else:
                # item1 是数据集名称，获取该数据集所有任务的 embeddings
                emb1_list = [
                    emb for task_name, emb in task_embeddings.items()
                    if get_dataset_name(task_name) == item1
                ]
            
            if item2.startswith("locomo-"):
                # item2 是 locomo 任务，直接使用任务 embedding
                emb2_list = [task_embeddings[item2]]
            else:
                # item2 是数据集名称，获取该数据集所有任务的 embeddings
                emb2_list = [
                    emb for task_name, emb in task_embeddings.items()
                    if get_dataset_name(task_name) == item2
                ]
            
            if len(emb1_list) == 0 or len(emb2_list) == 0:
                logger.warning(f"One of the items is empty: {item1} ({len(emb1_list)}), {item2} ({len(emb2_list)})")
                continue
            
            # 方法1：数据集级别（计算聚类中心的距离）
            center1 = np.mean(np.array(emb1_list), axis=0)
            center2 = np.mean(np.array(emb2_list), axis=0)
            center1_norm = center1 / (np.linalg.norm(center1) + 1e-8)
            center2_norm = center2 / (np.linalg.norm(center2) + 1e-8)
            cosine_sim_dataset = np.dot(center1_norm, center2_norm)
            similarity_dataset_level = float(1 - cosine_sim_dataset)
            
            # 方法2：任务对级别（计算所有任务对距离的平均值）
            distances = []
            for emb1 in emb1_list:
                for emb2 in emb2_list:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
                    cosine_sim = np.dot(emb1_norm, emb2_norm)
                    cosine_dist = 1 - cosine_sim
                    distances.append(cosine_dist)
            similarity_task_pair_level = float(np.mean(distances))
            
            if item1 not in cross_similarities:
                cross_similarities[item1] = {}
            cross_similarities[item1][item2] = {
                "dataset_level": similarity_dataset_level,
                "task_pair_level": similarity_task_pair_level
            }
            
            logger.info(f"  {item1} <-> {item2}:")
            logger.info(f"    Dataset-level: {similarity_dataset_level:.4f}")
            logger.info(f"    Task-pair-level: {similarity_task_pair_level:.4f}")
    
    # 6. 保存结果
    output_file = DATA_DIR / "task_similarity.json"
    # 获取所有数据集名称（用于保存，locomo 任务单独列出）
    datasets_for_save = set()
    for task_name in task_embeddings.keys():
        dataset_name = get_dataset_name(task_name)
        if dataset_name == "locomo":
            # locomo 任务不添加到数据集列表
            continue
        datasets_for_save.add(dataset_name)
    
    result = {
        "within_dataset_similarity": within_similarities,
        "cross_dataset_similarity": cross_similarities,
        "tasks": list(task_embeddings.keys()),
        "datasets": sorted(datasets_for_save),
        "embedding_model": str(EMBEDDING_MODEL_PATH)
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("Task similarity calculation completed!")


if __name__ == "__main__":
    main()

