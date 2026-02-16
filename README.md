<div align="center">
  <img src="figures/AgentMemoryBench.svg" width="100%" alt="Agent Memory Bench" />

  <br/>
  <br/>

  <a href="https://github.com/s010m00n/AgentMemoryBench/stargazers">
    <img src="https://img.shields.io/github/stars/s010m00n/AgentMemoryBench?style=for-the-badge&logo=github&color=ff6b6b" alt="Stars">
  </a>
  <a href="https://github.com/s010m00n/AgentMemoryBench/network/members">
    <img src="https://img.shields.io/github/forks/s010m00n/AgentMemoryBench?style=for-the-badge&logo=github&color=ee5a6f" alt="Forks">
  </a>
  <a href="https://github.com/s010m00n/AgentMemoryBench/issues">
    <img src="https://img.shields.io/github/issues/s010m00n/AgentMemoryBench?style=for-the-badge&logo=github&color=c44569" alt="Issues">
  </a>
  <a href="https://github.com/s010m00n/AgentMemoryBench/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge" alt="License">
  </a>

  <br/>
  <br/>

  <p align="center">
    <strong>A Unified Benchmark for Continual Agent Memory</strong>
    <br />
    <br />
    A comprehensive benchmark for evaluating memory mechanisms in LLM-based agents across continual learning scenarios, supporting both <strong>system memory</strong> (task workflows) and <strong>personal memory</strong> (user preferences).
    <br />
    <br />
    <a href="#overview">Overview</a> •
    <a href="#evaluation-modes">Evaluation Modes</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#creating-custom-memory-mechanisms">Custom Memory</a> •
    <a href="#implemented-memory-mechanisms">Methods</a>
  </p>
</div>

---

## 🎯 Overview

AgentMemoryBench provides a unified framework to evaluate how LLM agents learn and retain two types of memory:
- **System Memory**: Task workflows and execution patterns
- **Personal Memory**: User preferences and dialogue context

The benchmark spans **6 interactive tasks** across 4 grounding types:
- **Code-grounded**: Database (SQL), Operating System (Shell), Knowledge Graph (SPARQL)
- **Embodied**: ALFWorld (household tasks)
- **Web-grounded**: WebShop (e-commerce)
- **Dialogue-grounded**: LoCoMo (long-term conversations)

## 📊 Evaluation Modes

AgentMemoryBench supports **5 complementary evaluation modes** to provide multi-dimensional assessment of memory systems:

![Evaluation Modes](figures/evaluation_mode.png)

### 1. **Offline Mode**
Traditional train-test split evaluation. The agent learns from training samples (memory formation & evolution) and is tested on held-out samples (retrieval only).

**Metrics**: Average Success Rate (ASR), Average Steps (AS), F1-score, BLEU, LLM-as-Judge

### 2. **Online Mode**
Streaming evaluation where agents process samples sequentially with real-time memory updates. Performance is recorded after each sample to capture learning dynamics.

**Metrics**: Cumulative Success Rate (CSR), Learning Gain (LG), Stability Loss (SL)

### 3. **Replay Mode**
Periodic testing to measure knowledge retention and resistance to forgetting. After learning each stage, agents are tested on previously learned samples.

**Metrics**: Forgetting Rate (FR), Average Success Rate (ASR)

### 4. **Transfer Mode**
- **Cross-environment**: Tests knowledge generalization across different domains (e.g., DB→OS)
- **Within-environment**: Measures forward transfer—how learning current samples helps future ones

**Metrics**: Transfer Gain (TG), Forward Transfer Gain (FTG)

### 5. **Repair Mode**
Tests robustness and self-correction under erroneous feedback. Agents learn under incorrect rewards, then repair memory with correct feedback.

**Metrics**: Error Robustness (ER), Repair Gain (RG), Net Recovery (NR)

## 🏗️ Project Structure

```
AgentMemoryBench/
├── configs/                    # Configuration files
│   ├── assignment/            # Experiment configurations
│   │   └── default.yaml       # Main experiment config
│   ├── tasks/                 # Task-specific configs (6 tasks)
│   │   ├── dbbench.yaml       # Database (SQL)
│   │   ├── os.yaml            # Operating System (Shell)
│   │   ├── kg.yaml            # Knowledge Graph (SPARQL)
│   │   ├── alfworld.yaml      # Embodied AI
│   │   ├── webshop.yaml       # E-commerce
│   │   └── locomo-*.yaml      # Long conversations (0-9)
│   └── llmapi/                # LLM API configurations
│       ├── api.yaml           # API endpoint & key for agent LLM
│       ├── agent.yaml         # Agent model name
│       ├── evaluate_api.yaml  # API for LoCoMo LLM-as-Judge
│       └── evaluate_agent.yaml# Model for evaluation
│
├── data/                       # Task datasets
│   ├── dbbench/               # Database operations (SQL)
│   ├── os_interaction/        # OS commands (Shell)
│   ├── knowledgegraph/        # KG queries (SPARQL)
│   ├── alfworld/              # Embodied tasks
│   ├── webshop/               # E-commerce tasks
│   └── locomo/                # Long dialogues (10 conversations)
│
├── memory/                     # Memory mechanism implementations
│   ├── base.py                # Base class for all memory mechanisms
│   ├── registry.py            # Memory registry system
│   ├── zero_shot/             # Baseline (no memory)
│   ├── streamICL/             # RAG-based retrieval (topk=4)
│   ├── awmPro/                # System memory via workflows (topk=8)
│   ├── mem0/                  # Personal memory via preferences
│   └── MEMs/                  # Multi-memory coordination (proposed)
│
├── execution/                  # Execution engines
│   ├── base.py                # Base execution engine
│   └── single_agent/          # Single-agent executor
│
├── src/                        # Core implementation
│   ├── runner/                # Main entry point
│   │   ├── main.py            # Experiment runner
│   │   ├── builders.py        # Component builders
│   │   ├── config.py          # Configuration parser
│   │   └── schedule_utils.py  # Scheduling utilities
│   ├── client/                # Client-side scheduling
│   │   ├── backend.py         # Backend interface
│   │   └── scheduler.py       # Task scheduler
│   ├── server/                # Backend task servers (Docker)
│   │   └── tasks/             # Task implementations
│   └── utils/                 # Analysis utilities
│       ├── message_schema.py  # Message format compatibility layer
│       └── analyze_results_*.py # Result analysis scripts
│
├── extra/                      # Docker orchestration
│   ├── docker-compose.yml     # Service definitions
│   └── *.Dockerfile           # Task-specific containers
│
├── outputs/                    # Experiment results
│   └── [timestamp]/           # Grouped by experiment time
│       └── [task_name]/       # Grouped by task
│           └── [index].json   # Individual sample results
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Prerequisites

#### Python Environment
```bash
# Create conda environment with Python 3.9
conda create -n aMB python=3.9

# Activate environment
conda activate aMB

# Navigate to project directory
cd /path/to/AgentMemoryBench

# Install dependencies
pip install -r requirements.txt
```

#### Docker Installation
Docker is required to run backend task servers. Install Docker Desktop:
- **Windows/Mac**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Follow [official guide](https://docs.docker.com/engine/install/)

### 2. Data & Model Setup

#### Knowledge Graph (Freebase) Database

The Knowledge Graph task requires the Freebase database:

1. **Download database** (~50 GB):
   - Download link: [OneDrive](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/su_809_osu_edu/Ed0SY7sAS_ZGqNTovDYhVCcBxEmZfhL3B-chAiuoZCrpVg?e=vpHUei)
   - **Recommended**: Use a download manager (e.g., Free Download Manager) instead of browser

2. **Extract** the downloaded `virtuoso_db.zip`

3. **Configure path** in `extra/docker-compose.yml` (line 114):
   ```yaml
   freebase:
     build:
       context: ..
       dockerfile: extra/freebase.Dockerfile
     volumes:
       - "/absolute/path/to/virtuoso_db:/database"  # Use absolute path
     init: true
   ```

   **Important**:
   - Use **absolute paths**
   - Windows: Use forward slashes `/` (e.g., `C:/Users/...`)
   - Example: `B:/desktop/AgentMemoryBench/virtuoso_db:/database`

#### LoCoMo Tokenizer

Download the tokenizer model for fair evaluation:

```bash
# Download xlm-roberta-base from HuggingFace
# https://huggingface.co/FacebookAI/xlm-roberta-base

# Configure path in src/server/tasks/locomo/task.py (line 47)
tokenizer = AutoTokenizer.from_pretrained("/path/to/xlm-roberta-base")
```

#### Embedding Model (for streamICL, awmPro, MEMs)

Download the embedding model for fair comparison:

```bash
# Download bge-base-en-v1.5 from HuggingFace
# https://huggingface.co/BAAI/bge-base-en-v1.5

# Configure paths in YAML files:
# - memory/streamICL/streamICL.yaml
# - memory/awmPro/awmPro.yaml
# - memory/MEMs/MEMs.yaml
```

#### Mem0 API Key

To use the Mem0 method:

1. Register for API key at [mem0.ai](https://app.mem0.ai/)
2. Configure in `memory/mem0/mem0.yaml`:
   ```yaml
   api_key: "your_mem0_api_key_here"
   wait_time: 60.0  # Recommended: 60s for system tasks, 150s for personal, 100s for mixed
   ```

### 3. Start Backend Services

```bash
# Navigate to Docker directory
cd extra

# Build required containers
docker-compose build local-os-default
docker-compose build local-os-packages
docker-compose build local-os-ubuntu
docker-compose build freebase

# Start all services
docker-compose up
```

**Note**: Keep this terminal running. Services run on `http://localhost:5038`

### 4. Configure LLM API

**Recommended**: Use [SiliconFlow API](https://siliconflow.cn/) to avoid model name mismatches.

#### Agent LLM Configuration

Edit `configs/llmapi/api.yaml`:

```yaml
base_url: "https://api.siliconflow.cn/v1"
headers:
  Content-Type: application/json
  Authorization: "Bearer YOUR_API_KEY"
```

Edit `configs/llmapi/agent.yaml`:

```yaml
model: "Qwen/Qwen2.5-14B-Instruct"  # Or your preferred model
```

#### Evaluation LLM (for LoCoMo LLM-as-Judge)

Edit `configs/llmapi/evaluate_api.yaml`:

```yaml
base_url: "https://api.siliconflow.cn/v1"
headers:
  Content-Type: application/json
  Authorization: "Bearer YOUR_API_KEY"
```

Edit `configs/llmapi/evaluate_agent.yaml`:

```yaml
model: "Qwen/Qwen2.5-14B-Instruct"  # Or evaluation model
```

### 5. Configure Experiments

Edit `configs/assignment/default.yaml`:

```yaml
# Lifelong Learning Benchmark Configuration
# 配置要测试的任务、记忆机制、执行方法和实验参数

# ===== 任务配置 =====
# 指定要测试的任务列表（5个system memory任务+2个user memory任务，共7个任务）
tasks:
  # system memory任务
  # - name: dbbench-std
  #   config_path: configs/tasks/dbbench.yaml
  - name: os-std
    config_path: configs/tasks/os.yaml
  # - name: kg-std
  #   config_path: configs/tasks/kg.yaml
  # - name: alfworld-std
  #   config_path: configs/tasks/alfworld.yaml
  # - name: webshop-std
  #   config_path: configs/tasks/webshop.yaml

  # user memory任务
  # - name: locomo-0
  #   config_path: configs/tasks/locomo-0.yaml
  # - name: locomo-1
  #   config_path: configs/tasks/locomo-1.yaml
  # - name: locomo-2
  #   config_path: configs/tasks/locomo-2.yaml
  # - name: locomo-3
  #   config_path: configs/tasks/locomo-3.yaml
  # - name: locomo-4
  #   config_path: configs/tasks/locomo-4.yaml
  # - name: locomo-5
  #   config_path: configs/tasks/locomo-5.yaml
  # - name: locomo-6
  #   config_path: configs/tasks/locomo-6.yaml
  # - name: locomo-7
  #   config_path: configs/tasks/locomo-7.yaml
  # - name: locomo-8
  #   config_path: configs/tasks/locomo-8.yaml
  # - name: locomo-9
  #   config_path: configs/tasks/locomo-9.yaml

# ===== 记忆机制配置 =====
# 从 memory 文件夹中选择记忆机制（统一使用 snake_case 命名）
memory_mechanism:
  name: zero_shot  # 可选: zero_shot, stream_icl, mem0, awm_pro

# ===== 执行方法配置 =====
# 从 execution 文件夹中选择执行方法
execution_method:
  name: single_agent  # 当前版本仅支持 single_agent
  config_path: execution/single_agent/single_agent.yaml

# ===== 实验参数 =====
experiment:
  # 训练模式: online (在线学习) 或 offline (离线学习) 或 replay (重放学习) 或 transfer (迁移学习)  或 repair（修复学习）
  training_mode: online  # online | offline | replay | transfer | repair
  
  keep_number: 700 #只有training_mode等于online时，这个参数才有效 #为None或者小于等于0，则不进行截断

  train_size: 0.6 #只有training_mode等于offline时，这个参数才有效

  #在transfer_task中学习（update+enhance，相当于online），在transfer_after_task中进行测试（仅enhance）
  transfer_task: dbbench-std #只有training_mode等于transfer时，这个参数才有效
  transfer_after_task: os-std #只有training_mode等于transfer时，这个参数才有效
  forward_transfer_num: 3 #只有training_mode等于transfer且transfer_task==transfer_after_task时，这个参数才有效，表示前向迁移的步数

  #这两个参数的意思是，每学过m个样本（update+enhance，相当于online），就从学过的所有样本中随机抽样n个进行测试（仅enhance）
  replay_m: 20 #只有training_mode等于replay时，这个参数才有效
  replay_n: 20 #只有training_mode等于replay时，这个参数才有效
  replay_seed: 66 #只有training_mode等于replay时，这个参数才有效

  #这两个参数的意思是，将所有的case按照m分成x组，然后组与组之前是串行学习的，这没毛病，但是每个组中会有n个case的judge是错乱的
  repair_m: 20  # 只有training_mode等于repair时，这个参数才有效（对于普通任务），每组的样本数量
  repair_n: 20  # 只有training_mode等于repair时，这个参数才有效，每组中需要反转奖励的样本数量
  repair_seed: 66  # 只有training_mode等于repair时，这个参数才有效，选择反转样本的随机种子
  repair_size_locomo: 0.5  # 只有training_mode等于repair且任务为locomo时有效，表示每个session中需要反转的QA比例（0-1之间）

  ...
  
  cross_task: False  # True | False

  # 数据打乱: 是否打乱任务顺序，可以设置随机种子
  shuffle:
    enabled: True  # True | False
    seed: 66  # 整数，如果 enabled 为 true 时使用
```

### 6. Run Experiments

```bash
# Run with default configuration
python -m src.runner.main

# Or specify custom config
python -m src.runner.main --config configs/assignment/my_experiment.yaml
```

## 🛠️ Creating Custom Memory Mechanisms

### Step 1: Implement Memory Class

Create a new directory under `memory/` (e.g., `memory/my_memory/`):

```python
# memory/my_memory/my_memory.py
from __future__ import annotations
from typing import List, Dict, Any
import yaml
from ..base import MemoryMechanism

class MyMemory(MemoryMechanism):
    """Your custom memory mechanism"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize your memory storage

    def use_memory(
        self,
        task: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance messages with memory before LLM call.

        Args:
            task: Task name (e.g., "dbbench-std", "os-std")
            messages: Original message list

        Returns:
            Enhanced messages with retrieved memory
        """
        # Retrieve relevant experience from memory
        # Inject experience into messages
        return messages  # Return enhanced messages

    def update_memory(
        self,
        task: str,
        history: List[Dict[str, Any]],
        result: Dict[str, Any]
    ) -> None:
        """
        Update memory after sample execution.

        Args:
            task: Task name
            history: Full dialogue history
            result: Execution result (reward, status, etc.)
        """
        # Update your memory storage based on history and result
        pass

def load_my_memory_from_yaml(config_path: str) -> MyMemory:
    """Load memory from YAML config"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return MyMemory(config)
```

Create configuration file `memory/my_memory/my_memory.yaml`:

```yaml
name: my_memory
description: "My custom memory mechanism"

# Your configuration parameters
param1: value1
param2: value2
```

### Step 2: Register in Registry

Add registration in `memory/registry.py`:

```python
def _register_all_memories():
    # ... existing registrations ...

    # Register your memory mechanism (use snake_case)
    from memory.my_memory.my_memory import load_my_memory_from_yaml
    register_memory(
        name="my_memory",  # Use snake_case
        loader_func=load_my_memory_from_yaml,
        default_config_path="memory/my_memory/my_memory.yaml",
    )
```

### Step 3: Use Your Memory

Configure in `configs/assignment/default.yaml`:

```yaml
memory_mechanism:
  name: my_memory  # Use snake_case naming
  config_path: memory/my_memory/my_memory.yaml  # Optional
```

## 📈 Implemented Memory Mechanisms

| Method | Type | Description | Key Features |
|--------|------|-------------|--------------|
| **zero_shot** | Baseline | No memory | Reflects base LLM capability |
| **streamICL** | Retrieval | RAG-based ICL | Stores full trajectories, topk=4 |
| **awmPro** | System | Workflow memory | Extracts execution patterns, topk=8 |
| **mem0** | Personal | Preference memory | Graph-based storage with ADD/UPDATE/DELETE |
| **MEMs** | Hybrid | Multi-memory | Coordinates system & personal memory via trigger model |

### Fair Comparison Notes

- **streamICL**: Uses topk=4 following [original paper](https://arxiv.org/abs/2406.08747)
- **awmPro**: Modified from [AWM](https://arxiv.org/abs/2409.07429) with mem0-inspired management, topk=8 based on workflow induction experiments
- **mem0**: Uses best practices from [official implementation](https://arxiv.org/abs/2504.19413)

See ablation studies in paper for detailed topk analysis across different tasks.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Task datasets adapted from AgentBench and LoCoMo
- Evaluation protocols inspired by continual learning literature
- Memory baselines from StreamBench, AWM, and Mem0
