"""
构建器模块 - 负责构建 memory、execution engine 等组件
"""
from __future__ import annotations

from pathlib import Path

from execution.single_agent.single_agent import load_single_agent_engine_from_yaml
from memory.zero_shot.zero_shot import load_zero_shot_from_yaml
from src.runner.config import ExperimentConfig


ROOT_DIR = Path(__file__).resolve().parents[2]


def build_memory_from_config(cfg: ExperimentConfig):
    """
    根据 assignment.yaml 中的 memory_mechanism 构造记忆机制。
    支持 zero_shot、stream_icl、mem0、mems、awmpro。

    注意：无论配置什么 memory_mechanism，都会返回对应的实例用于 update_memory。
    但在 offline 模式下，use_memory 会强制使用 zero_shot（见 main 函数）。
    """
    mem_cfg = cfg.memory_mechanism or {}
    name = mem_cfg.get("name", "zero_shot")
    config_path = mem_cfg.get("config_path")

    if name == "zero_shot":
        if not config_path:
            config_path = ROOT_DIR / "memory" / "zero_shot" / "zero_shot.yaml"
        else:
            config_path = ROOT_DIR / config_path
        return load_zero_shot_from_yaml(str(config_path))
    elif name == "streamICL":
        if not config_path:
            config_path = ROOT_DIR / "memory" / "streamICL" / "streamICL.yaml"
        else:
            config_path = ROOT_DIR / config_path
        from memory.streamICL.streamICL import load_stream_icl_from_yaml
        return load_stream_icl_from_yaml(str(config_path))
    elif name == "mem0":
        if not config_path:
            config_path = ROOT_DIR / "memory" / "mem0" / "mem0.yaml"
        else:
            config_path = ROOT_DIR / config_path
        from memory.mem0.mem0 import load_mem0_from_yaml
        return load_mem0_from_yaml(str(config_path))
    elif name == "mems":
        if not config_path:
            config_path = ROOT_DIR / "memory" / "MEMs" / "MEMs.yaml"
        else:
            config_path = ROOT_DIR / config_path
        from memory.MEMs.MEMs import load_mems_from_yaml
        return load_mems_from_yaml(str(config_path))
    elif name == "awmPro":
        if not config_path:
            config_path = ROOT_DIR / "memory" / "awmPro" / "awmPro.yaml"
        else:
            config_path = ROOT_DIR / config_path
        from memory.awmPro.awmPro import load_awmpro_from_yaml
        return load_awmpro_from_yaml(str(config_path))
    else:
        raise NotImplementedError(f"Memory mechanism '{name}' not implemented yet (supported: zero_shot, stream_icl, mem0, mems, awmpro).")


def build_execution_engine_from_config(cfg: ExperimentConfig):
    """
    根据 assignment.yaml 中的 execution_method 构造执行引擎。
    支持 single_agent。
    """
    exec_cfg = cfg.execution_method or {}
    name = exec_cfg.get("name", "single_agent")
    config_path = exec_cfg.get("config_path")

    if name == "single_agent":
        if not config_path:
            config_path = ROOT_DIR / "execution" / "single_agent" / "single_agent.yaml"
        else:
            config_path = ROOT_DIR / config_path
        return load_single_agent_engine_from_yaml(str(config_path))
    else:
        raise NotImplementedError(f"Execution method '{name}' not implemented yet (supported: single_agent).")


def ensure_output_dir(base: Path) -> Path:
    """确保输出目录存在"""
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_schedule_from_config(
    exp_cfg: ExperimentConfig,
    backend,
    locomo_task_instance=None,
    locomo_task_name=None
):
    """
    根据配置构建调度序列。

    Args:
        exp_cfg: 实验配置
        backend: 后端客户端
        locomo_task_instance: locomo 任务实例（可选）
        locomo_task_name: locomo 任务名称（可选）

    Returns:
        (schedule, task_to_indices, replay_info) 元组
    """
    from src.runner.schedule_utils import (
        build_transfer_schedule,
        build_replay_schedule,
        build_replay_schedule_for_locomo,
        build_mixed_schedule,
        build_locomo_session_schedule,
    )
    from src.client.scheduler import build_schedule

    # 获取配置
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    cross_task = exp_cfg.experiment.get("cross_task", False)
    shuffle_enabled = exp_cfg.experiment.get("shuffle", False)
    seed = exp_cfg.experiment.get("seed")

    # 获取任务列表
    tasks_cfg = exp_cfg.tasks
    task_names = [t["name"] for t in tasks_cfg if "name" in t]

    # 获取每个任务的可用样本索引
    task_to_indices = {}
    for task_name in task_names:
        try:
            indices = backend.get_indices(task_name)
            task_to_indices[task_name] = indices
        except Exception as e:
            print(f"Warning: Failed to get indices for task {task_name}: {e}")
            task_to_indices[task_name] = []

    replay_info = None

    # 根据 training_mode 构建不同的调度
    if training_mode == "transfer":
        # Transfer 模式
        transfer_task = exp_cfg.experiment.get("transfer_task")
        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
        schedule = build_transfer_schedule(
            task_to_indices=task_to_indices,
            transfer_task=transfer_task,
            transfer_after_task=transfer_after_task,
            shuffle_enabled=shuffle_enabled,
            seed=seed,
        )
    elif training_mode == "replay":
        # Replay 模式
        if locomo_task_name:
            # Locomo 任务的 replay
            schedule, replay_info = build_replay_schedule_for_locomo(
                task_name=locomo_task_name,
                task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # 普通任务的 replay
            task_name = task_names[0]
            schedule, replay_info = build_replay_schedule(
                task_name=task_name,
                sample_indices=task_to_indices[task_name],
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
    elif locomo_task_name and locomo_task_instance:
        # Locomo 任务的混合调度
        system_memory_tasks = {k: v for k, v in task_to_indices.items() if k != locomo_task_name}
        if system_memory_tasks:
            # 有其他任务，使用混合调度
            schedule = build_mixed_schedule(
                system_memory_tasks=system_memory_tasks,
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # 只有 locomo 任务，使用 session 调度
            schedule = build_locomo_session_schedule(
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
    else:
        # 默认：使用 scheduler 的 build_schedule
        from src.client.scheduler import ScheduleConfig
        schedule_cfg = ScheduleConfig(
            cross_task=cross_task,
            shuffle=shuffle_enabled,
            seed=seed,
        )
        schedule = build_schedule(task_to_indices, schedule_cfg)

    return schedule, task_to_indices, replay_info
