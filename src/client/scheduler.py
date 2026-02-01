from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


TaskName = str
SampleIndex = int
Schedule = List[Tuple[TaskName, SampleIndex]]


@dataclass
class ScheduleConfig:
    cross_task: bool
    shuffle: bool
    seed: int | None = None


def build_schedule(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    config: ScheduleConfig,
) -> Schedule:
    """
    Build a unified (task_name, sample_index) schedule according to the global
    lifelong-learning settings (cross_task, shuffle, interval, ...).
    """
    if not task_to_indices:
        return []

    if config.cross_task and not config.shuffle:
        # 不支持 cross_task=True, shuffle=False 的情况
        raise ValueError(
            "cross_task=True and shuffle=False is not supported. "
            "Please use either cross_task=False or shuffle=True."
        )

    if not config.cross_task and not config.shuffle:
        # Case A: per-task in-order
        return _schedule_sequential(task_to_indices)

    if not config.cross_task and config.shuffle:
        # Case B: per-task shuffle, no cross-task mixing
        return _schedule_sequential_shuffled(task_to_indices, config.seed)

    # Case C: cross_task=True and shuffle=True -> global shuffle of all samples
    return _schedule_global_shuffle(task_to_indices, config.seed)


def _schedule_sequential(task_to_indices: Dict[TaskName, Sequence[SampleIndex]]) -> Schedule:
    schedule: Schedule = []
    for task, indices in task_to_indices.items():
        for idx in indices:
            schedule.append((task, idx))
    return schedule


def _schedule_sequential_shuffled(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    seed: int | None,
) -> Schedule:
    import random

    rng = random.Random(seed)
    schedule: Schedule = []
    for task, indices in task_to_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for idx in shuffled:
            schedule.append((task, idx))
    return schedule


def _schedule_round_robin(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    interval: int,
) -> Schedule:
    """
    Cross-task, no shuffle: each task contributes up to `interval` consecutive
    samples in turn, preserving the original order inside each task.
    """
    from collections import deque

    # Work on mutable queues of indices per task
    task_queues = {
        task: deque(indices) for task, indices in task_to_indices.items() if indices
    }
    tasks = list(task_queues.keys())
    if not tasks:
        return []

    schedule: Schedule = []
    task_idx = 0

    while task_queues:
        task = tasks[task_idx % len(tasks)]
        queue = task_queues.get(task)
        if not queue:
            # This task has been exhausted; remove it from rotation
            tasks.remove(task)
            if not tasks:
                break
            continue

        # Take up to `interval` items from this task
        for _ in range(interval):
            if not queue:
                break
            sample_idx = queue.popleft()
            schedule.append((task, sample_idx))

        # Drop empty queues
        if not queue:
            task_queues.pop(task, None)
            tasks = [t for t in tasks if t in task_queues]

        task_idx += 1

    return schedule


def _schedule_global_shuffle(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    seed: int | None,
) -> Schedule:
    import random

    all_pairs: Schedule = []
    for task, indices in task_to_indices.items():
        for idx in indices:
            all_pairs.append((task, idx))

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs


