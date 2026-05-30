from __future__ import annotations

from rewardhack_gym.core.models import Task


class PrivateTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def clear(self) -> None:
        self._tasks.clear()
