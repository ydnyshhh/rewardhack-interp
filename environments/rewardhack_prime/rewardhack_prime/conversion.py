from __future__ import annotations

from typing import Any

from rewardhack_gym.core.models import Task


def rewardhack_task_to_vf_task(task: Task) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": task.prompt}],
        "answer": "",
        "info": {
            "task_id": task.task_id,
            "family": task.family,
            "difficulty": task.difficulty,
            "expected_interface": task.expected_interface,
            "tags": list(task.tags),
            "exploit_surface": task.exploit_surface.to_dict(),
        },
    }
