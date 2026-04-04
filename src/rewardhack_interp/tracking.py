from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rewardhack_interp.config import WandbConfig
from rewardhack_interp.utils.paths import ensure_dir


class WandbTracker:
    def __init__(self, run: Any | None, config: WandbConfig | None) -> None:
        self.run = run
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.run is not None and self.config is not None and self.config.enabled

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        if step is None:
            self.run.log(values)
        else:
            self.run.log(values, step=step)

    def log_summary(self, values: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key, value in flatten_metrics(values).items():
            self.run.summary[key] = value

    def log_path(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        if not self.enabled or self.config is None or not self.config.log_artifacts:
            return

        import wandb

        resolved = Path(path)
        if not resolved.exists():
            return

        artifact_name = name or build_artifact_name(
            prefix=self.config.artifact_name_prefix,
            run_name=self.run.name or self.run.id,
            label=resolved.stem or resolved.name,
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata,
        )
        if resolved.is_dir():
            artifact.add_dir(str(resolved))
        else:
            artifact.add_file(str(resolved))
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])


@contextmanager
def start_wandb_run(
    *,
    wandb_config: WandbConfig | None,
    run_name: str,
    job_type: str,
    config_payload: dict[str, Any],
) -> Iterator[WandbTracker]:
    if wandb_config is None or not wandb_config.enabled:
        yield WandbTracker(run=None, config=wandb_config)
        return

    import wandb

    ensure_dir(wandb_config.dir)
    with wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project,
        dir=str(wandb_config.dir),
        name=wandb_config.name or run_name,
        notes=wandb_config.notes,
        tags=wandb_config.tags,
        config=config_payload,
        group=wandb_config.group,
        job_type=wandb_config.job_type or job_type,
        mode=wandb_config.mode,
        reinit="finish_previous",
        save_code=wandb_config.save_code,
    ) as run:
        yield WandbTracker(run=run, config=wandb_config)


def ensure_wandb_report_to(
    report_to: list[str],
    wandb_config: WandbConfig | None,
) -> list[str]:
    merged = list(report_to)
    if wandb_config is not None and wandb_config.enabled and "wandb" not in merged:
        merged.append("wandb")
    return merged


def flatten_metrics(
    payload: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=flat_key))
        else:
            flat[flat_key] = value
    return flat


def build_artifact_name(*, prefix: str, run_name: str, label: str) -> str:
    raw = f"{prefix}-{run_name}-{label}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-").lower()
