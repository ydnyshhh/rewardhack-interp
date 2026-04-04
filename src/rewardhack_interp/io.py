from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from rewardhack_interp.utils.paths import ensure_parent_dir

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = ensure_parent_dir(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return output_path


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_models(path: str | Path, model_cls: type[ModelT]) -> list[ModelT]:
    return [model_cls.model_validate(row) for row in read_jsonl(path)]
