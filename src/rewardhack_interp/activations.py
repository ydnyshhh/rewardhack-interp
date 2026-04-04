from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from rewardhack_interp.config import ActivationCaptureConfig
from rewardhack_interp.io import append_jsonl, read_json, write_json
from rewardhack_interp.types import ActivationCaptureArtifact, TrajectoryArtifact
from rewardhack_interp.utils.paths import ensure_dir, to_path_string


class ActivationCaptureRunner:
    def __init__(self, loaded_model: Any, config: ActivationCaptureConfig) -> None:
        self.loaded_model = loaded_model
        self.config = config

    def capture_from_rollout_record(
        self,
        record: TrajectoryArtifact,
        *,
        output_dir: str | Path | None = None,
    ) -> ActivationCaptureArtifact:
        import torch

        resolved_output_dir = ensure_dir(output_dir or self.config.output_dir)
        token_ids = record.generation.prompt_token_ids + record.generation.completion_token_ids
        model = self.loaded_model.model
        model_device = next(model.parameters()).device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=model_device)
        attention_mask = torch.ones_like(input_ids)

        hooked_modules = resolve_module_names(model, self.config.module_globs)
        captured_tensors: dict[str, Any] = {}
        hooks = []
        for name, module in model.named_modules():
            if name in hooked_modules:
                hooks.append(
                    module.register_forward_hook(
                        lambda _module, _args, output, *, module_name=name: capture_hook(
                            captured_tensors,
                            module_name,
                            output,
                            self.config.capture_dtype,
                        )
                    )
                )

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=self.config.capture_hidden_states,
                    use_cache=False,
                )
        finally:
            for hook in hooks:
                hook.remove()

        if self.config.capture_hidden_states:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise ValueError(
                    "Model did not return hidden states; "
                    "capture_hidden_states requires them."
                )
            for hidden_index, hidden_state in enumerate(hidden_states):
                if hidden_index == 0 and not self.config.include_embedding_state:
                    continue
                if hidden_index > 0 and self.config.hidden_state_layers is not None:
                    layer_index = hidden_index - 1
                    if layer_index not in self.config.hidden_state_layers:
                        continue
                tensor_name = (
                    "hidden_state.embed"
                    if hidden_index == 0
                    else f"hidden_state.layer_{hidden_index - 1}"
                )
                captured_tensors[tensor_name] = to_capture_tensor(
                    hidden_state,
                    self.config.capture_dtype,
                )

        captured_tensors["token_ids"] = input_ids.detach().cpu()[0]
        captured_tensors["attention_mask"] = attention_mask.detach().cpu()[0]
        tensor_path = resolved_output_dir / f"{record.trace_id}.pt"
        metadata_path = resolved_output_dir / f"{record.trace_id}.json"
        torch.save(captured_tensors, tensor_path)

        artifact = ActivationCaptureArtifact(
            trace_id=record.trace_id,
            rollout_run_id=record.run_id,
            source_model_name_or_path=record.model_name_or_path,
            source_adapter_name_or_path=record.adapter_name_or_path,
            cohort=record.cohort,
            tensor_path=to_path_string(tensor_path),
            prompt_token_count=record.generation.prompt_token_count,
            completion_token_count=record.generation.completion_token_count,
            tensor_names=sorted(
                name
                for name in captured_tensors
                if name not in {"token_ids", "attention_mask"}
            ),
            tensor_shapes={
                name: list(tensor.shape)
                for name, tensor in captured_tensors.items()
                if name not in {"token_ids", "attention_mask"}
            },
            token_ids=[int(token_id) for token_id in token_ids],
            capture_mode=self.config.capture_mode,
            module_globs=list(self.config.module_globs),
        )
        write_json(metadata_path, artifact.model_dump(mode="json"))
        return artifact


def capture_manifest_from_rollouts(
    loaded_model: Any,
    config: ActivationCaptureConfig,
    records: list[TrajectoryArtifact],
    *,
    manifest_path: str | Path,
) -> Path:
    runner = ActivationCaptureRunner(loaded_model=loaded_model, config=config)
    artifacts = [runner.capture_from_rollout_record(record) for record in records]
    return append_jsonl(manifest_path, [artifact.model_dump(mode="json") for artifact in artifacts])


def load_activation_artifact(path: str | Path) -> ActivationCaptureArtifact:
    return ActivationCaptureArtifact.model_validate(read_json(path))


def load_activation_tensors(artifact: ActivationCaptureArtifact) -> dict[str, Any]:
    import torch

    return torch.load(artifact.tensor_path, map_location="cpu")


def resolve_module_names(model: Any, globs: list[str]) -> set[str]:
    if not globs:
        return set()
    names = {name for name, _module in model.named_modules()}
    return {name for name in names if any(fnmatchcase(name, pattern) for pattern in globs)}


def capture_hook(
    captured_tensors: dict[str, Any],
    module_name: str,
    output: Any,
    capture_dtype: str,
) -> None:
    tensor = first_tensor(output)
    if tensor is None:
        return
    captured_tensors[f"module.{module_name}"] = to_capture_tensor(tensor, capture_dtype)


def first_tensor(output: Any) -> Any | None:
    try:
        import torch
    except ImportError:
        return None

    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        for item in output:
            if torch.is_tensor(item):
                return item
    return None


def to_capture_tensor(tensor: Any, capture_dtype: str) -> Any:
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    detached = tensor.detach().cpu()
    if detached.ndim >= 1 and detached.shape[0] == 1:
        detached = detached.squeeze(0)
    if detached.is_floating_point():
        detached = detached.to(dtype_map[capture_dtype])
    return detached.contiguous()
