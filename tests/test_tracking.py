from rewardhack_interp.config import WandbConfig
from rewardhack_interp.tracking import ensure_wandb_report_to, flatten_metrics


def test_ensure_wandb_report_to_adds_wandb() -> None:
    config = WandbConfig(enabled=True, project="rewardhack-interp")
    report_to = ensure_wandb_report_to(["tensorboard"], config)
    assert report_to == ["tensorboard", "wandb"]


def test_flatten_metrics_flattens_nested_dicts() -> None:
    flat = flatten_metrics({"counts": {"reward_hack": 3}, "rows_written": 5})
    assert flat == {"counts/reward_hack": 3, "rows_written": 5}
