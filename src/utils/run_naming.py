import datetime
from typing import Any, Dict


def build_run_name(cfg: Dict[str, Any]) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = cfg["model"]["name"]
    img_tag = f"{cfg['data']['input_size']}px"
    aug_tag = cfg["augmentations"]["name"]
    reg_tag = cfg["regularization"].get("tag", "reg")
    ls_value = cfg["loss"].get("label_smoothing", 0.0)
    scheduler_tag = cfg["scheduler"]["name"]
    return (
        f"{timestamp}_{model_tag}_{img_tag}_"
        f"{aug_tag}_{reg_tag}_ls{ls_value:.2f}_{scheduler_tag}"
    )


def build_submission_name(
    cfg: Dict[str, Any],
    metric_value: float,
    leaderboard_score: float,
    base_name: str = None,
) -> str:
    base = base_name or build_run_name(cfg)
    return f"{base}_f1_{metric_value:.4f}_result({leaderboard_score:.4f})"
