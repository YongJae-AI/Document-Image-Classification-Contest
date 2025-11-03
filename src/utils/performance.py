import torch


def configure_performance(cfg):
    perf_cfg = cfg.get("performance", {})

    if torch.cuda.is_available():
        allow_tf32 = perf_cfg.get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        precision = perf_cfg.get("matmul_precision")
        if precision:
            torch.set_float32_matmul_precision(precision)

    if perf_cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
