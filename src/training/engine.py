import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from src.utils.ema import ModelEma
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # tensorboard not installed
    SummaryWriter = None  # type: ignore
from src.utils.metrics import AverageMeter, MetricTracker


class Trainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metric_tracker: MetricTracker,
        logger,
        run_dir: Path,
        channels_last: bool = False,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_tracker = metric_tracker
        self.logger = logger
        self.run_dir = run_dir
        self.channels_last = channels_last
        training_cfg = cfg.get("training", {})
        self.total_epochs = int(training_cfg.get("epochs", 1))
        self.max_train_steps = training_cfg.get("max_train_steps")
        self.max_val_batches = training_cfg.get("max_val_batches")
        self.early_stopping_cfg = training_cfg.get("early_stopping") or {}
        self.early_mode = self.early_stopping_cfg.get("mode", "max")
        self.early_patience = self.early_stopping_cfg.get("patience")
        self.early_min_delta = self.early_stopping_cfg.get("min_delta", 0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)

        self.scaler = GradScaler(
            enabled=cfg["training"].get("amp", False) and torch.cuda.is_available()
        )

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"

        # TensorBoard 지원 (선택)
        tb_enabled = bool(cfg.get("logging", {}).get("tensorboard", False))
        self.tb_writer = None
        if tb_enabled and SummaryWriter is not None:
            log_dir = self.run_dir / "tb"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))

        # 최근 검증 per-class 통계 캐시
        self._last_per_class = None  # type: ignore
        self._last_confusion = None  # type: ignore

        ema_cfg = training_cfg.get("ema") or {}
        self.ema: Optional[ModelEma] = None
        if ema_cfg.get("enabled", False):
            decay = float(ema_cfg.get("decay", 0.9999))
            self.ema = ModelEma(self.model, decay=decay)

        swa_cfg = training_cfg.get("swa") or {}
        self.swa_model: Optional[AveragedModel] = None
        self.swa_start_epoch = 0
        self.swa_update_interval = 1
        self.swa_updates = 0
        if swa_cfg.get("enabled", False):
            self.swa_start_epoch = int(
                swa_cfg.get("start_epoch", max(1, self.total_epochs - 5))
            )
            self.swa_update_interval = int(swa_cfg.get("update_interval", 1))
            self.swa_model = AveragedModel(self.model)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> float:
        best_metric = float("-inf") if self.early_mode == "max" else float("inf")
        best_epoch = -1
        patience_counter = 0

        epochs = self.cfg["training"]["epochs"]
        log_interval = self.cfg["logging"]["log_interval"]

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch, log_interval)
            val_loss, metric = self._validate(valid_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                self.metric_tracker.primary_metric: metric,
                "lr": current_lr,
            }
            self._log_metrics(epoch_summary)

            # Print top-10 most misclassified classes for this epoch (descending by incorrect count)
            try:
                pc = self._last_per_class or {}
                totals = pc.get("per_class_total", [])
                corrects = pc.get("per_class_correct", [])
                if totals and corrects and len(totals) == len(corrects):
                    mis = [(i, int(t - c), int(t), (float(c) / float(t)) if t else None)
                           for i, (t, c) in enumerate(zip(totals, corrects))]
                    # sort by incorrect desc, then by total desc
                    mis.sort(key=lambda x: (x[1], x[2]), reverse=True)
                    topk = mis[:10]
                    # build human-readable line
                    parts = []
                    for cls, inc, tot, acc in topk:
                        if tot == 0:
                            parts.append(f"{cls}: -/-")
                        else:
                            parts.append(f"{cls}: {inc}/{tot} (acc={acc:.3f})")
                    self.logger.info("[Epoch %d] Top misclassified classes: %s", epoch, ", ".join(parts))
            except Exception:
                pass

            # per-class 최신 통계 파일을 에폭명으로 스냅샷
            try:
                latest_path = self.run_dir / "per_class_latest.json"
                epoch_path = self.run_dir / f"per_class_epoch{epoch:02d}.json"
                if latest_path.exists():
                    epoch_path.write_text(latest_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

            # Save confusion matrix artifacts (npy + png) and per-class CSV for this epoch
            try:
                from pathlib import Path as _P
                import numpy as _np
                plots_dir = self.run_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                if self._last_confusion is not None:
                    cm = _np.array(self._last_confusion, dtype=_np.int64)
                    (_P(plots_dir) / f"confusion_epoch{epoch:02d}.npy").write_bytes(cm.tobytes())
                    # Also save as csv for Excel/PPT
                    import pandas as _pd
                    _pd.DataFrame(cm).to_csv(plots_dir / f"confusion_epoch{epoch:02d}.csv", index=False)
                    # Try plotting heatmap (optional)
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as _plt
                        import seaborn as _sns
                        _plt.figure(figsize=(7,6))
                        _sns.heatmap(cm, cmap='Blues')
                        _plt.title(f'Confusion (Val) True x Pred — epoch {epoch}')
                        _plt.xlabel('Pred'); _plt.ylabel('True')
                        _plt.tight_layout()
                        _plt.savefig(plots_dir / f"confusion_epoch{epoch:02d}.png", dpi=150)
                        _plt.close()
                    except Exception:
                        pass
                # Per-class CSV snapshot
                if self._last_per_class is not None:
                    import pandas as _pd
                    pc = self._last_per_class
                    rows = []
                    pct = pc.get("per_class_total", [])
                    pcc = pc.get("per_class_correct", [])
                    pca = pc.get("per_class_accuracy", [])
                    for i in range(len(pct)):
                        rows.append({"class": i, "total": pct[i], "correct": pcc[i], "accuracy": pca[i]})
                    _pd.DataFrame(rows).to_csv(plots_dir / f"per_class_epoch{epoch:02d}.csv", index=False)
            except Exception:
                pass

            # TensorBoard 기록
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("loss/val", val_loss, epoch)
                self.tb_writer.add_scalar(f"metric/{self.metric_tracker.primary_metric}", metric, epoch)
                self.tb_writer.add_scalar("lr", current_lr, epoch)

            improved = self._is_improved(metric, best_metric)

            if improved:
                best_metric = metric
                best_epoch = epoch
                self._save_checkpoint(epoch, best_metric)
                patience_counter = 0
            else:
                patience_counter += 1 if self.early_patience is not None else 0

            self.logger.info(
                "Epoch %d/%d - train_loss: %.4f - val_loss: %.4f - %s: %.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                self.metric_tracker.primary_metric,
                metric,
            )

            if (
                self.swa_model is not None
                and epoch >= self.swa_start_epoch
                and (epoch - self.swa_start_epoch) % self.swa_update_interval == 0
            ):
                self.swa_model.update_parameters(self.model)
                self.swa_updates += 1

            if (
                self.early_patience is not None
                and patience_counter >= self.early_patience
            ):
                self.logger.info(
                    "Early stopping triggered (patience %d)", self.early_patience
                )
                break

        if best_epoch == -1:
            best_metric = metric
            best_epoch = epoch

        if self.swa_model is not None and self.swa_updates > 0:
            self.logger.info("Evaluating SWA averaged weights")
            original_state = {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            }
            self.swa_model.to(self.device)
            self.swa_model.copy_to(self.model)
            swa_val_loss, swa_metric = self._validate(valid_loader)
            self.logger.info(
                "SWA - val_loss: %.4f - %s: %.4f",
                swa_val_loss,
                self.metric_tracker.primary_metric,
                swa_metric,
            )
            if self._is_improved(swa_metric, best_metric):
                best_metric = swa_metric
                best_epoch = epochs + 1
                self._save_checkpoint(best_epoch, best_metric)
            self.model.load_state_dict(original_state, strict=False)

        self.logger.info("Best epoch: %d (%.4f)", best_epoch, best_metric)
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        return best_metric

    def _train_one_epoch(
        self, loader: DataLoader, epoch: int, log_interval: int
    ) -> float:
        self.model.train()
        loss_meter = AverageMeter("train_loss")

        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = images.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.to(memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            grad_clip = self.cfg["training"].get("grad_clip_norm")
            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            loss_meter.update(loss.item(), images.size(0))

            if batch_idx % log_interval == 0 or batch_idx == len(loader):
                self.logger.info(
                    "[Epoch %d] Step %d/%d - loss: %.4f",
                    epoch,
                    batch_idx,
                    len(loader),
                    loss_meter.avg,
                )

            if self.max_train_steps and batch_idx >= self.max_train_steps:
                break

        return loss_meter.avg

    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        ema_applied = False
        if self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
            ema_applied = True

        self.model.eval()
        loss_meter = AverageMeter("val_loss")
        self.metric_tracker.reset()

        # per-class buckets
        num_classes = int(self.cfg["data"]["num_classes"])
        per_class_total = [0] * num_classes
        per_class_correct = [0] * num_classes

        with torch.no_grad():
            # confusion matrix
            cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
            for batch_idx, (images, targets) in enumerate(loader, start=1):
                images = images.to(self.device, non_blocking=True)
                if self.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                loss = self.criterion(outputs, targets)

                loss_meter.update(loss.item(), images.size(0))
                self.metric_tracker.update(targets, outputs)

                # accumulate per-class counts and confusion
                preds = outputs.argmax(dim=1)
                for t, p in zip(targets, preds):
                    ti = int(t.item())
                    per_class_total[ti] += 1
                    per_class_correct[ti] += int(ti == int(p.item()))
                    pi = int(p.item())
                    cm[ti][pi] += 1

                if self.max_val_batches and batch_idx >= self.max_val_batches:
                    break

        metric = self.metric_tracker.compute()

        # cache + write latest per-class stats
        try:
            acc = [(c / t) if t else None for c, t in zip(per_class_correct, per_class_total)]
            payload = {
                "per_class_total": per_class_total,
                "per_class_correct": per_class_correct,
                "per_class_accuracy": acc,
            }
            self._last_per_class = payload
            self._last_confusion = cm
            latest_path = self.run_dir / "per_class_latest.json"
            latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        if ema_applied:
            self.ema.restore(self.model)

        return loss_meter.avg, metric

    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "metric": metric,
        }
        if self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()
        if self.swa_model is not None and self.swa_updates > 0:
            checkpoint["swa_state"] = {
                k: v.detach().cpu() for k, v in self.swa_model.state_dict().items()
            }
        path = self.checkpoint_dir / "best.pth"
        torch.save(checkpoint, path)
        # 함께 per-class snapshot도 저장
        try:
            if self._last_per_class is not None:
                (self.run_dir / "per_class_best.json").write_text(
                    json.dumps(self._last_per_class, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            pass

    def _log_metrics(self, summary: Dict[str, Any]) -> None:
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

    def _is_improved(self, metric: float, best: float) -> bool:
        if self.early_mode == "max":
            if best == float("-inf"):
                return True
            return metric > best + self.early_min_delta
        else:
            if best == float("inf"):
                return True
            return metric < best - self.early_min_delta
