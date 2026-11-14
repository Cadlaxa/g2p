# coding=utf-8
import os
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    Callback,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
import k2
import optuna
from optuna.integration import PyTorchLightningPruningCallback


class MetricsCallback(Callback):
    """Callback to track metrics for Optuna pruning."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        current_per = trainer.callback_metrics.get("val/per", None)
        if current_per is not None:
            self.metrics.append(float(current_per))


@hydra.main(version_base=None, config_path="configs", config_name="cfg")
def train(cfg: DictConfig):
    assert cfg.symbols.num_graphemes == len(
        cfg.symbols.graphemes
    ), f"{cfg.symbols.num_graphemes} != {len(cfg.symbols.graphemes)}"
    assert cfg.symbols.num_phonemes == len(
        cfg.symbols.phonemes
    ), f"{cfg.symbols.num_phonemes} != {len(cfg.symbols.phonemes)}"

    torch.set_float32_matmul_precision("high")

    dataset = hydra.utils.instantiate(cfg.dataset)

    from g2p_module import G2pModule
    from data_module import G2pDataModule

    datamodule = G2pDataModule(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        eval_split=cfg.train.eval_split,
        seed=cfg.train.get("seed", 42),
    )

    training_module = G2pModule(
        model=cfg.model,
        muon_lr=cfg.train.muon_lr,
        adam_lr=cfg.train.adam_lr,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler=cfg.train.lr_scheduler,
        ilm_loss_weight=cfg.train.ilm_loss_weight,
        ilm_scheduled_sampling_ratio=cfg.train.ilm_scheduled_sampling_ratio,
        grad_clip=cfg.train.grad_clip,
        loss_device=(
            "cuda" if k2.with_cuda and cfg.train.accelerator == "gpu" else "cpu"
        ),
        graphemes=OmegaConf.to_object(dataset.graphemes),
        phonemes=OmegaConf.to_object(dataset.phonemes),
        print_debug_preview=cfg.train.print_debug_preview,
        logs_stressless_error_rates=cfg.train.get("logs_stressless_error_rates", False),
        train_eval_interval=cfg.train.get("train_eval_interval", 1),
        num_preview_words=cfg.train.get("num_preview_words", 15),
    )

    output_dir = HydraConfig.get().runtime.output_dir

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor="val/score",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="g2p-best-score",
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        every_n_epochs=cfg.train.save_interval,
        filename="g2p-{epoch:03d}",
        save_top_k=-1,
        save_weights_only=True,
    )
    periodic_checkpoint.CHECKPOINT_EQUALS_CHAR = "-"
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    callbacks_list = [
        checkpoint_callback,
        periodic_checkpoint,
        lr_monitor,
        progress_bar,
    ]

    metrics_callback = MetricsCallback()
    callbacks_list.append(metrics_callback)

    trial = None
    try:
        trial = optuna.integration.get_current_trial()
        if trial is not None:
            print(f"Running Optuna trial {trial.number}")
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/per")
            callbacks_list.append(pruning_callback)
            optuna_tb_callback = optuna.integration.TensorBoardCallback(
                log_dir=output_dir, metric_name="val/per"
            )
            callbacks_list.append(optuna_tb_callback)
    except:
        pass

    tensorboard_logger = TensorBoardLogger(save_dir=output_dir, name="", version="")

    default_precision = "bf16-mixed" if cfg.train.accelerator == "gpu" else "32-true"
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=1,
        precision=cfg.train.get("precision", default_precision),
        check_val_every_n_epoch=cfg.train.get("val_interval", 1),
        callbacks=callbacks_list,
        logger=tensorboard_logger,
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    resume_path = os.environ.get("RESUME_CKPT")
    if resume_path and os.path.exists(resume_path):
        print(f"\n🚀 RESUMING TRAINING FROM: {resume_path}\n")
    else:
        resume_path = None

    # 1. PyTorch 2.6 Bypass: Intercept torch.load for weights_only=False
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    # 2. OneCycleLR Step Expansion when resuming
    import torch.optim.lr_scheduler as lr_sched

    if hasattr(lr_sched, "OneCycleLR"):
        _orig_get_lr = lr_sched.OneCycleLR.get_lr

        def _patched_get_lr(self):
            if self.last_epoch > self.total_steps:
                self.total_steps = self.last_epoch + 1000
            return _orig_get_lr(self)

        lr_sched.OneCycleLR.get_lr = _patched_get_lr

    trainer.fit(training_module, datamodule=datamodule, ckpt_path=resume_path)
    return float(training_module.score_min)  # optuna objective


if __name__ == "__main__":
    train()