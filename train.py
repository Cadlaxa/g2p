# coding=utf-8
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
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
    )

    output_dir = HydraConfig.get().runtime.output_dir

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        monitor="val/score",
        mode="min",
        save_top_k=1,
        filename="g2p-best-score",
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        every_n_epochs=cfg.train.save_interval,
        filename="g2p-{epoch:03d}",
        save_top_k=-1,
    )
    periodic_checkpoint.CHECKPOINT_EQUALS_CHAR = "-"
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks_list = [checkpoint_callback, periodic_checkpoint, lr_monitor]

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

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=1,
        callbacks=callbacks_list,
        logger=tensorboard_logger,
        gradient_clip_val=cfg.train.grad_clip,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(training_module, datamodule=datamodule)
    return float(training_module.score_min)  # optuna objective


if __name__ == "__main__":
    train()
