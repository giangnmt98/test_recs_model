import os
from datetime import timedelta
from typing import Optional

from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint

from recmodel.base.utils.mlflow import MLflowMaster


class CustomModelCheckpiont(ModelCheckpoint):
    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
        )
        self.model_name = filename

    def _save_checkpoint(self, trainer, filepath):
        if os.getenv("TEST_MODE") == "yes":
            code_paths = ["../recmodel", "../model_configs"]
        else:
            code_paths = ["recmodel", "model_configs"]
        MLflowMaster().mlflow.pytorch.log_model(
            trainer.model,
            artifact_path=self.model_name,
            registered_model_name=self.model_name,
            code_paths=code_paths,
        )
