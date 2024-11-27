import copy
from datetime import timedelta
from pathlib import Path

import pandas as pd

from model_configs.constant import (
    FILENAME_DATE_COL,
    FILENAME_DATE_FORMAT,
    FULL_TRAINED_LR,
    MAX_FULL_TRAINED_EPOCHS,
)
from recmodel.base.feature_manager import (
    FeatureManagerCollection,
    FromFileFeatureManager,
)
from recmodel.base.schemas.pipeline_config import PipelineConfig
from recmodel.base.utils.config import parse_pipeline_config
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.logger import logger
from recmodel.base.utils.mlflow import MLflowMaster
from recmodel.base.utils.utils import return_or_load
from recmodel.dataloader.dl_feature_assembly import FeatureAssembly
from recmodel.model.evaluation import Evaluation
from recmodel.model.model_pipelines import PytorchPipeline


class CoreRecModel:
    def __init__(
        self,
        infer_date: int,
        num_days_to_train: int,
        model_config,
        device_id: int,
        process_lib: str,
    ):
        self.infer_date = infer_date
        self.date_before_infer = (
            pd.to_datetime(self.infer_date, format=FILENAME_DATE_FORMAT)
            - timedelta(days=1)
        ).strftime(FILENAME_DATE_FORMAT)
        self.num_days_to_train = num_days_to_train
        self.model_config = model_config

        self.modeling_dir = (
            Path("experiments")
            / "daily"
            / self.model_config.config_name
            / str(self.infer_date)
        )

        if device_id != -1:
            self.process_lib = "cudf"
            GpuLoading(device_id)
        else:
            self.process_lib = process_lib

        FeatureAssembly(
            self.model_config.data_loader.user_id_col,
            self.model_config.data_loader.item_id_col,
            self.model_config.data_loader.user_context_feature_names,
            self.model_config.data_loader.item_context_feature_names,
            self.model_config.data_loader.time_order_event_col,
        )
        self.fulltrained_model_config = self.process_fulltrained_model_config()
        self.finetuned_model_config = self.process_finetuned_model_config()

    def process_fulltrained_model_config(self):
        """Add date filter to model config. Using the last day for validation and x
        days before for training."""
        _infer_date = pd.to_datetime(self.infer_date, format=FILENAME_DATE_FORMAT)
        val_date = int(self.date_before_infer)
        train_start_date = int(
            (_infer_date - timedelta(days=1 + self.num_days_to_train)).strftime(
                FILENAME_DATE_FORMAT
            )
        )
        val_date_filters = [f"{FILENAME_DATE_COL} == {val_date}"]
        train_date_filters = [
            f"{FILENAME_DATE_COL} >= {train_start_date}",
            f"{FILENAME_DATE_COL} < {val_date}",
        ]
        fulltrained_model_config = copy.deepcopy(self.model_config)
        fulltrained_model_config.data_loader.train_filters += train_date_filters
        fulltrained_model_config.data_loader.validation_filters += val_date_filters
        fulltrained_model_config.data_loader.content_filters += [
            f"{FILENAME_DATE_COL} <= {val_date}",
        ]
        fulltrained_model_config.model_wrapper.fit_params[
            "max_epochs"
        ] = MAX_FULL_TRAINED_EPOCHS
        fulltrained_model_config.model_wrapper.model_params.optimizer.params[
            "lr"
        ] = FULL_TRAINED_LR
        return fulltrained_model_config

    def process_finetuned_model_config(self):
        """Add date filter to model config. Using data of the day before the infer date
        to fine-tune model."""
        _infer_date = pd.to_datetime(self.infer_date, format=FILENAME_DATE_FORMAT)
        infer_date = int(_infer_date.strftime(FILENAME_DATE_FORMAT))
        train_date = int(self.date_before_infer)
        train_date_filters = [
            f"{FILENAME_DATE_COL} >= {train_date}",
            f"{FILENAME_DATE_COL} <= {infer_date}",
        ]
        finetuned_model_config = copy.deepcopy(self.model_config)
        finetuned_model_config.data_loader.train_filters += train_date_filters
        # No need validation dataset currently.
        finetuned_model_config.data_loader.validation_filters += train_date_filters
        finetuned_model_config.data_loader.content_filters += [
            f"{FILENAME_DATE_COL} <= {infer_date}"
        ]
        return finetuned_model_config

    def train_from_scratch(self):
        """Train model from scratch."""
        date_before_infer = pd.to_datetime(self.infer_date, format=FILENAME_DATE_FORMAT)
        date_before_infer = (date_before_infer - timedelta(days=1)).strftime(
            FILENAME_DATE_FORMAT
        )
        logger.warning("No model checkpoint found. Train from scratch.")
        PytorchPipeline(
            self.fulltrained_model_config,
            custom_run_dir=(self.modeling_dir.parent / date_before_infer).as_posix(),
            process_lib=self.process_lib,
            model_date=date_before_infer,
        ).run()

    def fine_tune(self, most_recent_model_checkpoint):
        logger.info(f"Fine-tune model from checkpoint: {most_recent_model_checkpoint}")
        PytorchPipeline(
            self.finetuned_model_config,
            custom_run_dir=self.modeling_dir.as_posix(),
            process_lib=self.process_lib,
            model_date=str(self.infer_date),
        ).fine_tune(model_path=most_recent_model_checkpoint)

    def run(self):
        list_registered_models = [
            model.name for model in MLflowMaster().mlflow.search_registered_models()
        ]
        current_model_name = (
            self.model_config.model_wrapper.cls_name.split(".")[-1]
            .lower()
            .replace("modelwrapper", "")
        )
        if (
            f"{current_model_name}_{self.date_before_infer}_fulltrained"
            not in list_registered_models
        ):
            self.train_from_scratch()
        self.fine_tune(
            f"models:/{current_model_name}_{self.date_before_infer}_fulltrained/latest"
        )

    def evaluate(self):
        Evaluation(
            model_pipeline_config_path=self.model_config,
            model_root_dir=self.modeling_dir.parent,
            num_days_to_train=self.num_days_to_train,
            process_lib=self.process_lib,
        ).evaluate(for_date=self.infer_date)


class InferRecModel:
    def __init__(
        self,
        model_config,
        infer_date: int,
        device_id: int,
        process_lib: str,
    ):
        if device_id != -1:
            self.process_lib = "cudf"
            GpuLoading(device_id)
        else:
            self.process_lib = process_lib
        self.model_config = return_or_load(
            model_config, PipelineConfig, parse_pipeline_config
        )
        self.current_model_name = (
            self.model_config.model_wrapper.cls_name.split(".")[-1]
            .lower()
            .replace("modelwrapper", "")
        )
        FeatureAssembly(
            self.model_config.data_loader.user_id_col,
            self.model_config.data_loader.item_id_col,
            self.model_config.data_loader.user_context_feature_names,
            self.model_config.data_loader.item_context_feature_names,
            self.model_config.data_loader.time_order_event_col,
        )
        self.infer_date = infer_date
        self.mlflow_master = MLflowMaster()
        self.pipeline = self.load_model()

    def load_model(self):
        pipeline = PytorchPipeline(self.model_config, process_lib=self.process_lib)
        pipeline.load_model(
            f"models:/{self.current_model_name}_{self.infer_date}_finetuned/latest"
        )
        return pipeline


class TestRecModel(CoreRecModel):
    def __init__(
        self,
        infer_date: int,
        num_days_to_train: int,
        config_path: str,
        device_id: int,
        data_path: str,
        cpu_process_lib: str = "pandas",
    ):
        model_config = parse_pipeline_config(config_path)
        super().__init__(
            infer_date=infer_date,
            num_days_to_train=num_days_to_train,
            model_config=model_config,
            device_id=device_id,
            process_lib=cpu_process_lib,
        )
        if device_id == 1:
            process_lib_list = [cpu_process_lib]
        else:
            process_lib_list = [cpu_process_lib, "cudf"]

        self.data_path = Path(data_path)
        FeatureManagerCollection(
            user_fm={
                process_lib: FromFileFeatureManager(
                    process_lib=process_lib,
                    dataset_path=self.data_path / "user_features_dhash",
                )
                for process_lib in process_lib_list
            },
            account_fm={
                process_lib: FromFileFeatureManager(
                    process_lib=process_lib,
                    dataset_path=self.data_path / "account_features",
                )
                for process_lib in process_lib_list
            },
            item_fm={
                process_lib: FromFileFeatureManager(
                    process_lib=process_lib,
                    dataset_path=self.data_path / "mytv_content_features_dhash",
                )
                for process_lib in process_lib_list
            },
            online_user_fm={
                process_lib: FromFileFeatureManager(
                    process_lib=process_lib,
                    dataset_path=self.data_path / "online_user_features_dhash",
                )
                for process_lib in process_lib_list
            },
            online_item_fm={
                process_lib: FromFileFeatureManager(
                    process_lib=process_lib,
                    dataset_path=self.data_path / "online_content_features_dhash",
                )
                for process_lib in process_lib_list
            },
            interacted_fm={
                process_lib: FromFileFeatureManager(
                    process_lib=process_lib,
                    dataset_path=self.data_path / "daily_movie_vod_dl_features",
                )
                for process_lib in process_lib_list
            },
        )
