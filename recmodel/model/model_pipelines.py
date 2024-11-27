from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from model_configs import constant as const
from recmodel.base.pipelines import BasePytorchPipeline, BaseTuningPipeline
from recmodel.base.schemas.pipeline_config import (
    DLModelParams,
    PipelineConfig,
    SearcherWrapper,
)
from recmodel.base.searcher_wrappers import PytorchSearcherWrapper
from recmodel.base.utils.config import parse_pipeline_config, parse_tuning_config
from recmodel.base.utils.utils import return_or_load
from recmodel.dataloader.dl import FeatureAssembly


class PytorchPipeline(BasePytorchPipeline):
    def __init__(
        self,
        config: Union[str, Path, PipelineConfig],
        custom_run_dir: Optional[str] = None,
        process_lib: str = "pandas",
        model_date: str = "",
    ):
        config = return_or_load(config, PipelineConfig, parse_pipeline_config)

        super(PytorchPipeline, self).__init__(
            config, custom_run_dir, process_lib=process_lib, model_date=model_date
        )

    def get_most_recent_model_checkpoint(self) -> str:
        """Get most recent model checkpoint before the infer date."""
        run_dir = self.exp_manager.run_dir
        run_date = int(run_dir.name.split("_")[0])
        modeling_dirs_before_infer_date = [
            d
            for d in run_dir.parent.iterdir()
            if d.is_dir()  # check if it is a directory
            and int(d.name.split("_")[0]) < run_date  # check before the current date
            and len(list(d.glob("*.ckpt"))) > 0  # check if it has checkpoint
        ]
        if len(modeling_dirs_before_infer_date) == 0:
            raise ValueError(
                f"No modeling directory before {run_dir.name} found. "
                f"Cannot fine-tune model."
            )

        most_recent_modeling_dir = sorted(modeling_dirs_before_infer_date)[-1]
        most_recent_model_checkpoint = sorted(most_recent_modeling_dir.glob("*.ckpt"))[
            -1
        ]
        return most_recent_model_checkpoint.as_posix()


class PytorchTuningPipeline(BaseTuningPipeline):
    def __init__(
        self,
        path_to_config: str,
        path_to_tuning_config: str,
        path_to_load_progress: str,
    ):
        pipeline_config = return_or_load(
            path_to_config, PipelineConfig, parse_pipeline_config
        )
        self.model_name = pipeline_config.config_name

        FeatureAssembly(
            None,
            pipeline_config.data_loader.user_id_col,
            pipeline_config.data_loader.item_id_col,
        )

        super(PytorchTuningPipeline, self).__init__(
            pipeline_config, path_to_tuning_config, path_to_load_progress
        )

    def _get_searcher_wrapper(self) -> PytorchSearcherWrapper:
        tuning_config = return_or_load(
            self.path_to_tuning_config, SearcherWrapper, parse_tuning_config
        )
        return PytorchSearcherWrapper(
            self.model_wrapper, tuning_config, self.path_to_load_progress
        )


def update_pipeline_config_by_infer_date(
    config: Union[str, Path, PipelineConfig],
    infer_date: int,
    is_finetune: bool = False,
    num_days_to_train: int = 90,
):
    """Update pipeline config by infer date. This is used for inference.

    Args:
        config (Union[str, Path, PipelineConfig]): pipeline config
        infer_date (int): infer date
        is_finetune (bool, optional): whether to finetune model. Defaults to False.
        num_days_to_train (int, optional): number of days to train. Defaults to 90.
            Only used when is_finetune is False.
    """
    config = return_or_load(config, PipelineConfig, parse_pipeline_config)
    assert isinstance(config, PipelineConfig)
    assert isinstance(config.model_wrapper.model_params, DLModelParams)
    train_filters_without_date = [
        fil
        for fil in config.data_loader.train_filters
        if const.FILENAME_DATE_COL not in fil
    ]
    val_filters_without_date = [
        fil
        for fil in config.data_loader.validation_filters
        if const.FILENAME_DATE_COL not in fil
    ]

    num_delay_days = 1  # num of days to delay for inference
    _infer_date = pd.to_datetime(infer_date, format=const.FILENAME_DATE_FORMAT)
    val_date = int(
        (_infer_date - timedelta(days=num_delay_days)).strftime(
            const.FILENAME_DATE_FORMAT
        )
    )
    val_date_filters = [f"{const.FILENAME_DATE_COL} == {val_date}"]
    if is_finetune:
        train_date_filters = [f"{const.FILENAME_DATE_COL} == {val_date}"]
    else:
        train_start_date = int(
            (_infer_date - timedelta(days=num_delay_days + num_days_to_train)).strftime(
                const.FILENAME_DATE_FORMAT
            )
        )
        train_date_filters = [
            f"{const.FILENAME_DATE_COL} >= {train_start_date}",
            f"{const.FILENAME_DATE_COL} < {val_date}",
        ]

    config.data_loader.train_filters = train_date_filters + train_filters_without_date
    config.data_loader.validation_filters += val_date_filters + val_filters_without_date
    return config
