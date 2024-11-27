from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from recmodel.base import experiment_manager
from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.experiment_manager import ExperimentManager, TuningExperimentManager
from recmodel.base.model_analysis import ModelAnalysis
from recmodel.base.model_wrappers import BaseModelWrapper
from recmodel.base.schemas.pipeline_config import PipelineConfig, SearcherWrapper
from recmodel.base.searcher_wrappers import BaseSearcherWrapper
from recmodel.base.utils import factory
from recmodel.base.utils.config import parse_pipeline_config, parse_tuning_config
from recmodel.base.utils.logger import logger
from recmodel.base.utils.utils import return_or_load


class _BasePipeline(ABC):
    """Base class for pipeline.

    Attributions:
        config:
            a pipeline config object.
        custom_run_dir:
            custom run dir that user can specify
        data_loader:
            a data loader object, a subclass of base.data_loaders.BaseDataLoader.
        exp_manager:
            a base.experiment_manager.ExperimentManager object.
        model_wrapper:
            a model wrapper object, a subclass of base.model_wrappers.BaseModelWrapper.
    """

    def __init__(
        self,
        config: Union[str, Path, PipelineConfig],
        custom_run_dir: Optional[str] = None,
        process_lib: str = "pandas",
        model_date: str = "",
    ):
        self.config = return_or_load(config, PipelineConfig, parse_pipeline_config)
        self.custom_run_dir = custom_run_dir
        logger.info("=" * 80)
        logger.info(f"Running pipeline with config name {self.config.config_name}")
        logger.info("=" * 80)
        self.process_lib = process_lib
        self.exp_manager = self._get_exp_manager()
        self.data_loader = self._get_data_loader()
        self.model_date = model_date
        self.model_wrapper = self._get_model_wrapper(self.model_date)

    @classmethod
    def init_from_model_path(cls, model_path: str):
        """Initialize the pipeline by model path."""
        return cls(experiment_manager.get_config_path_from_model_path(model_path))

    def load_model(self, model_path: str):
        """Loads a trained model from model_path."""
        self.model_wrapper = self._get_model_wrapper(self.model_date)
        self.model_wrapper.load_model(model_path)

    def load_most_recent_saved_model(self):
        """Loads the most recent saved model.

        The run directory is implied from the self.exp_manager.
        """
        saved_model_path = self.exp_manager.get_most_recent_saved_model_path()
        self.load_model(saved_model_path.as_posix())
        logger.info(f"Loaded model from {saved_model_path}")

    def get_most_recent_run_dir(self):
        self.exp_manager.get_most_recent_run_dir()

    def _get_model_wrapper(self, model_date="") -> BaseModelWrapper:
        return factory.create(self.config.model_wrapper.cls_name)(
            self.config.model_wrapper, model_date=model_date
        )

    def _get_data_loader(self) -> BaseDataLoader:
        return factory.create(self.config.data_loader.cls_name)(
            self.config.data_loader, process_lib=self.process_lib
        )

    def _get_exp_manager(self):
        return ExperimentManager(self.config, self.custom_run_dir)

    @abstractmethod
    def run(self):
        raise NotImplementedError


class BaseModelingPipeline(_BasePipeline):
    """Base class for modeling pipeline."""

    def run(self):
        logger.add(self.exp_manager.get_log_path())
        self.exp_manager.create_new_run_dir()
        self.train(train_full_data=False)
        self.analyze_model()
        if self.config.with_train_full_data:
            self.train(train_full_data=True)
        self.save_pipeline_bundle()

    def train(self, train_full_data: bool = False):
        if train_full_data:
            logger.info("Start training the model on both train and validation data.")
        else:
            logger.info("Start training the model on train data.")
        self.model_wrapper.fit(self.data_loader, train_full_data=train_full_data)

    def analyze_model(self) -> None:
        """Analyze the model on the validation dataset.

        The trained model is evaluated based on metrics for predictions slicing by
        each categorical feature specified by features_to_analyze.
        """
        logger.info("Model Analysis")
        if self.config.model_analysis:
            assert len(self.config.model_analysis.metrics) > 0, (
                "At least one metrics in model_analysis must be specified. "
                "Add the metrics in model_analysis in the pipeline config"
            )
            assert len(self.config.model_analysis.by_features) > 0, (
                "At least one by_features in model_analysis must be specified. "
                "Add the by_features in model_analysis in the pipeline config"
            )
            ModelAnalysis(
                self.data_loader,
                self.model_wrapper,
                self.config.model_analysis,
                output_dir=self.exp_manager.get_model_analysis_dir(),
            ).analyze()
        else:
            pass

    def save_pipeline_bundle(self):
        pass


class BasePytorchPipeline(BaseModelingPipeline):
    def _get_model_wrapper(self, model_date="") -> BaseModelWrapper:
        return factory.create(self.config.model_wrapper.cls_name)(
            self.config.model_wrapper,
            self.exp_manager.run_dir,
            model_date=model_date,
        )

    def run(self):
        logger.add(self.exp_manager.get_log_path())
        self.exp_manager.create_new_run_dir()
        self.train(train_full_data=False)

    def fine_tune(self, model_path: str):
        logger.add(self.exp_manager.get_log_path())
        logger.info(f"Start fine-tuning the model from checkpoint: {model_path}.")
        self.exp_manager.create_new_run_dir()
        self.load_model(model_path)
        self.train(train_full_data=True)


class BaseTuningPipeline(_BasePipeline):
    """Base class for tuning pipeline."""

    def __init__(
        self,
        path_to_config: Union[str, Path, PipelineConfig],
        path_to_tuning_config: Union[str, Path, SearcherWrapper],
        path_to_load_progress: str,
    ):
        self.path_to_config = path_to_config
        self.path_to_tuning_config = path_to_tuning_config
        self.path_to_load_progress = path_to_load_progress
        super(BaseTuningPipeline, self).__init__(path_to_config)
        logger.info("=" * 80)
        logger.info(f"Start hyperparameter tuning with {path_to_tuning_config}")
        logger.info("=" * 80)
        self.searcher_wrapper = self._get_searcher_wrapper()

    def _get_exp_manager(self) -> TuningExperimentManager:
        return TuningExperimentManager(self.path_to_config, self.path_to_tuning_config)

    def _get_searcher_wrapper(self) -> BaseSearcherWrapper:
        tuning_config = return_or_load(
            self.path_to_tuning_config, SearcherWrapper, parse_tuning_config
        )
        return BaseSearcherWrapper(
            self.model_wrapper, tuning_config, self.path_to_load_progress
        )

    def _get_data_loader(self) -> BaseDataLoader:
        return factory.create(self.config.data_loader.cls_name)(self.config.data_loader)

    def tune_hyperparam(self):
        model_dir = self.exp_manager.run_dir
        self.searcher_wrapper.fit(self.data_loader, model_dir)

    def run(self):
        logger.add(self.exp_manager.get_log_path())
        self.exp_manager.create_new_run_dir()
        self.tune_hyperparam()
