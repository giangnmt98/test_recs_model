from abc import ABC
from pathlib import Path
from typing import Union

from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.util import load_logs

from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.model_wrappers import BaseModelWrapper
from recmodel.base.schemas.pipeline_config import SearcherWrapper
from recmodel.base.utils import factory, logger


class BaseSearcherWrapper(ABC):
    """Base class for searcher wrapper.

    Attributions:
        model_wrapper:
            A model wrapper object, a subclass of base.model_wrappers.BaseModelWrapper.
        searcher_params:
            A init params of searcher.
        tuning_params:
            A dictionary of params with parameters names (str) as keys and lists of
                parameter settings to try as values.
    """

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        params: SearcherWrapper,
        path_to_load_progress: str,
    ):
        self.path_to_load_progress = path_to_load_progress
        self.model_wrapper = model_wrapper
        self.params = params
        self.searcher_params = params.searcher_params
        self.data_loader_params = params.data_loader_filter
        self.tuning_params = params.tuning_params

    def _build_searcher(self, tuning_function):
        raise NotImplementedError

    def fit(self, data_loader: BaseDataLoader, model_dir: Union[str, Path]) -> None:
        raise NotImplementedError


class PytorchSearcherWrapper(BaseSearcherWrapper):
    def _configure_tuning_params(self):
        def is_list_of_type(lst, type):
            if lst and isinstance(lst, list):
                return all(isinstance(elem, type) for elem in lst)
            else:
                return False

        configure_dict = {}
        for key, value in self.tuning_params.items():
            if is_list_of_type(value, str):
                for category in value:
                    configure_dict[key + "#cate#" + category] = [0, 1]
            elif is_list_of_type(value, list):
                for i, v in enumerate(value):
                    configure_dict[key + "#list#" + str(i)] = v
            else:
                configure_dict[key] = value
        return configure_dict

    def _build_searcher(self, data_loader):
        data_loader.params.train_filters = self.data_loader_params.train_filters
        data_loader.params.validation_filters = (
            self.data_loader_params.validation_filters
        )
        datamodule = data_loader.get_datamodel(
            features_order=self.model_wrapper.feature_order,
            train_full_data=False,
            turn_off_teardown=True,
        )

        def wrapper_tuning_function(**tuning_params):
            return self.model_wrapper.tuning_function(datamodule, **tuning_params)

        print(self.searcher_params)
        return factory.create(self.params.searcher_cls)(
            f=wrapper_tuning_function,
            pbounds=self._configure_tuning_params(),
            random_state=self.searcher_params["random_state"],
        )

    def fit(self, data_loader: BaseDataLoader, model_dir: Union[str, Path]) -> None:
        searcher = self._build_searcher(data_loader)
        if self.path_to_load_progress != "":
            load_logs(searcher, logs=[str(self.path_to_load_progress)])
        searcher_path = str(model_dir) + "/searcher.log"
        file_logger = JSONLogger(path=searcher_path)
        screen_logger = ScreenLogger(verbose=2)
        searcher.subscribe(Events.OPTIMIZATION_STEP, file_logger)
        searcher.subscribe(Events.OPTIMIZATION_STEP, screen_logger)
        searcher.subscribe(Events.OPTIMIZATION_START, screen_logger)
        searcher.subscribe(Events.OPTIMIZATION_END, screen_logger)
        searcher.maximize(
            init_points=self.searcher_params["init_points"],
            n_iter=self.searcher_params["n_iter"],
        )
        result = self.model_wrapper._group_tuning_params(searcher.max["params"])
        logger.logger.enable("")
        logger.logger.info(f"Best combination of parameters: {result}")
