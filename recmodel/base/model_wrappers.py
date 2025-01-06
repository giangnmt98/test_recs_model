from abc import ABC, abstractmethod
from typing import List

from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.schemas.pipeline_config import ModelWrapper


class BaseModelWrapper(ABC):
    def __init__(self, params: ModelWrapper):
        self.params = params
        self.model_params = params.model_params
        self.fit_params = params.fit_params
        self.model = self.build_model()
        self.feature_order: List[str] = []

    def fit(self, data_loader: BaseDataLoader, train_full_data: bool = False):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        """Predicts data inputs.

        data should be the remaining part of dataset without the label column.
        """
        pass

    @abstractmethod
    def predict_rank(self, pairs, data_loader, train_pairs=None):
        """Predict the rank of selected pairs. Performs best when evaluating only a
        handful of interactions. If you need to compute predictions for every user,
        use the predict() method instead.

        Args:
            pairs: The user-item pairs whose rank will be computed.
            data_loader: A BaseDataLoader object
            train_pairs: The trained user-item pairs which will be excluded from rank
                computation.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def get_item_embedding(self, data_loader):
        pass

    def get_user_embedding(self, data_loader):
        pass

    def get_feature_importance(self, input_data, num_samples: int = 100):
        """Computes feature importance for each feature based on an input data.

        Most of the models are supported by SHAP (https://github.com/slundberg/shap).
        For unsupported models, please override this method by a workable solution.
        """
        pass

    def _group_tuning_params(self, tuning_params):
        pass
