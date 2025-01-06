from abc import ABC
from typing import List, Optional

import torch

from recmodel.base.feature_manager import FeatureManagerCollection
from recmodel.base.schemas.pipeline_config import DataLoader


class BaseDataLoader(ABC):
    """Base class to load the dataframe which is extracted from feature manager and
        prepare datasets for modeling phase.

    Attributes:
        params: The yaml dataloader param config
        user_id_col: The user id column name
        item_id_col: The item id column name
        label_col: The label column name
        duration_col: The duration column name
        user_feature_names: The list of user feature names
        item_feature_names: The list of item feature names
        all_feature_names: The list of all feature names
        user_feature_manager: The user feature manager object
        item_feature_manager: The item feature manager object
        interacted_feature_manager: The interacted feature manager object
        dataset: For lightfm.
    """

    def __init__(self, params: DataLoader, process_lib: str = "pandas"):
        self.process_lib = process_lib
        self.params = params
        self.time_order_event_col = self.params.time_order_event_col
        self.popularity_item_group_col = self.params.popularity_item_group_col
        self.factor_popularity_sample_groups = (
            self.params.factor_popularity_sample_groups
        )
        self.popularity_sample_groups = self.params.popularity_sample_groups
        self.factor_tail_sample_groups = self.params.factor_tail_sample_groups
        self.tail_sample_groups = self.params.tail_sample_groups
        self.user_id_col = self.params.user_id_col
        self.item_id_col = self.params.item_id_col
        self.label_col = self.params.label_col
        self.duration_col = self.params.duration_col
        self.weight_col = self.params.weight_col
        self.account_feature_names = self.params.account_feature_names
        self.profile_feature_names = self.params.user_feature_names
        self.user_feature_names = (
            self.account_feature_names + self.profile_feature_names
        )
        self.context_feature_names = self.params.context_feature_names
        self.user_context_feature_names = self.params.user_context_feature_names
        self.item_context_feature_names = self.params.item_context_feature_names
        self.item_feature_names = self.params.item_feature_names
        self.batch_size = self.params.batch_size
        self.shuffle = self.params.shuffle
        self.all_feature_names = self._get_all_features()
        self.dataset = None
        self.debug = self.params.debug
        self.force_extract: bool = False
        self._train_df = None
        self._val_df = None
        self._test_df = None
        self.offline_observation_feature_manager = (
            self._get_offline_observation_feature_manager()
        )
        self.spark_operation = self.offline_observation_feature_manager.spark_operation

    def _get_all_features(self) -> List[str]:
        all_features = (
            [self.user_id_col, self.item_id_col]
            + self.user_feature_names
            + self.item_feature_names
            + self.user_context_feature_names
            + self.item_context_feature_names
            + self.context_feature_names
        )
        if self.label_col:
            all_features.append(self.label_col)
        if self.weight_col:
            all_features.append(self.weight_col)
        if self.duration_col:
            all_features.append(self.duration_col)
        return all_features

    def _get_offline_observation_feature_manager(self):
        return FeatureManagerCollection().offline_observation_fm[self.process_lib]

    def extract_dataframe(
        self,
        filters: List[str],
        is_train: bool,
        features_to_select: Optional[List[str]] = None,
    ):
        pass

    def get_train_ds(
        self, include_val_ds: bool = False, features_order: Optional[List[str]] = None
    ):
        """Gets train dataset."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def get_val_ds(self, features_order: Optional[List[str]] = None):
        """Gets val dataset."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def get_datamodel(
        self, features_order: Optional[List[str]] = None, train_full_data: bool = False
    ):
        """Gets val dataset."""
        raise NotImplementedError("Must be implemented in subclasses.")


class PandasTensorDataGenerator:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source:
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
