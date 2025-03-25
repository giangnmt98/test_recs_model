from abc import ABC, abstractmethod
from typing import Dict, List, Union

from recmodel.base.utils.fileops import load_parquet_data
from recmodel.base.utils.singleton import SingletonMeta
from recmodel.base.utils.spark import SparkOperations
from recmodel.base.utils.utils import convert_string_filters_to_pandas_filters


class BaseFeatureManager(ABC):
    def __init__(self, process_lib):
        self.process_lib = process_lib

    @abstractmethod
    def extract_dataframe(
        self, features_to_select: List[str], filters: Union[List[str], None] = None
    ):
        pass


class FromFileFeatureManager(BaseFeatureManager):
    def __init__(self, process_lib, dataset_path, config):
        super().__init__(process_lib)
        self.dataset_path = dataset_path
        self.spark_operation = SparkOperations()
        self._spark = None
        self.get_spark_session()
        self.config = config

    def get_spark_session(self):
        """Return spark session if exists else init one"""
        if self.process_lib == "pyspark" and not self._spark:
            self._spark = self.spark_operation.get_spark_session()
        return self._spark

    def extract_dataframe(
        self, features_to_select: List[str], filters: Union[List[str], None] = None
    ):
        query_to_filter = (
            convert_string_filters_to_pandas_filters(filters) if filters else None
        )
        return load_parquet_data(
            self.dataset_path,
            with_columns=features_to_select,
            filters=query_to_filter,
            process_lib=self.process_lib,
            config=self.config,
        )


class FeatureManagerCollection(metaclass=SingletonMeta):
    def __init__(
        self,
        offline_observation_fm: Dict[str, BaseFeatureManager],
    ):
        self.offline_observation_fm = offline_observation_fm
