import gc
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyspark.sql.functions as F
import torch
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from tqdm import tqdm

from model_configs import constant as const
from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.schemas.pipeline_config import DataLoader
from recmodel.base.utils import logger
from recmodel.base.utils.fileops import save_parquet_data
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.utils import anti_join, drop_train_pairs
from recmodel.dataloader.dl_datamodule import DLDataModule
from recmodel.dataloader.dl_feature_assembly import FeatureAssembly
from recmodel.dataloader.popularity_tricks import ReSampleSolvePopularity


class PytorchDataLoader(BaseDataLoader):
    gpu_loading = GpuLoading()
    gpu_loading.set_gpu_use()
    # todo: support gpu if we can use dash_cudf to load big dataframe

    def __init__(self, params: DataLoader, process_lib="pandas"):
        if const.IS_USE_CUDF_AT_DATALOADER:
            if self.gpu_loading.is_gpu_available():
                self.process_lib = "cudf"
            else:
                self.process_lib = process_lib
        else:
            self.process_lib = process_lib
        super().__init__(params, process_lib=self.process_lib)
        self.batch_idx_col = "batch_idx"
        self._train_df_batch: Dict[int, Any] = {}
        self._procesed_train_path: Optional[Union[List, str]] = None
        self._procesed_val_path: Optional[Union[List, str]] = None
        self._train_cache: Dict[int, Any] = {}
        self._val_cache: Dict[int, Any] = {}
        self.number_mini_batch = 1
        self.pop_resample = ReSampleSolvePopularity(
            self.gpu_loading,
            self,
        )
        self.factor_tail_sample_groups = params.factor_tail_sample_groups

    # def extract_external_dataframe(
    #     self, features_to_select: List[str] = [], is_user: bool = False
    # ):
    #     if is_user:
    #         features_to_select = [self.user_id_col] + features_to_select
    #         df = self.user_feature_manager.extract_dataframe(
    #             features_to_select=features_to_select
    #         )
    #         if self.process_lib == "pyspark":
    #             df = self.spark_operation.toPandas(df)
    #         if self.gpu_loading.is_gpu_available():
    #             df = self.gpu_loading.get_pd_or_cudf().from_pandas(df)
    #
    #     else:
    #         features_to_select = [self.item_id_col] + features_to_select
    #         df = self.item_feature_manager.extract_dataframe(
    #             features_to_select=features_to_select,
    #             filters=self.params.content_filters,
    #         )
    #         if self.process_lib == "pyspark":
    #             df = self.spark_operation.toPandas(df)
    #         if self.gpu_loading.is_gpu_available():
    #             df = self.gpu_loading.get_pd_or_cudf().from_pandas(df)
    #     return df
    #
    # def include_external_features(
    #     self,
    #     df: pd.DataFrame,
    #     external_user_features: List[str] = [],
    #     external_item_features: List[str] = [],
    # ):
    #     if len(external_user_features) > 0:
    #         external_user_features_df = self.extract_external_dataframe(
    #             features_to_select=external_user_features, is_user=True
    #         )
    #         df = df.merge(external_user_features_df, how="left", on=self.user_id_col)
    #     if len(external_item_features) > 0:
    #         external_item_features_df = self.extract_external_dataframe(
    #             features_to_select=external_item_features
    #         )
    #         df = df.merge(external_item_features_df, how="left", on=self.item_id_col)
    #     return df

    def extract_main_dataframe(
        self,
        filters: List[str],
        is_train: bool,
        user_feature_cols: List,
        item_feature_cols: List,
        interacted_cols: Optional[List[str]] = None,
        features_order: Optional[List[str]] = None,
    ):
        assert features_order is not None
        assert interacted_cols is not None
        if self.process_lib == "pyspark":
            big_df = self.offline_observation_feature_manager.extract_dataframe(
                features_to_select=list(
                    set(user_feature_cols + item_feature_cols + interacted_cols)
                ),
                filters=filters,
            )

            if self.time_order_event_col == "":
                big_df = big_df.drop_duplicates()

            df_pop_items = self.get_df_pop_items(filters, is_train)

            df = FeatureAssembly().add_context_feature(big_df)

            if not is_train:
                df = drop_train_pairs(
                    df,
                    self._train_df,
                    self.item_id_col,
                    self.user_id_col,
                    process_lib=self.process_lib,
                )

            else:
                # reduce memory which is cached in GPU/CPU
                # if we don't select columns, It will cost a lot of memory and
                # cached in GPU until the program is terminated!
                self._train_df = df[
                    [self.item_id_col, self.user_id_col]
                ].drop_duplicates()

            df.persist(storageLevel=StorageLevel.MEMORY_ONLY)

            return self.get_ds(
                df,
                df_pop_items=df_pop_items,
                is_train=is_train,
                features_order=features_order,
                part_idx=-1,
            )
        else:
            if self.process_lib == "cudf":
                self.offline_observation_feature_manager.process_lib = "cudf"
            else:
                self.offline_observation_feature_manager.process_lib = "pandas"
            big_df = self.offline_observation_feature_manager.extract_dataframe(
                features_to_select=list(
                    set(user_feature_cols + item_feature_cols + interacted_cols)
                )
                + [self.batch_idx_col],
                filters=filters,
            )
            # revert for interacted_feature_manager to self.process_lib
            self.offline_observation_feature_manager.process_lib = self.process_lib

            df_pop_items = self.get_df_pop_items(filters, is_train)

            paths = []
            if self.batch_idx_col not in big_df.columns:
                big_df[self.batch_idx_col] = 0
                logger.logger.warning("self.number_mini_batch = 1")
            self.number_mini_batch = big_df.batch_idx.nunique()
            big_df.set_index(self.batch_idx_col, inplace=True)

            for batch_idx, df in tqdm(big_df.groupby(self.batch_idx_col)):
                if self.gpu_loading.is_gpu_available() and isinstance(df, pd.DataFrame):
                    df = self.gpu_loading.get_pd_or_cudf().from_pandas(df)

                if len(df) == 0:
                    continue
                if self.batch_idx_col in df.columns:
                    del df[self.batch_idx_col]

                if self.time_order_event_col == "":
                    df = df.drop_duplicates()

                df = FeatureAssembly().add_context_feature(big_df)
                if not is_train:
                    df = drop_train_pairs(
                        df,
                        self._train_df_batch[batch_idx],
                        self.item_id_col,
                        self.user_id_col,
                        process_lib=self.process_lib,
                        print_log=False,
                    )
                    # we must remove it when it's done!
                    del self._train_df_batch[batch_idx]
                else:
                    # reduce memory which is cached in GPU/CPU
                    # if we don't select columns, It will cost a lot of memory and
                    # cached in GPU until the program is terminated!
                    self._train_df_batch[batch_idx] = df[
                        [self.item_id_col, self.user_id_col]
                    ].drop_duplicates()
                paths.append(
                    self.get_ds(
                        df,
                        df_pop_items=df_pop_items,
                        is_train=is_train,
                        features_order=features_order,
                        part_idx=batch_idx,
                    )
                )
                del df
                gc.collect()
                torch.cuda.empty_cache()

            del big_df

            gc.collect()
            torch.cuda.empty_cache()
            return paths

    def get_df_pop_items(self, filters, is_train):
        if self.process_lib == "pyspark":
            emptyDF = self.spark_operation.get_spark_session().createDataFrame(
                [], StructType([])
            )
            if is_train:
                big_df = self.offline_observation_feature_manager.extract_dataframe(
                    features_to_select=[self.item_id_col, self.time_order_event_col]
                    + self.item_feature_names
                    + self.context_feature_names
                    + [self.batch_idx_col, self.popularity_item_group_col],
                    filters=filters + [f"{self.label_col} == 2"],
                ).drop_duplicates()

                df_pop_items = (
                    FeatureAssembly()
                    .add_context_feature(big_df)
                    .select(
                        self.item_id_col,
                        *self.item_feature_names,
                        *self.context_feature_names,
                        self.time_order_event_col,
                        self.popularity_item_group_col,
                    )
                    .drop_duplicates()
                )
                df_pop_items = df_pop_items.filter(
                    (
                        F.col(self.popularity_item_group_col).isin(
                            self.popularity_sample_groups
                        )
                    )
                )

            else:
                df_pop_items = emptyDF

        elif self.process_lib == "cudf":
            import cudf

            if is_train:
                big_df = (
                    self.offline_observation_feature_manager.extract_dataframe(
                        features_to_select=[self.item_id_col, self.time_order_event_col]
                        + self.item_feature_names
                        + self.context_feature_names
                        + [self.batch_idx_col, self.popularity_item_group_col],
                        filters=filters + [f"{self.label_col} == 2"],
                    )
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                df_pop_items = FeatureAssembly().add_context_feature(big_df)
                df_pop_items = (
                    df_pop_items.loc[
                        :,
                        [
                            self.item_id_col,
                            *self.item_feature_names,
                            *self.context_feature_names,
                            self.time_order_event_col,
                            self.popularity_item_group_col,
                        ],
                    ]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                # Ensure the values in `self.popularity_sample_groups`
                # match the data type of the `self.popularity_item_group_col` column
                self.popularity_sample_groups = [
                    df_pop_items[self.popularity_item_group_col].dtype.type(x)
                    for x in self.popularity_sample_groups
                ]
                df_pop_items = (
                    df_pop_items[
                        df_pop_items[self.popularity_item_group_col].isin(
                            self.popularity_sample_groups
                        )
                    ]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

            else:
                df_pop_items = cudf.DataFrame()
        else:
            if is_train:
                big_df = (
                    self.offline_observation_feature_manager.extract_dataframe(
                        features_to_select=[self.item_id_col, self.time_order_event_col]
                        + self.item_feature_names
                        + self.context_feature_names
                        + [self.batch_idx_col, self.popularity_item_group_col],
                        filters=filters + [f"{self.label_col} == 2"],
                    )
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                df_pop_items = FeatureAssembly().add_context_feature(big_df)
                df_pop_items = (
                    df_pop_items.loc[
                        :,
                        [
                            self.item_id_col,
                            *self.item_feature_names,
                            *self.context_feature_names,
                            self.time_order_event_col,
                            self.popularity_item_group_col,
                        ],
                    ]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                df_pop_items = (
                    df_pop_items[
                        df_pop_items[self.popularity_item_group_col].isin(
                            self.popularity_sample_groups
                        )
                    ]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
            else:
                df_pop_items = pd.DataFrame()

        return df_pop_items

    def extract_dataframe(
        self,
        filters: List[str],
        is_train: bool,
        features_to_select: Optional[List[str]] = None,
        time_order_event_col="",
        features_order: Optional[List[str]] = None,
    ):
        if features_to_select is None:
            features_to_select = self.all_feature_names

        profile_cols = list(
            set(self.profile_feature_names).intersection(set(features_to_select))
        )
        account_cols = list(
            set(self.account_feature_names).intersection(set(features_to_select))
        )
        item_cols = list(
            set(self.item_feature_names).intersection(set(features_to_select))
        )
        interacted_cols = (
            [self.user_id_col, self.item_id_col]
            + self.user_context_feature_names
            + self.item_context_feature_names
        )
        if self.label_col:
            interacted_cols.append(self.label_col)
        if self.duration_col:
            interacted_cols.append(self.duration_col)
        if time_order_event_col:
            interacted_cols.append(time_order_event_col)
        if self.weight_col:
            interacted_cols.append(self.weight_col)
        if self.popularity_item_group_col:
            interacted_cols.append(self.popularity_item_group_col)

        user_feature_cols = [self.user_id_col] + account_cols + profile_cols
        item_feature_cols = [self.item_id_col] + item_cols
        return self.extract_main_dataframe(
            filters=filters,
            is_train=is_train,
            user_feature_cols=user_feature_cols,
            item_feature_cols=item_feature_cols,
            interacted_cols=interacted_cols,
            features_order=features_order,
        )

    def get_ds(
        self,
        df: Union[pd.DataFrame, DataFrame],
        df_pop_items: Union[pd.DataFrame, DataFrame],
        is_train: bool,
        features_order: Optional[List[str]] = None,
        part_idx: int = -1,
    ):
        if isinstance(df, DataFrame):
            from pyspark.sql.window import Window

            if self.time_order_event_col not in df.columns:
                # just assign a date
                df = df.withColumn(self.time_order_event_col, F.lit(20230101))

            df_p = df.filter(F.col(self.label_col) == 2)
            df_pn = df.filter(F.col(self.label_col) == 0)
            df_n = df.filter(F.col(self.label_col) == 1)

            # if user-item is labeled as 2, user-item should be removed
            # at df_pn and df_n
            df_remove = df_p.select(
                F.col(self.item_id_col), F.col(self.user_id_col)
            ).dropDuplicates()
            df_pn = df_pn.join(
                df_remove, on=[self.user_id_col, self.item_id_col], how="leftanti"
            )
            df_n = df_n.join(
                df_remove, on=[self.user_id_col, self.item_id_col], how="leftanti"
            )

            # if user-item is labeled as 0, user-item should be removed at df_n
            df_remove = df_pn.select(
                F.col(self.item_id_col), F.col(self.user_id_col)
            ).dropDuplicates()
            df_n = df_n.join(
                df_remove, on=[self.item_id_col, self.user_id_col], how="leftanti"
            )

            df = df_p.unionByName(df_n, allowMissingColumns=False)

            assert isinstance(features_order, list)
            columns = (
                [self.user_id_col, self.item_id_col, self.popularity_item_group_col]
                + features_order
                + [self.label_col, self.weight_col, self.time_order_event_col]
            )

            df = self.pop_resample.downnegative_sample(df, columns)

            df = self.pop_resample.upnegative_sample(
                df, columns, df_pop_items, is_train, part_idx=part_idx
            )

            if self.debug:
                df_check = df.groupby(
                    F.col(self.popularity_item_group_col), F.col(self.label_col)
                ).agg(
                    F.count(F.col(self.label_col)).alias("n_sample"),
                    F.countDistinct(F.col(self.item_id_col)).alias("n_item"),
                )
                df_check = df_check.withColumn(
                    "rate", F.col("n_sample") / F.col("n_item")
                )

                df_check.orderBy(
                    self.label_col, F.col("n_sample"), ascending=False
                ).show()

            # shuffle all df
            df = df.select("*").orderBy(F.rand())
            df = df.select(columns)

            df = df.withColumn(
                "occurence",
                F.row_number().over(
                    Window.partitionBy(F.col(self.time_order_event_col)).orderBy(
                        F.lit("A")
                    )
                ),
            )
            df = df.withColumn(
                "batch_idx", F.floor(F.col("occurence") / self.batch_size)
            ).drop("occurence")
            interacted_parquet_path = self.spark_operation.get_checkpoint_dir()
            self.save_temp_dataset(
                df.select(
                    [self.time_order_event_col, "batch_idx"]
                    + features_order
                    + [self.weight_col, self.label_col]
                ),
                features_order=features_order,
                interacted_parquet_path=interacted_parquet_path,
                is_train=is_train,
                part_idx=part_idx,
            )

            df.unpersist()

            return interacted_parquet_path

        else:
            assert isinstance(df, self.gpu_loading.get_pd_or_cudf().DataFrame)
            if self.time_order_event_col not in df.columns:
                # just assign a date
                df[self.time_order_event_col] = 20230101

            assert isinstance(features_order, list)
            df_p = df[df[self.label_col] == 2]
            df_pn = df[df[self.label_col] == 0]
            df_n = df[df[self.label_col] == 1]

            df_remove = df_p[[self.item_id_col, self.user_id_col]].drop_duplicates()
            df_pn = anti_join(
                df_pn, df_remove, on_columns=[self.user_id_col, self.item_id_col]
            )

            df_n = anti_join(
                df_n, df_remove, on_columns=[self.user_id_col, self.item_id_col]
            )

            df_remove = df_pn[[self.item_id_col, self.user_id_col]].drop_duplicates()

            df_n = anti_join(
                df_n, df_remove, on_columns=[self.user_id_col, self.item_id_col]
            )

            df = self.gpu_loading.get_pd_or_cudf().concat(
                [df_p, df_n], ignore_index=True
            )

            columns = (
                [self.user_id_col, self.item_id_col, self.popularity_item_group_col]
                + features_order
                + [self.label_col, self.weight_col, self.time_order_event_col]
            )

            # If any group in factor_tail_sample_groups is not equal to 1,
            # perform negative sampling
            if any(group != 1 for group in self.factor_tail_sample_groups):
                # Perform down-sampling of negative samples
                df = self.pop_resample.downnegative_sample(df, columns)

                # Perform up-sampling of negative samples with provided parameters
                df = self.pop_resample.upnegative_sample(
                    df, columns, df_pop_items, is_train, part_idx=part_idx
                )

            # shuffle all df
            df = df[columns]
            df = df.reset_index(drop=True)
            df = df.iloc[
                self.gpu_loading.get_np_or_cupy().random.permutation(df.index)
            ].reset_index(drop=True)

            df["batch_idx"] = df.groupby(self.time_order_event_col).cumcount()
            df["batch_idx"] = df["batch_idx"] // (
                self.batch_size // self.number_mini_batch
            )

            interacted_parquet_path = self.spark_operation.get_checkpoint_dir()
            self.save_temp_dataset(
                df[
                    [self.time_order_event_col, "batch_idx"]
                    + features_order
                    + [self.weight_col, self.label_col]
                ],
                features_order=features_order,
                interacted_parquet_path=interacted_parquet_path,
                is_train=is_train,
                part_idx=part_idx,
            )
            return interacted_parquet_path

    def save_temp_dataset(
        self,
        df: Union[pd.DataFrame, DataFrame],
        features_order: List,
        interacted_parquet_path: str = "",
        is_train: bool = True,
        part_idx: int = -1,
    ):
        """
        Saves a temporary dataset to a Parquet file.
        Args:
            df (pandas.DataFrame or cudf.DataFrame): Input DataFrame.
            interacted_parquet_path (str): Path to save the Parquet file.
        Returns:
            None
        """

        if isinstance(df, DataFrame):
            for col in df.columns:
                df = df.filter(F.col(col).isNotNull())
            save_parquet_data(
                df.select(
                    [self.time_order_event_col, "batch_idx"]
                    + features_order
                    + [self.weight_col, self.label_col]
                ),
                save_path=interacted_parquet_path,
                process_lib="pyspark",
                partition_cols=[self.time_order_event_col, "batch_idx", self.label_col],
                overwrite=False,
            )

            df.unpersist()

        else:
            if not const.IS_USE_MEMORY_CACHE:
                for col in df.columns:
                    df = df[df[col].notna()]
                save_parquet_data(
                    df[
                        [self.time_order_event_col, "batch_idx"]
                        + features_order
                        + [self.weight_col, self.label_col]
                    ],
                    save_path=interacted_parquet_path,
                    process_lib=self.process_lib,
                    partition_cols=[
                        self.time_order_event_col,
                        "batch_idx",
                        self.label_col,
                    ],
                    overwrite=True,
                )
                del df
            else:
                cols = [self.time_order_event_col, "batch_idx"]
                df = df.set_index([self.time_order_event_col, "batch_idx"])[
                    features_order + [self.weight_col, self.label_col]
                ]
                if (not isinstance(df, pd.DataFrame)) and (
                    not const.IS_USE_GPU_BATCHDATASET
                ):
                    df = df.to_pandas()

                for col in df.columns:
                    if df[col].dtypes == pd.Int64Dtype():
                        df = df[df[col].notna()]
                        df[col] = df[col].astype("int64")
                    elif df[col].dtypes == pd.Int32Dtype():
                        df = df[df[col].notna()]
                        df[col] = df[col].astype("int32")

                if is_train:
                    self._train_cache[part_idx] = {
                        k: self.__convert_dataframe_to_tensor(v)
                        for k, v in df.groupby(cols)
                    }
                else:
                    self._val_cache[part_idx] = {
                        k: self.__convert_dataframe_to_tensor(v)
                        for k, v in df.groupby(cols)
                    }

    def __convert_dataframe_to_tensor(self, df):
        if self.gpu_loading.is_gpu_available() and const.IS_USE_GPU_BATCHDATASET:
            pred_tensor = torch.from_dlpack(df.to_dlpack())
        else:
            pred_tensor = torch.tensor(df.values, device="cpu")
        return pred_tensor

    def get_train_ds(
        self,
        include_val_ds: bool = False,
        features_order: Optional[List[str]] = None,
    ):
        """Gets train dataset."""
        _procesed_train_path = self.extract_dataframe(
            filters=self.params.train_filters,
            is_train=True,
            time_order_event_col=self.time_order_event_col,
            features_order=features_order,
        )
        logger.logger.info(
            f"self.params.train_filters = "
            f"{self.params.train_filters}: {_procesed_train_path}"
        )

        return _procesed_train_path

    def get_val_ds(self, features_order: Optional[List[str]] = None):
        """Gets val dataset."""
        if len(self.params.validation_filters) == 0:
            return None

        _procesed_val_path = self.extract_dataframe(
            filters=self.params.validation_filters,
            is_train=False,
            time_order_event_col=self.time_order_event_col,
            features_order=features_order,
        )
        logger.logger.info(
            "self.params.validation_filters = "
            f"{self.params.validation_filters}: {_procesed_val_path}"
        )
        return _procesed_val_path

    def get_datamodel(
        self,
        features_order: Optional[List[str]] = None,
        train_full_data: bool = False,
        turn_off_teardown: bool = False,
    ):
        """
        Gets the data model for training and validation.
        Args:
            features_order (List[str], optional):
                List of feature names in the desired order (default is None).
            train_full_data (bool, optional):
                Whether to use the full training data (default is False).
        Returns:
            DLDataModule: A data module for training and validation.
        """
        _procesed_train_path = self.get_train_ds(
            features_order=features_order,
        )

        if not train_full_data:
            if len(self.params.validation_filters) == 0:
                return None

            _procesed_val_path = self.get_val_ds(
                features_order=features_order,
            )

        else:
            _procesed_val_path = _procesed_train_path
            self._val_cache = self._train_cache
        # clean dataframe which is cached at self._train_df_batch
        keys = list(self._train_df_batch.keys())
        for k in keys:
            del self._train_df_batch[k]
        gc.collect()
        torch.cuda.empty_cache()

        if (not const.IS_USE_CUDF_AT_DATALOADER) or (not const.IS_USE_MEMORY_CACHE):
            if self.gpu_loading.is_gpu_available():
                pin_memory_device = f"cuda:{self.gpu_loading.get_gpu_device_id()}"
                pin_memory = True
                num_workers = int(str(os.cpu_count())) // 2
            else:
                pin_memory_device = "cpu"
                pin_memory = False
                num_workers = int(str(os.cpu_count())) // 2
        else:
            if not self.gpu_loading.is_gpu_available():
                pin_memory_device = "cpu"
                pin_memory = False
                num_workers = int(str(os.cpu_count())) // 2
            elif const.IS_USE_GPU_BATCHDATASET:
                pin_memory_device = f"cuda:{self.gpu_loading.get_gpu_device_id()}"
                pin_memory = False
                num_workers = 0
            else:
                pin_memory_device = f"cuda:{self.gpu_loading.get_gpu_device_id()}"
                pin_memory = True
                num_workers = int(str(os.cpu_count())) // 2

        logger.logger.info(
            f"pin_memory_device={pin_memory_device} "
            f"pin_memory={pin_memory} "
            f"num_workers={num_workers} "
        )

        return DLDataModule(
            label_col=self.label_col,
            weight_col=self.weight_col,
            procesed_train_path=_procesed_train_path,
            procesed_val_path=_procesed_val_path,
            features_order=features_order,
            shuffle=self.shuffle,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device,
            num_workers=num_workers,
            train_cache=self._train_cache,
            val_cache=self._val_cache,
            turn_off_teardown=turn_off_teardown,
        )
