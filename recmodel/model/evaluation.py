import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from model_configs import constant as const
from recmodel.base.model_analysis import DLModelAnalysis
from recmodel.base.schemas.pipeline_config import PipelineConfig
from recmodel.base.utils.config import parse_pipeline_config
from recmodel.base.utils.fileops import load_parquet_data
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.logger import logger
from recmodel.base.utils.mlflow import MLflowMaster
from recmodel.base.utils.utils import (
    drop_train_pairs,
    get_date_before,
    get_date_filters,
    return_or_load,
)
from recmodel.dataloader.dl import FeatureAssembly
from recmodel.model.model_pipelines import PytorchPipeline


class Evaluation:
    """Evaluate model on a given date"""

    def __init__(
        self,
        model_pipeline_config_path: str,
        model_root_dir: Union[str, Path],
        num_days_to_train: int,
        data_path: str,
        process_lib: str = "pandas",
        skip_feature_important: bool = False,
    ):
        self.process_lib = process_lib
        self.model_pipeline_config = return_or_load(
            model_pipeline_config_path, PipelineConfig, parse_pipeline_config
        )
        self.num_days_to_train = num_days_to_train
        self.data_path = data_path
        self.pipeline = self._get_pipeline()
        self.model_root_dir = Path(model_root_dir)
        self.offline_observation_fm = (
            self.pipeline.data_loader.offline_observation_feature_manager
        )
        self.user_id = self.pipeline.data_loader.user_id_col
        self.item_id = self.pipeline.data_loader.item_id_col
        self.predict_col = "prediction"
        self.gpu_loading = GpuLoading()
        self.gpu_loading.set_gpu_use()
        self.spark_operation = self.offline_observation_fm.spark_operation
        self.skip_feature_important = skip_feature_important
        self.mlflow_master = MLflowMaster()

    def _get_model_checkpoint_by_date(self, for_date: int, is_train_ds: bool = False):
        all_register_models = self.mlflow_master.mlflow.search_registered_models()
        selected_models = []
        for model in all_register_models:
            if (
                str(for_date) in model.name
                and str(self.model_pipeline_config.config_name) in model.name
            ):
                if is_train_ds:
                    if "finetuned" in model.name:
                        selected_models.append(f"models:/{model.name}/latest")
                else:
                    selected_models.append(f"models:/{model.name}/latest")
        return selected_models

    def _get_pipeline(self) -> PytorchPipeline:
        return PytorchPipeline(self.model_pipeline_config, process_lib=self.process_lib)

    def _read_eval_df(self, for_date: int):
        eval_data_path = Path(self.data_path).parent / "materialized_offline_data"
        user_eval_data_path = eval_data_path / "user_features"
        item_eval_data_path = eval_data_path / "item_features"
        user_path = glob(f"{str(user_eval_data_path)}/{for_date}")[0]
        item_path = glob(f"{str(item_eval_data_path)}/{for_date}")[0]

        if self.process_lib == "pyspark":
            spark = self.spark_operation.get_spark_session()
            user_feature_df = load_parquet_data(
                user_path, process_lib=self.process_lib, spark=spark
            )
            item_feature_df = load_parquet_data(
                item_path, process_lib=self.process_lib, spark=spark
            )
            user_feature_df = user_feature_df.withColumnRenamed("key0", self.user_id)
            item_feature_df = item_feature_df.withColumnRenamed("key0", self.item_id)
        else:
            user_feature_df = load_parquet_data(
                user_path, process_lib=self.process_lib, should_optimize=False
            )
            item_feature_df = load_parquet_data(
                item_path, process_lib=self.process_lib, should_optimize=False
            )
            user_feature_df = user_feature_df.rename(columns={"key0": self.user_id})
            item_feature_df = item_feature_df.rename(columns={"key0": self.item_id})
        return user_feature_df, item_feature_df

    def _get_df_to_eval_and_df_to_score(self, for_date: int) -> pd.DataFrame:
        """Get dataframe to evaluate on a given date"""
        label_col = self.pipeline.data_loader.label_col
        duration_col = self.pipeline.data_loader.duration_col
        interacted_cols_to_use = [
            "profile_id",
            "username",
            self.user_id,
            self.item_id,
            label_col,
            duration_col,
        ]
        user_feature_df, item_feature_df = self._read_eval_df(for_date=for_date)

        date_to_eval = get_date_before(for_date, 1)

        num_date_before = self.num_days_to_train
        interacted_date_filters = get_date_filters(date_to_eval, num_date_before) + [
            f"{label_col} == 2"
        ]
        logger.info(f"Loading users from date {date_to_eval}...")
        filters = [
            f"{const.FILENAME_DATE_COL} == {date_to_eval}",
            f"{label_col} == 2",
        ]
        if const.EVALUATE_FOR_INFERING_USER_ONLY:
            interacted_date_filters += ["is_infering_user == True"]
            filters += ["is_infering_user == True"]
        id_cols = [self.user_id, self.item_id]
        df = self.offline_observation_fm.extract_dataframe(
            features_to_select=interacted_cols_to_use
            + self.model_pipeline_config.model_analysis.by_features
            + self.model_pipeline_config.model_analysis.by_external_user_features
            + self.model_pipeline_config.model_analysis.by_external_item_features,
            filters=filters,
        ).drop_duplicates(subset=id_cols)

        # drop interacted pairs
        logger.info(f"Loading interacted pairs from date {interacted_date_filters}...")
        interacted_df = self.offline_observation_fm.extract_dataframe(
            features_to_select=id_cols,
            filters=interacted_date_filters,
        ).drop_duplicates()
        df_to_eval = self._drop_train_pairs_and_assemble(
            df, interacted_df, date_to_eval
        )

        if isinstance(df, DataFrame):
            df_users = df.select(self.user_id).distinct().withColumn("key", F.lit(1))
            df_items = (
                interacted_df.select(self.item_id)
                .distinct()
                .withColumn("key", F.lit(1))
            )

            df_to_score = df_users.join(df_items, on="key").drop("key")
        else:
            df_to_score = (
                df[[self.user_id]]
                .drop_duplicates()
                .assign(key=1)
                .merge(
                    interacted_df[[self.item_id]].drop_duplicates().assign(key=1),
                    on="key",
                )
                .drop("key", axis=1)
            )
        df_to_score = self._drop_train_pairs_and_assemble(
            df_to_score, interacted_df, date_to_eval, user_feature_df, item_feature_df
        )

        return df_to_score, df_to_eval

    def _drop_train_pairs_and_assemble(
        self,
        df,
        interacted_df,
        date_to_eval,
        user_feature_df=None,
        item_feature_df=None,
    ):
        if isinstance(df, DataFrame):
            df = drop_train_pairs(
                df,
                interacted_df,
                self.item_id,
                self.user_id,
                process_lib="pyspark",
            )
        else:
            df = drop_train_pairs(
                df,
                interacted_df,
                self.item_id,
                self.user_id,
                process_lib="pandas",
            )
        if user_feature_df is not None and item_feature_df is not None:
            df = FeatureAssembly().assemble(
                df, user_feature_df, item_feature_df, for_date=date_to_eval
            )
        return df

    def _get_content_df_to_eval(self, for_date: int) -> pd.DataFrame:
        cols_to_use = [self.item_id]
        label_col = self.pipeline.data_loader.label_col
        num_date_before = self.num_days_to_train
        date_to_eval = get_date_before(for_date, 1)
        date_filters = get_date_filters(date_to_eval, num_date_before)
        logger.info(f"Loading content from date {date_filters}...")
        filters = [f"{label_col} == 2"] + date_filters
        df = self.offline_observation_fm.extract_dataframe(
            features_to_select=cols_to_use,
            filters=filters,
        ).drop_duplicates()  # những item có trong 90 ngày lịch sử trc ngày eval
        if isinstance(df, DataFrame):
            df = df.withColumn("content_id", F.col(self.item_id))
        else:
            df["content_id"] = df[self.item_id].copy()
        return df

    def _get_mini_batch_size(self):
        """
        mini_batch_size formula:
        a = coefficient
        x = current available memory
        y = total memory
        z = len of item

        I have a relation: f(x) = int(a * (x/y)/z)
        """
        if self.gpu_loading.is_gpu_available():
            mini_batch_size_coeff = 10**8
            mini_batch_size = max(
                10**6,
                int(
                    mini_batch_size_coeff
                    * (
                        self.gpu_loading.get_available_memory()
                        / self.gpu_loading.get_total_memory()
                    )
                ),
            )
        else:
            mini_batch_size = 10**6

        return mini_batch_size

    def _predict_each_model_checkpoint(
        self,
        df_to_score: pd.DataFrame,
        for_date: int,
        model_checkpoint_path: str,
        is_train_ds: bool,
    ):
        """Predict for each model checkpoint.

        Args:
            df_to_score (pd.DataFrame): dataframe to score
            for_date (int): date to evaluate
            model_checkpoint_path (str): model checkpoint path
            num_top_k (int): number of top k to evaluate
            is_train_ds (bool): whether the dataset to evaluate is train dataset. If
                True, the date to evaluate will be the date before the given date to
                avoid dropping interacted items.
        """
        logger.info(f"Start evaluating for date {for_date}...")
        if is_train_ds:
            for_date = get_date_before(for_date, 1)
        self.pipeline.load_model(model_checkpoint_path)

        mini_batch_size = self._get_mini_batch_size()

        batch_data = []
        for i in range(0, len(df_to_score), mini_batch_size):
            mini_batch = df_to_score.iloc[i : i + mini_batch_size]

            mini_batch_tensor = mini_batch[self.pipeline.model_wrapper.feature_order]
            for col in mini_batch_tensor.columns:
                if self.process_lib == "cudf":
                    mini_batch_tensor = mini_batch_tensor[
                        mini_batch_tensor[col].notna()
                    ]
                elif self.process_lib == "pandas":
                    if mini_batch_tensor[col].dtypes == pd.Int64Dtype():
                        mini_batch_tensor = mini_batch_tensor[
                            mini_batch_tensor[col].notna()
                        ]
                        mini_batch_tensor[col] = mini_batch_tensor[col].astype("int64")
                    elif mini_batch_tensor[col].dtypes == pd.Int32Dtype():
                        mini_batch_tensor = mini_batch_tensor[
                            mini_batch_tensor[col].notna()
                        ]
                        mini_batch_tensor[col] = mini_batch_tensor[col].astype("int32")

            if self.gpu_loading.is_gpu_available():
                mini_batch_tensor = torch.from_dlpack(mini_batch_tensor.to_dlpack())
            else:
                mini_batch_tensor = torch.tensor(
                    mini_batch_tensor.to_numpy(),
                )
            batch_data.append([mini_batch_tensor])

        predicted_tensor = self.pipeline.model_wrapper.predict(batch_data)

        if self.gpu_loading.is_gpu_available():
            df_to_score[
                self.predict_col
            ] = self.gpu_loading.get_np_or_cupy().from_dlpack(
                torch.to_dlpack(predicted_tensor.reshape(-1, 1).squeeze())
            )
        else:
            df_to_score[self.predict_col] = (
                predicted_tensor.reshape(-1, 1).squeeze().cpu().numpy()
            )
        return df_to_score

    def _evaluate_each_model_checkpoint(
        self,
        df_to_eval: pd.DataFrame,
        df_to_score: pd.DataFrame,
        for_date: int,
        num_top_k: int,
        is_train_ds: bool = False,
    ):
        """Evaluate for each model checkpoint

        Args:
            df_to_eval (pd.DataFrame): dataframe to evaluate
            df_to_score (pd.DataFrame): dataframe to score
            for_date (int): date to evaluate
            num_top_k (int): number of top k to evaluate
            is_train_ds (bool, optional): whether the dataset to evaluate is train
                dataset. Defaults to False.
        """
        model_checkpoint_paths = self._get_model_checkpoint_by_date(
            for_date, is_train_ds
        )

        for model_checkpoint_path in model_checkpoint_paths:
            ckpt_type = (
                "finetuned"
                if "finetuned" in str(model_checkpoint_path)
                else "fulltrained"
            )
            logger.info(f"Evaluating {model_checkpoint_path}...")
            pred_df = self._predict_each_model_checkpoint(
                df_to_score,
                for_date,
                model_checkpoint_path,
                is_train_ds,
            )
            pred_df["prediction_rank"] = pred_df.groupby(self.user_id)[
                self.predict_col
            ].rank(method="first", ascending=False)
            pred_df = pred_df[[self.user_id, self.item_id, "prediction_rank"]]
            pred_df = pred_df[pred_df["prediction_rank"] <= num_top_k]
            df_to_evaluate = df_to_eval[
                df_to_eval[self.user_id].isin(pred_df[self.user_id])
            ]
            df_to_evaluate = df_to_evaluate.merge(
                pred_df, on=[self.user_id, self.item_id], how="left"
            )
            df_to_evaluate["prediction_rank"] = df_to_evaluate[
                "prediction_rank"
            ].fillna(9999)

            if is_train_ds:
                data_name = "train"
            else:
                data_name = "val"
            DLModelAnalysis(
                df_to_evaluate,
                self.pipeline,
                self.pipeline.data_loader,
                self.model_pipeline_config.model_analysis,
                output_dir=model_checkpoint_path.split("/")[1],
                data_name=data_name,
                item_id_col=self.item_id,
                user_id_col=self.user_id,
                for_date=for_date,
                ckpt_type=ckpt_type,
                pred_rank_col="prediction_rank",
            ).analyze()

    def _get_feature_importance(
        self,
        df_to_eval: pd.DataFrame,
        for_date: int,
    ) -> pd.DataFrame:
        assert hasattr(self.pipeline.model_wrapper, "trainer")
        assert hasattr(self.pipeline.model_wrapper, "feature_order")
        model_checkpoint_paths = self._get_model_checkpoint_by_date(for_date)

        for model_checkpoint_path in model_checkpoint_paths:
            ckpt_type = (
                "finetuned"
                if "finetuned" in str(model_checkpoint_path)
                else "fulltrained"
            )
            self.pipeline.load_model(model_checkpoint_path)
            self.pipeline.model_wrapper.trainer.enable_progress_bar = False
            logger.info(f"Getting feature importance of {model_checkpoint_path}...")
            if not self.gpu_loading.is_gpu_available():
                pred_array = (
                    df_to_eval[self.pipeline.model_wrapper.feature_order]
                    .to_numpy()
                    .astype(int)
                )
            else:
                pred_array = (
                    df_to_eval[self.pipeline.model_wrapper.feature_order]
                    .astype(int)
                    .to_cupy()
                )
            feature_importance = self.pipeline.model_wrapper.get_feature_importance(
                pred_array, num_samples=const.NUM_SAMPLE_FOR_FEATURE_IMPORTANCE
            )
            logger.info(f"Feature importance for {ckpt_type} model")
            for k, v in feature_importance.items():
                logger.info(f"{k}: {v}")
            file_csv = f"feature_importance_{ckpt_type}.csv"
            feature_names = "/".join(list(feature_importance.keys()))
            feature_importance_score = "/".join(
                [str(round(v, 4)) for k, v in feature_importance.items()]
            )
            feature_importance_df = pd.DataFrame()
            feature_importance_df["feature"] = [feature_names]
            feature_importance_df["feature_important"] = [feature_importance_score]
            feature_importance_df["model_name"] = self.pipeline.config.config_name
            feature_importance_df["evaluate_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = tmp_dir + "/" + file_csv
                feature_importance_df.to_csv(path, index=False)
                self.mlflow_master.mlflow.log_artifact(
                    local_path=path,
                    artifact_path=model_checkpoint_path.split("/")[1] + "/eval_info",
                )

    def _check_if_pyspark_df_and_move_to_gpu(self, df):
        if isinstance(df, DataFrame):
            df = self.spark_operation.toPandas(df)
        if self.gpu_loading.is_gpu_available() and isinstance(df, pd.DataFrame):
            df = self.gpu_loading.get_pd_or_cudf().from_pandas(df)
        return df

    def evaluate(self, for_date: int, num_top_k: int = 1000):
        content_rule_df = self._get_content_df_to_eval(for_date)
        df_to_score, df_to_eval = self._get_df_to_eval_and_df_to_score(for_date)

        content_rule_df = self._check_if_pyspark_df_and_move_to_gpu(content_rule_df)
        df_to_score = self._check_if_pyspark_df_and_move_to_gpu(df_to_score)
        df_to_eval = self._check_if_pyspark_df_and_move_to_gpu(df_to_eval)

        df_to_eval = df_to_eval[
            df_to_eval[self.item_id].isin(content_rule_df[self.item_id])
        ]

        # user_df = df_to_eval.drop(columns=[self.item_id]).drop_duplicates(
        #     subset=[self.user_id], keep="last"
        # )

        date_as_train_ds = for_date
        date_as_val_ds = get_date_before(for_date, num_days_before=1)
        # feature importance
        # if not self.skip_feature_important:
        #     self._get_feature_importance(df_to_eval=df_to_eval, for_date=for_date)
        # evaluate with the latest model checkpoint
        logger.info(f"Evaluating data of {for_date} as train data...")
        self._evaluate_each_model_checkpoint(
            df_to_eval=df_to_eval,
            df_to_score=df_to_score,
            for_date=date_as_train_ds,
            num_top_k=num_top_k,
            is_train_ds=True,
        )
        # evaluate with the second-latest model checkpoint
        logger.info(f"Evaluating data of {for_date} as validation data...")
        self._evaluate_each_model_checkpoint(
            df_to_eval=df_to_eval,
            df_to_score=df_to_score,
            for_date=date_as_val_ds,
            num_top_k=num_top_k,
            is_train_ds=False,
        )
