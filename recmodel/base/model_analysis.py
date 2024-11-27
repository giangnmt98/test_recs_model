import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from pyspark.sql import DataFrame

from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.metrics import BaseMetric, DurationSum, get_instantiated_metric_dict
from recmodel.base.model_wrappers import BaseModelWrapper
from recmodel.base.schemas import pipeline_config
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.logger import logger
from recmodel.base.utils.mlflow import MLflowMaster


class ModelAnalysis:
    """A class performing model analysis on validation dataset on different dimensions.

    For each dimension in features_to_analyze (must be a categorical feature),
    group samples by each possible value then compute the metric scores for the groups.
    Results are then saved in to a csv file with header:
    feature_name, sample_count, metric_0, metric_1, ....
    Rows in each dataframe are sorted by metric_0 column from worst to best (since we
    often want to know how the model performance on the hard groups).
    Repeat for all dimensions.

    Attributes:
        data_loader:
            A data loader object.
        model_wrapper:
            A model wrapper object.
        features_to_analyze:
            A list of feature names to do the analysis.
        metrics:
            A list of metric objects to be computed.
        output_dir:
            A string of output directory.
        pred_col:
            A string of prediction column (default to "prediction").
        pred_rank_col:
            A string of rank prediction column (default to "prediction_rank").
        need_predict:
            A bool value to tell if method self.model_wrapper.predict is needed.
        need_predict_rank:
            A bool value to tell if method self.model_wrapper.predict_rank is needed.
        is_pandas:
            A boolean value indicating the library to process data is pandas or not
                (spark).
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        model_wrapper: Union[BaseModelWrapper, Any],
        params: pipeline_config.ModelAnalysis,
        output_dir: str,
        pred_col: str = "prediction",
        pred_rank_col: str = "prediction_rank",
        ckpt_type: str = "finetuned",
    ):
        self.data_loader = data_loader
        self.model_wrapper = model_wrapper
        self.features_to_analyze = params.by_features
        self.sep_col = params.sep_col
        self.metrics = _get_metrics(params.metrics)
        self.output_dir = output_dir
        self.label_col = self.data_loader.label_col
        self.pred_col = pred_col
        self.pred_rank_col = pred_rank_col
        self.ckpt_type = ckpt_type
        need_pred_rank_list = [metric.need_pred_rank for metric in self.metrics]
        self.need_predict_rank = any(need_pred_rank_list)
        self.need_predict = any(
            not need_pred_rank for need_pred_rank in need_pred_rank_list
        )
        self.is_pandas = True if self.data_loader.process_lib == "pandas" else False
        # A bool value to determine when to print the overall scores. This is set to
        # True at initialization, then to False right after printing the first overall
        # scores.
        self._show_overall_flag = True
        self.mlflow_master = MLflowMaster()

    def analyze(self):
        self._analyze_one_dataset("train", sep_col=self.sep_col)
        self._analyze_one_dataset("val", sep_col=self.sep_col)

    def _analyze_one_dataset(self, dataset_name: str, sep_col: str = ""):
        df_with_pred_all = self._get_df_with_pred(dataset_name)
        if sep_col:
            unique_sep_col_values = df_with_pred_all[sep_col].unique()
            for sep_col_value in unique_sep_col_values:
                df_with_pred = df_with_pred_all[
                    df_with_pred_all[sep_col] == sep_col_value
                ]
                self._show_overall_flag = True
                for feature_name in self.features_to_analyze:
                    self._analyze_on_feature(
                        df_with_pred, feature_name, dataset_name, str(sep_col_value)
                    )
        else:
            self._show_overall_flag = True
            df_with_pred = df_with_pred_all
            for feature_name in self.features_to_analyze:
                self._analyze_on_feature(df_with_pred, feature_name, dataset_name)

    def _get_df_with_pred(self, dataset_name: str) -> pd.DataFrame:
        """Get the feature dataframe and its prediction."""
        df = self._get_feature_df(dataset_name)
        trained_pairs = (
            self.data_loader.get_train_ds() if dataset_name == "val" else None
        )
        save_path = self._get_df_pred_save_path(dataset_name)
        if self.is_pandas:
            if self.need_predict:
                df[self.pred_col] = self.model_wrapper.predict(df)
            if self.need_predict_rank:
                df[self.pred_rank_col] = self.model_wrapper.predict_rank(
                    df, train_pairs=trained_pairs, data_loader=self.data_loader
                )
            df.to_csv(f"{save_path}.csv", index=False)
        else:
            # predict and save dataframe by pyspark then load dataframe again by pandas.
            if self.need_predict_rank:
                df = self.model_wrapper.predict_rank(
                    df, train_pairs=trained_pairs, data_loader=self.data_loader
                )
            else:
                df = self.model_wrapper.predict(df)
            df.write.mode("overwrite").parquet(save_path.as_posix())
            df = pd.read_parquet(save_path)
        return df

    def _get_feature_df(self, dataset_name: str):
        all_features = (
            list(self.features_to_analyze) + self.data_loader.all_feature_names
        )
        all_features = list(set(all_features))
        if dataset_name == "train":
            return self.data_loader.extract_dataframe(
                filters=self.data_loader.params.train_filters,
                features_to_select=all_features,
                is_train=True,
            )
        elif dataset_name == "val":
            return self.data_loader.extract_dataframe(
                filters=self.data_loader.params.validation_filters,
                features_to_select=all_features,
                is_train=False,
            )
        else:
            raise ValueError(f"dataset_name ({dataset_name}) not in ('train', 'val')")

    def _analyze_on_feature(
        self,
        df_with_pred: pd.DataFrame,
        feature_name: str,
        dataset_name: str,
        sep_col_value: str = "",
    ):
        """Analyzes the predictions based on one feature."""
        # get list of metric score dict for each value in feature_name
        list_of_group_dicts = [
            self._get_metric_dict(feature_name, feature_value, group)
            for feature_value, group in df_with_pred.groupby(feature_name)
        ]
        # add overall score
        overall_scores = self._get_metric_dict(feature_name, "OVERALL", df_with_pred)
        list_of_group_dicts.append(overall_scores)
        if self._show_overall_flag:
            self._show_overall_scores(overall_scores, dataset_name, sep_col_value)
            self._show_overall_flag = False

        df_group = pd.DataFrame(list_of_group_dicts).sort_values(
            by=self.metrics[0].score_names[0],
            ascending=self.metrics[0].is_higher_better,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = tmp_dir + f"/{feature_name}.csv"
            df_group.to_csv(path, index=False)
            artifact_path = (
                self.output_dir
                + f"/eval_info/{dataset_name}_{self.ckpt_type}/{self.metrics[0]}"
            )
            self.mlflow_master.mlflow.log_artifact(
                local_path=path, artifact_path=artifact_path
            )
        logger.info(
            f"Saved model analysis slicing against {feature_name} to {artifact_path}"
        )

    def _get_metric_dict(
        self, feature_name: str, feature_value: Any, df_with_pred: pd.DataFrame
    ) -> Dict[str, Union[str, float]]:
        res = {f"{feature_name}": feature_value, "sample_count": len(df_with_pred)}
        for metric in self.metrics:
            pred_col = self.pred_rank_col if metric.need_pred_rank else self.pred_col
            scores = metric.compute_scores(
                df_with_pred,
                label_col=self.label_col,
                pred_col=pred_col,
                user_id_col=self.data_loader.user_id_col,
            )
            res.update(scores)
        return res

    def _get_df_pred_save_path(self, dataset_name: str):
        parent_dir = Path(self.output_dir) / dataset_name
        if not parent_dir.exists():
            parent_dir.mkdir()
        return parent_dir / "prediction"

    def _get_df_feature_metric_csv_path(
        self, dataset_name: str, col: str, metric: str, sep_col_value: str = ""
    ) -> Path:
        base_dir = Path(self.output_dir) / f"{dataset_name}_{self.ckpt_type}"
        if sep_col_value:
            base_dir /= sep_col_value
        if dataset_name in ["train", "val"]:
            base_dir /= metric
        dirname = base_dir

        if not dirname.exists():
            dirname.mkdir(parents=True, exist_ok=True)

        return dirname / f"{col}.csv"

    def _show_overall_scores(
        self, overall_scores: Dict[str, Any], dataset_name: str, sep_col_value: str
    ) -> None:
        logger.info("=" * 20 + f" OVERALL SCORES on {dataset_name} dataset " + "=" * 20)
        if self.sep_col:
            logger.info(f"Analyzing base on {self.sep_col}=={sep_col_value}")
        logger.info("{:<20}: {}".format("Num samples", overall_scores["sample_count"]))
        save_df = pd.DataFrame()
        for key, val in overall_scores.items():
            if val == "OVERALL" or key == "sample_count":
                continue
            save_df[key] = [val]
            self.mlflow_master.mlflow.log_metric(
                dataset_name + "_" + self.ckpt_type + "_" + key, val
            )
            logger.info("{:<20}: {}".format(key, val))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = tmp_dir + "/overall_score.csv"
            save_df.to_csv(path, index=False)
            artifact_path = (
                self.output_dir
                + f"/eval_info/{dataset_name}_{self.ckpt_type}/overall_score"
            )
            self.mlflow_master.mlflow.log_artifact(
                local_path=path, artifact_path=artifact_path
            )


def _get_metrics(metric_names: List[str]) -> List[BaseMetric]:
    metric_by_name = get_instantiated_metric_dict()
    return [metric_by_name[metric_name] for metric_name in metric_names]


class DLModelAnalysis(ModelAnalysis):
    def __init__(
        self,
        df_with_pred_all: pd.DataFrame,
        pipeline: Any,
        data_loader: BaseDataLoader,
        params: pipeline_config.ModelAnalysis,
        output_dir: str,
        data_name: str,
        for_date: int,
        pred_col: str = "prediction",
        pred_rank_col: str = "prediction_rank",
        user_id_col: str = "user_id",
        item_id_col: str = "item_id",
        ckpt_type: str = "finetuned",
    ):
        super().__init__(
            data_loader=data_loader,
            model_wrapper=None,
            params=params,
            output_dir=output_dir,
            pred_col=pred_col,
            pred_rank_col=pred_rank_col,
            ckpt_type=ckpt_type,
        )
        self.for_date = for_date
        self.metric = "recall"
        self.pipeline = pipeline
        self.gpu_loading = GpuLoading()
        self.gpu_loading.set_gpu_use()
        self.data_name = data_name
        self.features_to_analyze = (
            params.by_features
            + params.by_external_item_features
            + params.by_external_user_features
        )
        self.df_with_pred_all = df_with_pred_all
        if len(params.by_external_user_features) > 0:
            user_features = self.data_loader.account_feature_manager.extract_dataframe(
                features_to_select=[user_id_col] + params.by_external_user_features
            )
            if isinstance(user_features, DataFrame):
                user_features = user_features.toPandas()
            if self.gpu_loading.is_gpu_available() and isinstance(
                user_features, pd.DataFrame
            ):
                user_features = self.gpu_loading.get_pd_or_cudf().from_pandas(
                    user_features
                )
            self.df_with_pred_all = self.df_with_pred_all.merge(
                user_features, on=user_id_col, how="left"
            )

        if len(params.by_external_item_features) > 0:
            item_features = self.data_loader.item_feature_manager.extract_dataframe(
                features_to_select=[item_id_col] + params.by_external_item_features,
                filters=[f"filename_date <= {self.for_date}"],
            )
            if isinstance(item_features, DataFrame):
                item_features = item_features.toPandas()
            if self.gpu_loading.is_gpu_available() and isinstance(
                item_features, pd.DataFrame
            ):
                item_features = self.gpu_loading.get_pd_or_cudf().from_pandas(
                    item_features
                )
            self.df_with_pred_all = self.df_with_pred_all.merge(
                item_features, on=item_id_col, how="left"
            )

    def analyze(self):
        self._analyze_one_dataset(self.data_name, sep_col=self.sep_col)
        self._show_duration_scores(
            df_to_eval=self.df_with_pred_all,
            for_date=self.for_date,
            positive_label=2,
            data_name=self.data_name,
        )

    def _show_duration_scores(
        self,
        df_to_eval: pd.DataFrame,
        for_date: int,
        positive_label: int,
        data_name: str,
    ) -> None:
        label_col = self.data_loader.label_col
        duration_scores = DurationSum().interact_scores(
            df=df_to_eval,
            label_col=label_col,
            focus_col="duration",
            positive_label=positive_label,
        )
        active_users = df_to_eval[df_to_eval[label_col] == positive_label][
            "user_id"
        ].nunique()
        duration_df = pd.DataFrame()
        if data_name in ["train", "val"]:
            logger.info(
                "=" * 20 + f" DURATION SCORES on {data_name} dataset " + "=" * 20
            )
            logger.info("{:<20}: {}".format("Active users: ", active_users))
            duration_df["Date"] = [for_date]
            duration_df["Active_users"] = [active_users]
            for key, value in duration_scores.items():
                logger.info("{:<20}: {}".format(key, value))
                duration_df[key] = [value]
                self.mlflow_master.mlflow.log_metric(
                    (data_name + "_" + self.ckpt_type + "_" + key).replace("@", "_"),
                    value,
                )
        duration_df["evaluate_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_df["model_name"] = self.pipeline.config.config_name
        duration_df["data_type"] = data_name
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = tmp_dir + "/overall_score.csv"
            duration_df.to_csv(path, index=False)
            artifact_path = (
                self.output_dir + f"/eval_info/{data_name}_{self.ckpt_type}/duration"
            )
            self.mlflow_master.mlflow.log_artifact(
                local_path=path, artifact_path=artifact_path
            )

        logger.info(f"Saved model analysis duration scores to {artifact_path}")

    def _analyze_on_feature(
        self,
        df_with_pred: pd.DataFrame,
        feature_name: str,
        dataset_name: str,
        sep_col_value: str = "",
    ):
        """Analyzes the predictions based on one feature."""
        # get list of metric score dict for each value in feature_name
        list_of_group_dicts = [
            self._get_metric_dict(feature_name, feature_value, group)
            for feature_value, group in df_with_pred.groupby(feature_name)
        ]
        # add overall score
        overall_scores = self._get_metric_dict(feature_name, "OVERALL", df_with_pred)
        list_of_group_dicts.append(overall_scores)
        if self._show_overall_flag:
            self._show_overall_scores(overall_scores, dataset_name, sep_col_value)
            self._show_overall_flag = False

        df_group = pd.DataFrame(list_of_group_dicts).sort_values(
            by=self.metrics[0].score_names[0],
            ascending=self.metrics[0].is_higher_better,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = tmp_dir + f"/{feature_name}.csv"
            df_group.to_csv(path, index=False)
            artifact_path = (
                self.output_dir
                + f"/eval_info/{dataset_name}_{self.ckpt_type}/{self.metric}"
            )
            self.mlflow_master.mlflow.log_artifact(
                local_path=path, artifact_path=artifact_path
            )
        logger.info(
            f"Saved model analysis slicing against {feature_name} to {artifact_path}"
        )

    def _show_overall_scores(
        self, overall_scores: Dict[str, Any], dataset_name: str, sep_col_value: str
    ) -> None:
        logger.info("=" * 20 + f" OVERALL SCORES on {dataset_name} dataset " + "=" * 20)
        if self.sep_col:
            logger.info(f"Analyzing base on {self.sep_col}=={sep_col_value}")
        logger.info("{:<20}: {}".format("Num samples", overall_scores["sample_count"]))
        save_df = pd.DataFrame()

        for key, val in overall_scores.items():
            if val == "OVERALL" or key == "sample_count":
                continue
            save_df[key] = [val]
            self.mlflow_master.mlflow.log_metric(
                (dataset_name + "_" + self.ckpt_type + "_" + key).replace("@", "_"), val
            )
            logger.info("{:<20}: {}".format(key, val))
        save_df["evaluate_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_df["model_name"] = self.pipeline.config.config_name
        save_df["data_type"] = dataset_name
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = tmp_dir + "/overall_score.csv"
            save_df.to_csv(path, index=False)
            artifact_path = (
                self.output_dir
                + f"/eval_info/{dataset_name}_{self.ckpt_type}/{self.metric}"
            )
            self.mlflow_master.mlflow.log_artifact(
                local_path=path, artifact_path=artifact_path
            )

    def _get_df_with_pred(self, dataset_name: str) -> pd.DataFrame:
        return self.df_with_pred_all
