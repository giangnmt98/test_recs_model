from typing import Dict, List, Optional, Union

import numpy as np
from sklearn import metrics as sk_metrics

from recmodel.base.utils.gpu import GpuLoading


class BaseMetric:
    """Base class for metrics.

    Usually, one metric returns one score like in the case of accuracy, rmse, mse, etc.
    However, in some cases, metrics might contain several values as precision, recall,
    f1. In these cases, method compute_scores will return a list of scores.

    Attributes:
        name:
            A string of class attribute, unique for each metric.
        is_higher_better:
            A bool value to tell if the higher the metric, the better the performance.
            The metric here is the first score in the output of compute_scores function.
            Note that, for some metrics like RMSE, lower numbers are better.
        need_pred_rank:
            A bool value to decide predict or predict_rank in model will be used.
    """

    name: str = ""
    score_names = [""]
    is_higher_better: Union[bool, None] = None
    need_pred_rank: bool = False

    def compute_scores(
        self, df, label_col: str, pred_col: str, user_id_col: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute list of scores.

        Args:
            df: The dataframe contains prediction.
            label_col: The label column name.
            pred_col: The prediction column name.
            user_id_col: The user id column name. Some rank metrics need to compute for
                each user.
        """
        assert (
            self.is_higher_better is not None
        ), "Subclasses must define is_higher_better"
        scores = self._compute_scores(df, label_col, pred_col, user_id_col)
        assert len(self.score_names) == len(
            scores
        ), "self.score_names (len = {}) and scores (len = {})".format(
            len(self.score_names), len(scores)
        )
        return {
            score_name: score for (score_name, score) in zip(self.score_names, scores)
        }

    def _compute_scores(
        self, df, label_col: str, pred_col: str, user_id_col: Optional[str] = None
    ) -> List[float]:
        raise NotImplementedError

    def interact_scores(
        self, df, label_col: str, focus_col: str, positive_label: int
    ) -> Dict[str, int]:
        assert (
            self.is_higher_better is not None
        ), "Subclasses must define is_higher_better"
        scores = self._interact_scores(df, label_col, focus_col, positive_label)
        assert len(self.score_names) == len(
            scores
        ), "self.score_names (len = {}) and scores (len = {})".format(
            len(self.score_names), len(scores)
        )
        return {
            score_name: score for (score_name, score) in zip(self.score_names, scores)
        }

    def _interact_scores(
        self, df, label_col: str, duration_col: str, positive_label: int
    ) -> List[int]:
        raise NotImplementedError


class DurationSum(BaseMetric):
    name = "duration"
    score_names = [
        "duration@1",
        "duration@5",
        "duration@10",
        "duration@20",
        "duration@100",
        "duration@200",
        "duration@500",
        "duration@1000",
        "duration_all",
    ]
    is_higher_better = True
    ranks = [1, 5, 10, 20, 100, 200, 500, 1000, 9999]

    def _interact_scores(
        self, df, label_col: str, focus_col: str, positive_label: int
    ) -> List[int]:
        active_users = df[df[label_col] == positive_label]["user_id"].nunique()
        return [
            df[(df["prediction_rank"] <= rank) & (df[label_col] == positive_label)][
                focus_col
            ].sum()
            / active_users
            for rank in self.ranks
        ]


class RMSE(BaseMetric):
    """Root Mean Square Error."""

    name = "rmse"
    score_names = ["rmse"]
    is_higher_better = False

    def _compute_scores(
        self, df, label_col: str, pred_col: str, user_id_col: Optional[str] = None
    ) -> List[float]:
        return [sk_metrics.mean_squared_error(df[label_col], df[pred_col]) ** 0.5]


class RecallAtK(BaseMetric):
    """Recall at K"""

    name = "recall_at_k"
    score_names = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@100",
        "recall@200",
        "recall@500",
        "recall@1000",
    ]
    is_higher_better = True
    ranks = [1, 5, 10, 20, 100, 200, 500, 1000]
    need_pred_rank: bool = True

    def _compute_scores(
        self, df, label_col: str, pred_col: str, user_id_col: Optional[str] = None
    ) -> List[float]:
        # this implementation is not supported in cudf
        # agg_params = {"retrieved": (pred_col, "count")}
        # df = df.copy()
        # for rank in self.ranks:
        #     df[f"hit_{rank}"] = df[pred_col] <= rank
        #     agg_params[f"hit_{rank}"] = (f"hit_{rank}", "sum")
        # score_df = df.groupby(user_id_col).agg(**agg_params)
        agg_params = {pred_col: "count"}
        df = df.copy()
        for rank in self.ranks:
            df[f"hit_{rank}"] = df[pred_col] <= rank
            agg_params[f"hit_{rank}"] = "sum"
        score_df = df.groupby(user_id_col).agg(agg_params)
        score_df = score_df.rename(columns={pred_col: "retrieved"})
        return [
            (score_df[f"hit_{rank}"] / score_df["retrieved"]).mean()
            for rank in self.ranks
        ]


class NDCGAtK(BaseMetric):
    """NDCG at K"""

    name = "ndcg_at_k"
    score_names = [
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "ndcg@100",
        "ndcg@200",
        "ndcg@500",
        "ndcg@1000",
    ]
    is_higher_better = True
    ranks = [1, 5, 10, 20, 100, 200, 500, 1000]
    need_pred_rank: bool = True
    idcg_list = [
        sum([1 / np.log2(i + 1) for i in range(1, num_relevant_items + 1)])
        for num_relevant_items in range(1, 101)
    ]
    gpu_loading = GpuLoading()

    def _compute_scores(
        self, df, label_col: str, pred_col: str, user_id_col: Optional[str] = None
    ) -> List[float]:
        agg_params = {pred_col: "count"}
        df = df.copy()
        for rank in self.ranks:
            df[f"numerator_dcg_{rank}"] = df[pred_col] <= rank
            df[f"denominator_dcg_{rank}"] = np.log2(df[pred_col] + 1)
            df[f"dcg_{rank}"] = (
                df[f"numerator_dcg_{rank}"] / df[f"denominator_dcg_{rank}"]
            )
            agg_params[f"dcg_{rank}"] = "sum"
        score_df = df.groupby(user_id_col).agg(agg_params)
        score_df = score_df.rename(columns={pred_col: "num_relevant_items"})
        for rank in self.ranks:
            temp_df = self.gpu_loading.get_pd_or_cudf().DataFrame()
            temp_df["num_relevant_items"] = list(range(1, 101))
            temp_df[f"idcg_{rank}"] = self.idcg_list
            score_df = score_df.merge(temp_df, on="num_relevant_items", how="left")
            score_df.loc[score_df["num_relevant_items"] >= rank, f"idcg_{rank}"] = sum(
                [1 / np.log2(i + 1) for i in range(1, rank + 1)]
            )
        return [
            (score_df[f"dcg_{rank}"] / score_df[f"idcg_{rank}"]).mean()
            for rank in self.ranks
        ]


class PrecisionAtK(BaseMetric):
    """Precision at K"""

    name = "precision_at_k"
    score_names = [
        "precision@1",
        "precision@5",
        "precision@10",
        "precision@20",
        "precision@100",
        "precision@200",
        "precision@500",
        "precision@1000",
    ]
    is_higher_better = True
    ranks = [1, 5, 10, 20, 100, 200, 500, 1000]
    need_pred_rank: bool = True

    def _compute_scores(
        self, df, label_col: str, pred_col: str, user_id_col: Optional[str] = None
    ) -> List[float]:
        # this implementation is not supported in cudf
        # agg_params = {}
        # df = df.copy()
        # for rank in self.ranks:
        #     df[f"hit_{rank}"] = df[pred_col] <= rank
        #     agg_params[f"hit_{rank}"] = (f"hit_{rank}", "sum")
        # score_df = df.groupby(user_id_col).agg(**agg_params)

        agg_params = {}
        df = df.copy()
        for rank in self.ranks:
            df[f"hit_{rank}"] = df[pred_col] <= rank
            agg_params[f"hit_{rank}"] = "sum"
        score_df = df.groupby(user_id_col).agg(agg_params)

        return [(score_df[f"hit_{rank}"] / rank).mean() for rank in self.ranks]


def get_instantiated_metric_dict() -> Dict[str, BaseMetric]:
    res = {}
    for sub_class in BaseMetric.__subclasses__():
        metric = sub_class()
        res[metric.name] = metric
    return res
