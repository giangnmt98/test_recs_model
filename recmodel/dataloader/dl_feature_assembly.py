import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from model_configs.constant import FILENAME_DATE_FORMAT, FILENAME_DATE_FORMAT_PYSPARK
from recmodel.base.feature_manager import FeatureManagerCollection
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.singleton import SingletonMeta
from recmodel.base.utils.spark import SparkOperations
from recmodel.base.utils.utils import (
    get_dayofweek_from_date,
    is_weekend_from_encoded_weekday,
)


class FeatureAssembly(metaclass=SingletonMeta):
    def __init__(
        self,
        user_id_col=None,
        item_id_col=None,
        user_context_feature_names=None,
        item_context_feature_names=None,
        time_order_event_col=None,
    ):
        self.item_id_col = item_id_col
        self.user_id_col = user_id_col
        self.user_context_feature_names = user_context_feature_names
        self.item_context_feature_names = item_context_feature_names
        self.time_order_event_col = time_order_event_col
        self.spark_operation = SparkOperations()
        self.gpu_loading = GpuLoading()

    def assemble(self, df, user_feature_df, item_feature_df, for_date=None):
        df = self.add_normal_feature(df, user_feature_df, item_feature_df)
        df = self.add_context_feature(df, for_date)
        return df

    def add_normal_feature(self, df, user_feature_df, item_feature_df):
        if isinstance(df, DataFrame):
            df = df.repartition(
                self.spark_operation.get_partitions(), F.col(self.user_id_col)
            )
            if not user_feature_df.rdd.isEmpty():
                df = df.join(
                    user_feature_df.repartition(
                        self.spark_operation.get_partitions(), F.col(self.user_id_col)
                    ),
                    on=self.user_id_col,
                    how="inner",
                )
            df = df.repartition(
                self.spark_operation.get_partitions(), F.col(self.item_id_col)
            )
            df = df.join(
                item_feature_df.repartition(
                    self.spark_operation.get_partitions(), F.col(self.item_id_col)
                ),
                on=self.item_id_col,
                how="inner",
            )
        else:
            if len(user_feature_df) > 0:
                df = df.merge(
                    user_feature_df,
                    on=self.user_id_col,
                    how="inner",
                )

            df = df.merge(
                item_feature_df,
                on=self.item_id_col,
                how="inner",
            )

        return df

    def _get_online_user_feature_df(self, for_date, process_lib):
        online_user_feature_df = (
            FeatureManagerCollection()
            .online_user_fm[process_lib]
            .extract_dataframe(
                features_to_select=[self.user_id_col] + self.user_context_feature_names,
                filters=[f"filename_date == {for_date}"],
            )
        )
        return online_user_feature_df

    def _add_context_feature_for_infering(self, df, for_date):
        if isinstance(df, pd.DataFrame):
            online_user_feature_df = self._get_online_user_feature_df(
                for_date, "pandas"
            )
        elif isinstance(df, DataFrame):
            online_user_feature_df = self._get_online_user_feature_df(
                for_date, "pyspark"
            )
        else:
            if "pyspark" in FeatureManagerCollection().online_user_fm:
                online_user_feature_df = self._get_online_user_feature_df(
                    for_date, "pyspark"
                )
                online_user_feature_df = (
                    self.gpu_loading.get_pd_or_cudf().DataFrame.from_pandas(
                        online_user_feature_df.toPandas()
                    )
                )
            else:
                online_user_feature_df = self._get_online_user_feature_df(
                    for_date, "pandas"
                )
                online_user_feature_df = (
                    self.gpu_loading.get_pd_or_cudf().DataFrame.from_pandas(
                        online_user_feature_df
                    )
                )

        if isinstance(df, DataFrame):
            df = df.repartition(
                self.spark_operation.get_partitions(), F.col(self.user_id_col)
            )
            df = df.join(
                online_user_feature_df.repartition(
                    self.spark_operation.get_partitions(), F.col(self.user_id_col)
                ),
                on=self.user_id_col,
                how="inner",
            )
        else:
            df = df.merge(
                online_user_feature_df,
                on=self.user_id_col,
                how="inner",
            )
        return df

    def add_context_feature(self, df, for_date=None):
        df = self.add_is_weekend_feature(df, for_date=for_date)
        # if for_date is not None:
        #     df = self._add_context_feature_for_infering(df, for_date=for_date)
        return df

    def add_is_weekend_feature(self, df, for_date=None):
        if isinstance(df, DataFrame):
            if for_date is not None:
                df = df.withColumn(
                    "encoded_weekday",
                    F.dayofweek(
                        F.to_date(
                            F.lit(for_date),
                            FILENAME_DATE_FORMAT_PYSPARK,
                        )
                    )
                    - 1,
                )
                df = df.withColumn(
                    "is_weekend",
                    F.when(
                        (F.col("encoded_weekday") == 0)
                        | (F.col("encoded_weekday") == 6),
                        F.lit(1),
                    ).otherwise(F.lit(0)),
                ).drop("encoded_weekday")
            else:
                df = df.withColumn(
                    "encoded_weekday",
                    F.dayofweek(
                        F.to_date(
                            F.col(self.time_order_event_col),
                            FILENAME_DATE_FORMAT_PYSPARK,
                        )
                    )
                    - 1,
                )
                df = df.withColumn(
                    "is_weekend",
                    F.when(
                        (F.col("encoded_weekday") == 0)
                        | (F.col("encoded_weekday") == 6),
                        F.lit(1),
                    ).otherwise(F.lit(0)),
                ).drop("encoded_weekday")
        else:
            if for_date is not None:
                df["is_weekend"] = is_weekend_from_encoded_weekday(
                    get_dayofweek_from_date(for_date)
                )
            else:
                df["encoded_weekday"] = (
                    self.gpu_loading.get_pd_or_cudf()
                    .to_datetime(
                        df[self.time_order_event_col], format=FILENAME_DATE_FORMAT
                    )
                    .dt.weekday
                    + 1
                )
                df["encoded_weekday"].replace(7, 0, inplace=True)

                df["is_weekend"] = 0
                df.loc[
                    (df["encoded_weekday"] == 0) | (df["encoded_weekday"] == 6),
                    "is_weekend",
                ] = 1
        return df
