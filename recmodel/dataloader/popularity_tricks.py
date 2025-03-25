import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.utils.utils import anti_join


class ReSampleSolvePopularity:
    def __init__(
        self,
        gpu_loading,
        dataloader: BaseDataLoader,
    ):
        self.gpu_loading = gpu_loading
        self.label_col = dataloader.label_col
        self.weight_col = dataloader.weight_col
        self.tail_sample_groups = dataloader.tail_sample_groups
        self.popularity_item_group_col = dataloader.popularity_item_group_col
        self.factor_tail_sample_groups = dataloader.factor_tail_sample_groups
        self.popularity_sample_groups = dataloader.popularity_sample_groups
        self.factor_popularity_sample_groups = (
            dataloader.factor_popularity_sample_groups
        )
        self.user_id_col = dataloader.user_id_col
        self.item_id_col = dataloader.item_id_col
        self.user_feature_names = dataloader.user_feature_names
        self.time_order_event_col = dataloader.time_order_event_col

    def downnegative_sample(self, df, columns):
        """
        Performs down-sampling of negative samples based on specified criteria.
        Args:
            df (pandas.DataFrame or cudf.DataFrame): Input DataFrame.
            columns (List[str]): List of column names to select.
        Returns:
            pandas.DataFrame or cudf.DataFrame: The down-sampled DataFrame.
        """
        if isinstance(df, DataFrame):
            tail_fractions = {}
            for i in range(len(self.tail_sample_groups)):
                tail_fractions[str(self.tail_sample_groups[i])] = float(
                    self.factor_tail_sample_groups[i]
                )

            df_p = df.filter((F.col(self.label_col) != 1)).select(columns)
            df_n_p = df.filter(
                (F.col(self.label_col) == 1)
                & (
                    F.col(self.popularity_item_group_col).isin(
                        self.popularity_sample_groups
                    )
                )
            ).select(columns)
            df_n_np = df.filter(
                (F.col(self.label_col) == 1)
                & (F.col(self.popularity_item_group_col).isin(self.tail_sample_groups))
            ).select(columns)
            if len(tail_fractions) > sum(tail_fractions.values()):
                print(tail_fractions)
                df_n_np = df_n_np.withColumn(
                    "samplegroup",
                    F.concat_ws(
                        "@",
                        F.col(self.popularity_item_group_col),
                        F.col(self.time_order_event_col),
                    ),
                )
                pd_tail_fractions = df_n_np.select("samplegroup").distinct().toPandas()
                pd_tail_fractions[
                    self.popularity_item_group_col
                ] = pd_tail_fractions.samplegroup.str.split("@", expand=True)[0]
                pd_tail_fractions["factor"] = pd_tail_fractions[
                    self.popularity_item_group_col
                ].map(tail_fractions)
                tail_fractions = dict(
                    zip(
                        pd_tail_fractions["samplegroup"].tolist(),
                        pd_tail_fractions["factor"].tolist(),
                    )
                )
                df_n_np = df_n_np.sampleBy(
                    "samplegroup",
                    fractions=tail_fractions,
                ).drop("samplegroup")

            df = df_p.unionByName(df_n_p, allowMissingColumns=False).unionByName(
                df_n_np, allowMissingColumns=False
            )
        else:
            self.popularity_sample_groups = [
                df[self.popularity_item_group_col].dtype.type(x)
                for x in self.popularity_sample_groups
            ]

            self.tail_sample_groups = [
                df[self.popularity_item_group_col].dtype.type(x)
                for x in self.tail_sample_groups
            ]

            df_p = df[df[self.label_col] != 1][columns]
            df_n_p = df[
                (df[self.label_col] == 1)
                & (
                    df[self.popularity_item_group_col].isin(
                        self.popularity_sample_groups
                    )
                )
            ][columns]
            df_n_np = df[
                (df[self.label_col] == 1)
                & (df[self.popularity_item_group_col].isin(self.tail_sample_groups))
            ][columns]

            if len(self.factor_tail_sample_groups) > sum(
                self.factor_tail_sample_groups
            ):
                df_n_np_samples = []

                for i in range(len(self.tail_sample_groups)):
                    frac = self.factor_tail_sample_groups[i]
                    df_n_np_samples.append(
                        df_n_np[
                            (
                                df_n_np[self.popularity_item_group_col]
                                == self.tail_sample_groups[i]
                            )
                        ]
                        .groupby(self.time_order_event_col)
                        .sample(frac=frac)
                    )
                df_n_np = self.gpu_loading.get_pd_or_cudf().concat(
                    df_n_np_samples, axis=0, ignore_index=True, sort=False
                )
                del df_n_np_samples

            df = self.gpu_loading.get_pd_or_cudf().concat(
                [df_p, df_n_p, df_n_np], ignore_index=True, sort=False
            )
            del df_p
            del df_n_p
            del df_n_np
        return df

    def upnegative_sample(self, df, columns, df_pop_items, is_train, part_idx):
        """
        Performs up-sampling of negative samples based on specified criteria.
        Args:
            df (pandas.DataFrame or cudf.DataFrame): Input DataFrame.
            columns (List[str]): List of column names to select.
        Returns:
            pandas.DataFrame or cudf.DataFrame: The up-sampled DataFrame.
        """
        if isinstance(df, DataFrame):
            pop_fractions = {}
            for i in range(len(self.popularity_sample_groups)):
                pop_fractions[str(self.popularity_sample_groups[i])] = float(
                    self.factor_popularity_sample_groups[i]
                )
            # check is is to optimize speed
            if (sum(pop_fractions.values()) > 1e-4) and (
                not df_pop_items.rdd.isEmpty()
            ):
                df_active_users = (
                    df.filter((F.col(self.label_col) == 2))
                    .select(
                        self.user_id_col,
                        *self.user_feature_names,
                        self.time_order_event_col,
                    )
                    .drop_duplicates()
                )

                df_pop_neg = df_pop_items.join(
                    df_active_users, on=self.time_order_event_col, how="inner"
                )

                # remove 0, 2 user-item from df_pop_neg
                df_nn = (
                    df.filter(F.col(self.label_col) != 1)
                    .select(F.col(self.user_id_col), F.col(self.item_id_col))
                    .dropDuplicates()
                )

                df_pop_neg = df_pop_neg.join(
                    df_nn, on=[self.user_id_col, self.item_id_col], how="leftanti"
                )

                df_pop_neg = df_pop_neg.sampleBy(
                    self.popularity_item_group_col,
                    fractions=pop_fractions,
                )

                df_pop_neg = df_pop_neg.withColumn(self.label_col, F.lit(1))
                df_pop_neg = df_pop_neg.withColumn(self.weight_col, F.lit(1.0))

                df = df.select(columns).unionByName(
                    df_pop_neg.select(columns), allowMissingColumns=False
                )
        else:
            # Convert values in popularity sample groups to
            # the appropriate dtype of the column in the DataFrame

            self.popularity_sample_groups = [
                df[self.popularity_item_group_col].dtype.type(x)
                for x in self.popularity_sample_groups
            ]
            # Convert values in tail sample groups to
            # the appropriate dtype of the column in the DataFrame
            self.tail_sample_groups = [
                df[self.popularity_item_group_col].dtype.type(x)
                for x in self.tail_sample_groups
            ]

            # Check if there are valid sampling fractions
            # and if there are entries in df_pop_items
            if (sum(self.factor_popularity_sample_groups) > 1e-4) and len(
                df_pop_items
            ) > 0:
                # Filter active users (label == 2) and select relevant columns
                df_active_users = (
                    df[df[self.label_col] == 2]
                    .loc[
                        :,
                        [
                            self.user_id_col,
                            *self.user_feature_names,
                            self.time_order_event_col,
                        ],
                    ]
                    .drop_duplicates()
                )

                # Perform an inner join between df_pop_items
                # and active users based on the time order column
                df_pop_neg = self.gpu_loading.get_pd_or_cudf().merge(
                    df_pop_items,
                    df_active_users,
                    on=self.time_order_event_col,
                    how="inner",
                )

                # Exclude user-item pairs with labels other than 1 (negative samples)
                df_nn = df[df[self.label_col] != 1][
                    [self.user_id_col, self.item_id_col]
                ].drop_duplicates()
                # Perform an anti-join to remove such pairs from df_pop_neg
                df_pop_neg = anti_join(
                    df_pop_neg, df_nn, on_columns=[self.user_id_col, self.item_id_col]
                )
                del df_nn

                df_pop_neg_samples = []

                # Iterate over the popularity sample groups
                # and sample negative items based on the fractions
                for i in range(len(self.popularity_sample_groups)):
                    frac = self.factor_popularity_sample_groups[i]
                    df_pop_neg_samples.append(
                        df_pop_neg[
                            (
                                df_pop_neg[self.popularity_item_group_col]
                                == self.popularity_sample_groups[i]
                            )
                        ].sample(frac=frac)
                    )

                # Concatenate the sampled negative items into a single DataFrame
                df_pop_neg = self.gpu_loading.get_pd_or_cudf().concat(
                    df_pop_neg_samples, axis=0, ignore_index=True, sort=False
                )
                del df_pop_neg_samples

                # Set the label and weight columns for the negative samples
                df_pop_neg[self.label_col] = 1
                df_pop_neg[self.weight_col] = 1.0
                # Ensure that the label column dtype
                # matches that of the original DataFrame
                df_pop_neg[self.label_col] = df_pop_neg[self.label_col].astype(
                    df[self.label_col].dtype
                )

                # Select the required columns from the negative samples
                df_pop_neg = df_pop_neg[columns]

                # Concatenate the original DataFrame with the new negative samples
                df = self.gpu_loading.get_pd_or_cudf().concat(
                    [df[columns], df_pop_neg], ignore_index=True, sort=False
                )
                del df_pop_neg
        return df
