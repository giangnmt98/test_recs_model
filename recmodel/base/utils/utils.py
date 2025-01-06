import ast
import typing
import warnings
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from model_configs.constant import FILENAME_DATE_COL, FILENAME_DATE_FORMAT
from recmodel.base.utils.logger import logger


def suppress_warnings(warning_type):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warning_type)
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_current_time_stamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def return_or_load(object_or_path, object_type, load_func):
    """Returns the input directly or load the object from file.
    Returns the input if its type is object_type, otherwise load the object using the
    load_func
    """
    if isinstance(object_or_path, object_type):
        return object_or_path
    return load_func(object_or_path)


def convert_pandas_query_to_pyarrow_filter(
    query: str,
) -> typing.Tuple[str, str, typing.Any]:
    """
    Convert pandas query to pyarrow filter

    Args:
        query: pandas query string. This is a boolean expression with 3 components:
            column name, operator, value. For example, "a > 1".

    Returns:
        A tuple with 3 components: column name, operator, value. For example,
        ("a", ">", 1).
    """
    ast_op_mapping = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Is: "==",
        ast.IsNot: "!=",
        ast.In: "in",
        ast.NotIn: "not in",
    }
    ast_node = ast.fix_missing_locations(ast.parse(query))
    assert isinstance(ast_node.body[0], ast.Expr)
    assert isinstance(
        ast_node.body[0].value, ast.Compare
    ), "Only support one condition currently"
    expression = ast_node.body[0].value
    assert isinstance(expression.left, ast.Name)
    column_name = expression.left.id
    op = expression.ops[0]
    op_str = ast_op_mapping[type(op)]
    value = ast.literal_eval(expression.comparators[0])

    return (column_name, op_str, value)


def convert_string_filters_to_pandas_filters(
    filters: typing.List[str],
) -> typing.List[typing.Tuple[str, str, typing.Any]]:
    """Converts a list of string filters to a list of tuple filters.

    Args:
        filters: A list of string filters. Each string filter is a boolean expression
            with 3 components: column name, operator, value. For example,
            ["a > 1", "b == 2"].

    Returns:
        A list of tuple filters. Each tuple filter is a boolean expression with 3
        components: column name, operator, value. For example, [("a", ">", 1), ("b",
        "==", 2)].
    """
    tuple_filters = []
    for f in filters:
        tuple_filter = convert_pandas_query_to_pyarrow_filter(f)
        tuple_filters.append(tuple_filter)
    return tuple_filters


def anti_join(df1, df2, on_columns):
    assert type(df1) == type(df2)
    if isinstance(df1, pd.DataFrame):
        dtypes = df1.dtypes
        outer = df1.merge(
            df2,
            on=on_columns,
            how="outer",
            indicator=True,
        )
        df = outer[(outer._merge == "left_only")].drop("_merge", axis=1)
        df = df.astype(dtypes)

    elif isinstance(df1, DataFrame):
        return df1.join(df2[on_columns], how="leftanti")
    else:
        df = df1.merge(
            df2,
            on=on_columns,
            how="leftanti",
        )
    return df


def drop_train_pairs(
    df,
    train_df,
    item_id_col: str,
    user_id_col: str,
    process_lib: str,
    print_log: bool = True,
) -> pd.DataFrame:
    """Drop train user/item pairs to avoid re-recommending known positives."""
    if train_df is None:
        raise NotImplementedError("Please call get_train_ds() first.")
    train_df = train_df[[user_id_col, item_id_col]].drop_duplicates()
    if process_lib in ["cudf", "pandas"]:
        num_pairs_before_drop = len(df)
        df = anti_join(df, train_df, on_columns=[user_id_col, item_id_col])
        # train_user_item_pairs = train_df[user_id_col].astype(str) + train_df[
        #     item_id_col
        # ].astype(str)
        # train_user_item_pairs = set(train_user_item_pairs)
        # df = df.copy()
        # df["pair"] = df[user_id_col].astype(str) + df[item_id_col].astype(str)
        # df = df[~df["pair"].isin(train_user_item_pairs)]
        # df = df.drop(columns=["pair"])

        num_pairs_drop = num_pairs_before_drop - len(df)
        if print_log:
            logger.info(
                f"Drop {num_pairs_drop} pairs in validation/test set, "
                "which appear in train pairs."
            )
        return df.reset_index(drop=True)
    elif process_lib == "pyspark":
        return df.join(train_df, on=[user_id_col, item_id_col], how="leftanti")
    else:
        raise NotImplementedError


def merge_profile_features_with_account_features(
    profile_feature_df, account_feature_df, id_col
):
    if isinstance(profile_feature_df, DataFrame) and isinstance(
        account_feature_df, DataFrame
    ):
        before_length = profile_feature_df.count()
        profile_feature_df = profile_feature_df.withColumn(
            "account", F.split(F.col(id_col), "#").getItem(1)
        )
        account_feature_df = account_feature_df.withColumn(
            "account", F.split(F.col(id_col), "#").getItem(1)
        ).drop(id_col)
        feature_df = profile_feature_df.join(
            account_feature_df, on="account", how="left"
        ).drop("account")
        after_length = feature_df.count()
        feature_df = feature_df.dropna()
    elif isinstance(profile_feature_df, pd.DataFrame) and isinstance(
        account_feature_df, pd.DataFrame
    ):
        before_length = profile_feature_df.shape[0]
        profile_feature_df["account"] = (
            profile_feature_df[id_col].str.split("#").str.get(1)
        )
        account_feature_df["account"] = (
            account_feature_df[id_col].str.split("#").str.get(1)
        )
        account_feature_df.drop(columns=[id_col], inplace=True)
        for col in account_feature_df.columns:
            if account_feature_df[col].dtype == "int32":
                account_feature_df[col] = account_feature_df[col].astype("Int32")
            elif account_feature_df[col].dtype == "int64":
                account_feature_df[col] = account_feature_df[col].astype("Int64")
        feature_df = pd.merge(
            profile_feature_df, account_feature_df, on="account", how="left"
        )
        feature_df.drop(columns=["account"], inplace=True)
        after_length = feature_df.shape[0]
        feature_df = feature_df.dropna().reset_index(drop=True)
        for col in feature_df.columns:
            if feature_df[col].dtype == "Int32":
                feature_df[col] = feature_df[col].astype("int32")
            elif feature_df[col].dtype == "Int64":
                feature_df[col] = feature_df[col].astype("int64")
    else:
        import cudf

        before_length = profile_feature_df.shape[0]

        profile_feature_df = fix_issue_cudf_split_and_get(
            profile_feature_df, id_col, "account", "#", 1
        )
        account_feature_df = fix_issue_cudf_split_and_get(
            account_feature_df, id_col, "account", "#", 1
        )
        account_feature_df.drop(columns=[id_col], inplace=True)
        feature_df = cudf.merge(
            profile_feature_df, account_feature_df, on="account", how="left"
        )
        feature_df.drop(columns=["account"], inplace=True)
        after_length = feature_df.shape[0]
        feature_df = feature_df.dropna().reset_index(drop=True)
    assert (
        before_length == after_length
    ), "Merging profile features with account features results in different length!"
    return feature_df


def fix_issue_cudf_split_and_get(
    df, col, name, delimiter, position, is_convert_number=False
):
    import cudf

    df = df.copy(deep=True)
    df["order"] = df.index
    pdf = df[["order", col]].to_pandas()
    id_series = pdf[col].str.split(delimiter).str.get(position)
    if is_convert_number:
        cdf = cudf.from_pandas(pdf)
        cdf[name] = cudf.to_numeric(id_series, errors="coerce")
    else:
        pdf[name] = id_series
        cdf = cudf.from_pandas(pdf)
    if name in df.columns:
        del df[name]
    df = df.merge(cdf[["order", name]], on="order", how="left").drop(["order"], axis=1)
    return df


def get_date_before(
    for_date: int,
    num_days_before: int,
) -> int:
    """Get date before for_date.

    Args:
        for_date: The date to get date before.
        num_days_before: The number of days before for_date.
    """
    date_before = pd.to_datetime(for_date, format=FILENAME_DATE_FORMAT) - timedelta(
        days=num_days_before
    )
    date_before = int(date_before.strftime(FILENAME_DATE_FORMAT))
    return date_before


def get_date_filters(
    for_date: int,
    num_days_to_load_data: typing.Optional[int],
    is_pyarrow_format: bool = False,
    including_infer_date: bool = False,
) -> typing.List[typing.Any]:
    """Get date filter to avoid reading all data.

    Args:
        for_date: The max date to load data.
        num_days_to_load_data: The maximum number of days on which history data can
            be load.
        is_pyarrow_format: The format of the filter
    """
    date_filters: typing.List[typing.Any] = []
    if including_infer_date:
        date_filters = (
            [f"{FILENAME_DATE_COL} <= {for_date}"]
            if not is_pyarrow_format
            else [(FILENAME_DATE_COL, "<=", for_date)]
        )
    else:
        date_filters = (
            [f"{FILENAME_DATE_COL} < {for_date}"]
            if not is_pyarrow_format
            else [(FILENAME_DATE_COL, "<", for_date)]
        )
    if num_days_to_load_data is not None:
        start_date = get_date_before(for_date, num_days_to_load_data)
        start_date_filter = (
            f"{FILENAME_DATE_COL} >= {start_date}"
            if not is_pyarrow_format
            else (FILENAME_DATE_COL, ">=", start_date)
        )
        date_filters.append(start_date_filter)
    return date_filters


def get_dayofweek_from_date(input_date: int):
    dayofweek = datetime.strptime(str(input_date), "%Y%m%d").weekday() + 1
    if dayofweek == 7:
        dayofweek = 0
    return dayofweek


def is_weekend_from_encoded_weekday(encoded_weekday: int):
    if encoded_weekday == 0 or encoded_weekday == 6:
        is_weekend = 1
    else:
        is_weekend = 0
    return is_weekend
