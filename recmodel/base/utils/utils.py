import ast
import glob
import os
import typing
import warnings
from datetime import datetime, timedelta
from functools import wraps

import cudf
import numpy as np
import pandas as pd
import pyarrow as pa
import pynvml
import pyspark.sql.functions as F
import yaml
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


def get_free_vram():
    """
    Lấy dung lượng VRAM trống (MB) của GPU đầu tiên.

    Returns:
        float: Dung lượng VRAM trống (MB) nếu lấy được thông tin, None nếu có lỗi.
    """
    try:
        # Khởi tạo NVIDIA Management Library (NVML) để lấy thông tin GPU
        pynvml.nvmlInit()

        # Lấy GPU đầu tiên (thường là GPU mặc định)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Lấy thông tin bộ nhớ GPU (VRAM) bao gồm: bộ nhớ tổng,
        # đang sử dụng và còn trống
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Chuyển dung lượng bộ nhớ trống (bytes) sang MB
        free_vram_mb = info.free / 1024 / 1024

        pynvml.nvmlShutdown()  # Đóng NVML sau khi kết thúc
        return free_vram_mb
    except Exception as e:
        # Xử lý ngoại lệ và in cảnh báo nếu không lấy được thông tin
        print(f"Warning: Could not get GPU memory info: {e}")
        return None  # Nếu có lỗi, trả về None


def estimate_batch_size(vram_mb, file_mb=250, min_batch=1, max_batch=200):
    """
    Tính toán số lượng batch dựa trên VRAM trống (MB).

    Args:
        vram_mb (float): Dung lượng VRAM trống (MB).
        file_mb (int, optional): Dung lượng trung bình
        mỗi file Parquet trên GPU (mặc định 250MB).
        min_batch (int, optional): Số file nhỏ nhất trong mỗi batch (mặc định 1).
        max_batch (int, optional): Số file lớn nhất trong mỗi batch (mặc định 200).

    Returns:
        int: Batch size được ước tính.
    """
    if vram_mb is None:
        # Nếu không lấy được VRAM trống, sử dụng batch size tối thiểu
        return min_batch

    # Tính toán batch size dựa trên dung lượng VRAM và dung lượng file
    estimated_batch = max(min_batch, min(int(vram_mb / file_mb), max_batch))

    # In ra batch size được tự động chọn
    print(
        f"Auto-selected batch size: {estimated_batch}"
        f" (VRAM available: {vram_mb:.2f} MB)"
    )
    return estimated_batch


def get_mapping_from_config(config: dict, column: str) -> cudf.DataFrame:
    """
    Chuyển ánh xạ từ cấu hình (dict) thành DataFrame với các cột 'original'
     và 'encoded'.

    Args:
        config (dict): Cấu hình chứa ánh xạ cột.
        column (str): Tên cột cần ánh xạ.

    Returns:
        cudf.DataFrame: DataFrame chứa ánh xạ với cấu trúc:
            - 'original': giá trị ban đầu.
            - 'encoded': giá trị được ánh xạ.

    Raises:
        ValueError: Nếu không tìm thấy ánh xạ cho cột trong config.
    """
    # Ghép khóa ánh xạ từ tên cột (ví dụ: "column_mapping")
    mapping_key = f"{column}_mapping"

    # Lấy ánh xạ từ config
    mapping_dict = config.get(mapping_key, {})
    if not mapping_dict:
        # Nếu không tìm thấy ánh xạ, báo lỗi
        raise ValueError(f"No mapping found for '{mapping_key}' in config")

    # Chuyển ánh xạ dạng dict thành DataFrame với hai cột 'original' và 'encoded'
    return cudf.DataFrame(
        {"original": list(mapping_dict.keys()), "encoded": list(mapping_dict.values())}
    )


def extract_partition_values(relative_parts: list, partition_cols: set) -> dict:
    """
    Trích xuất giá trị partition từ đường dẫn với các partition cột cụ thể.

    Args:
        relative_parts (list): Danh sách các phần của đường dẫn (bỏ phần base_path).
                              Mỗi phần thường có dạng 'key=value'.
        partition_cols (set): Tập hợp các partition cột cần trích xuất.

    Returns:
        dict: Từ điển chứa các cặp 'key' và 'value' của partition tương ứng.
    """
    partition_values = {}  # Từ điển lưu trữ các giá trị partition
    for part in relative_parts:
        if "=" in part:
            # Chia nhỏ phần có dạng 'key=value' thành khóa và giá trị
            key, value = part.split("=")

            if key in partition_cols:
                # Kiểm tra xem key có nằm trong partition cột không
                try:
                    # Giả định nếu giá trị là số, chuyển thành int
                    partition_values[key] = int(value)
                except ValueError:
                    # Nếu không chuyển thành số được, lưu dạng string
                    partition_values[key] = value
    return partition_values


def is_partition_match(partition_values: dict, partition_filters: list) -> bool:
    """
    Kiểm tra giá trị partition có khớp với điều kiện lọc hay không.
    """
    for col, op, val in partition_filters:
        if col in partition_values:
            partition_value = partition_values[col]
            if (
                (op == "<=" and partition_value > val)
                or (op == ">=" and partition_value < val)
                or (op == "==" and partition_value != val)
                or (op == "<" and partition_value >= val)
                or (op == ">" and partition_value <= val)
            ):
                return False
    return True


def filter_partitioned_files(
    base_path: str, partition_filters: list, partition_cols: set
) -> list:
    """
    Lọc danh sách file dựa trên partition filters với cấu trúc cha - con.

    Args:
        base_path (str): Đường dẫn gốc chứa các file để lọc.
        partition_filters (list): Danh sách các điều kiện filter,
         mỗi điều kiện có dạng (cột, toán tử, giá trị).
        partition_cols (set): Tập hợp các cột partition cần quan tâm để kiểm tra.

    Returns:
        list: Danh sách đường dẫn của các file thỏa mãn các điều kiện filter.
    """
    filtered_files = []  # Danh sách các file đã lọc
    base_path_levels = len(
        base_path.rstrip(os.sep).split(os.sep)
    )  # Số cấp thư mục của base_path

    # Duyệt qua tất cả các file .parquet trong thư mục gốc (bao gồm cả thư mục con)
    for file_path in glob.glob(f"{base_path}/**/*.parquet", recursive=True):
        # Lấy các phần đường dẫn tương đối (bỏ phần base_path)
        relative_parts = file_path.split(os.sep)[base_path_levels:]

        # Trích xuất các giá trị partition từ đường dẫn
        partition_values = extract_partition_values(relative_parts, partition_cols)

        # Lọc các bộ lọc liên quan đến các cột hiện tại
        relevant_filters = [f for f in partition_filters if f[0] in partition_values]

        # Kiểm tra xem file có khớp các điều kiện filter hay không
        if is_partition_match(partition_values, relevant_filters):
            filtered_files.append(file_path)  # Nếu khớp, thêm file vào danh sách

    return filtered_files


def apply_label_encoding_cudf(
    df: cudf.DataFrame, mapping_df: cudf.DataFrame
) -> cudf.DataFrame:
    """
    Mã hóa giá trị cột 'popularity_item_group' trong DataFrame thông qua ánh xạ.

    Args:
        df (cudf.DataFrame): DataFrame đầu vào cần được mã hóa.
        mapping_df (cudf.DataFrame): DataFrame ánh xạ,
         chứa hai cột 'original' và 'encoded'.

    Returns:
        cudf.DataFrame: DataFrame với cột 'popularity_item_group'
        được mã hóa thay thế.
    """
    if "popularity_item_group" in df.columns:
        # Đổi tên cột 'original' của mapping_df
        # thành 'popularity_item_group' để đồng bộ với DataFrame chính
        mapping_df = mapping_df.rename(columns={"original": "popularity_item_group"})

        # Merge DataFrame với ánh xạ (mapping_df) để thêm cột 'encoded'
        df = df.merge(mapping_df, on="popularity_item_group", how="left")

        # Nếu có cột 'encoded', thay thế cột gốc bằng cột 'encoded'
        if "encoded" in df.columns:
            df = df.drop(columns=["popularity_item_group"])  # Xóa cột gốc
            df = df.rename(
                columns={"encoded": "popularity_item_group"}
            )  # Đổi tên cột mới

        # Đảm bảo cột 'popularity_item_group' là kiểu Int32 để tối ưu bộ nhớ
        df["popularity_item_group"] = df["popularity_item_group"].astype("Int32")

    return df


def optimize_dataframe_types(df: cudf.DataFrame, config: dict) -> cudf.DataFrame:
    """
    Chuyển đổi kiểu dữ liệu các cột trong cudf.DataFrame thành kiểu tối ưu hóa.

    Args:
        df (cudf.DataFrame): DataFrame đầu vào cần tối ưu kiểu dữ liệu cho các cột.
        config (dict): Cấu hình định nghĩa kiểu tối ưu cho
        mỗi cột thông qua khóa "optimize_type".

    Returns:
        cudf.DataFrame: DataFrame với kiểu dữ liệu các cột đã được tối ưu.

    Raises:
        Exception: Ghi lại lỗi nếu không thể chuyển đổi kiểu dữ liệu.
    """
    # Lấy cấu hình kiểu dữ liệu tối ưu (nếu có)
    optimize_type = config.get("optimize_type", {})

    # Duyệt qua từng cột và kiểu dữ liệu mục tiêu trong config
    for col, target_dtype in optimize_type.items():
        if col in df.columns:
            try:
                # Chuyển đổi kiểu dữ liệu cột sang kiểu được chỉ định
                df[col] = df[col].astype(target_dtype)
            except Exception as e:
                # Nếu có lỗi, in cảnh báo kèm tên cột và lỗi chi tiết
                print(f"Could not convert column '{col}' to '{target_dtype}': {e}")

    return df


def load_config(file_path: str) -> dict:
    """
    Tải file cấu hình YAML và trả về dưới dạng dictionary.

    Args:
        file_path (str): Đường dẫn tới file cấu hình YAML.

    Returns:
        dict: Một dictionary đại diện cho cấu hình được load.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file cấu hình.
        ValueError: Nếu file YAML không hợp lệ (lỗi parsing).
    """
    try:
        # Mở file cấu hình YAML và load nội dung thành dictionary
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        # Báo lỗi nếu file không tồn tại
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")
    except yaml.YAMLError as e:
        # Báo lỗi nếu cú pháp YAML sai
        raise ValueError(f"Error parsing YAML file '{file_path}': {e}")


def validate_file_paths(file_paths, process_lib):
    """
    Xác thực danh sách đường dẫn file và thư viện xử lý dữ liệu.

    Args:
        file_paths (list): Danh sách đường dẫn tới các file đầu vào.
        process_lib (str): Tên thư viện xử lý dữ liệu ('pandas' hoặc 'cudf').

    Returns:
        list: Danh sách các đường dẫn file đã được xác thực.

    Raises:
        ValueError: Nếu danh sách file_paths trống.
        ValueError: Nếu `process_lib` không được hỗ trợ.
    """
    # Kiểm tra nếu danh sách file_paths rỗng
    if isinstance(file_paths, list) and len(file_paths) == 0:
        raise ValueError("file_paths list cannot be empty.")
    # Kiểm tra nếu thư viện xử lý dữ liệu không được hỗ trợ
    if process_lib not in {"pandas", "cudf"}:
        raise ValueError(f"Unsupported process_lib: {process_lib}")
    return file_paths


def __convert_pyarrowschema_to_pandasschema(p, is_pass_null=False):
    """
    Chuyển đổi kiểu dữ liệu của pyarrow Schema thành pandas schema.

    Args:
        p (pyarrow.DataType): Kiểu dữ liệu của pyarrow.
        is_pass_null (bool, optional): Nếu True,
        bật hỗ trợ kiểu nullable (dành cho pandas 1.x).

    Returns:
        np.dtype | str | None: Kiểu dữ liệu pandas hoặc None nếu không khớp.
    """
    # Ánh xạ các kiểu dữ liệu pyarrow vào numpy/pandas
    if p == pa.string():
        return np.dtype("O")  # Dữ liệu dạng object (chuỗi)
    elif p == pa.int32():
        return "int32" if not is_pass_null else "Int32"  # Hỗ trợ nullable
    elif p == pa.int64():
        return "int64" if not is_pass_null else "Int64"  # Hỗ trợ nullable
    elif p == pa.float32():
        return np.dtype("float32")  # Số thực 32-bit
    elif p == pa.float64():
        return np.dtype("float64")  # Số thực 64-bit
    else:
        return None  # Trả về None nếu không khớp


def convert_schema(data_frame, schema, col_types):
    """
    Thay đổi kiểu dữ liệu của DataFrame dựa trên schema hoặc mapping cột (col_types).

    Args:
        data_frame (pandas.DataFrame | cudf.DataFrame): DataFrame
         cần chuyển đổi kiểu dữ liệu.
        schema (dict | None): Schema của các cột (kiểu dữ liệu mong muốn).
        col_types (dict): Từ điển mapping {tên cột: kiểu dữ liệu pyarrow}.

    Returns:
        pandas.DataFrame | cudf.DataFrame: DataFrame
         với kiểu dữ liệu đã được chuyển đổi.

    Raises:
        Exception: Bỏ qua lỗi chuyển đổi nếu gặp vấn đề khi áp dụng kiểu dữ liệu.
    """
    # Nếu schema được cung cấp, thay đổi toàn bộ kiểu dữ liệu theo schema
    if schema:
        return data_frame.astype(schema)

    # Nếu không có schema, chuyển đổi từng cột dựa trên col_types
    for col in data_frame.columns:
        try:
            # Gọi hàm chuyển đổi từ pyarrow schema sang pandas/numpy schema
            np_type = __convert_pyarrowschema_to_pandasschema(col_types[col])
            if np_type:
                data_frame[col] = data_frame[col].astype(np_type)
        except Exception as ex:
            # Báo lỗi nhưng tiếp tục chuyển đổi với kiểu nullable nếu cần
            print(ex)
            np_type = __convert_pyarrowschema_to_pandasschema(
                col_types[col], is_pass_null=True
            )
            if np_type:
                data_frame[col] = data_frame[col].astype(np_type)
    return data_frame
