import gc
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

import cudf
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from tqdm import tqdm

from recmodel.base.utils.spark import SparkOperations
from recmodel.base.utils.utils import (
    apply_label_encoding_cudf,
    convert_schema,
    estimate_batch_size,
    filter_partitioned_files,
    get_free_vram,
    get_mapping_from_config,
    load_config,
    optimize_dataframe_types,
    validate_file_paths,
)


def load_parquet_data_by_pyspark(
    file_paths: Union[Path, str, List[Path], List[str]],
    with_columns: Optional[List[str]] = None,
    filters: Optional[List[Any]] = None,
    spark: Optional[Any] = None,
    schema: Optional[Any] = None,
):
    assert spark is not None
    if isinstance(file_paths, list):
        spark_filenames = [
            item.as_posix() if isinstance(item, Path) else item for item in file_paths
        ]
        if schema:
            df = spark.read.schema(schema).parquet(*spark_filenames)
        else:
            df = spark.read.parquet(*spark_filenames)
        if with_columns is not None:
            df = df.select(with_columns)
        if filters is None:
            return df
        return filters_by_expression_in_pyspark(df, filters)
    else:
        file_paths = (
            file_paths.as_posix() if isinstance(file_paths, Path) else file_paths
        )
        if schema:
            df = (
                spark.read.option("mergeSchema", "true")
                .schema(schema)
                .parquet(file_paths)
            )
        else:
            df = spark.read.option("mergeSchema", "true").parquet(file_paths)
        if with_columns is not None:
            df = df.select(with_columns)

        if filters is None:
            return df

        return filters_by_expression_in_pyspark(df, filters)


def load_parquet_data(
    file_paths: Union[Path, str, List[Path], List[str]],
    with_columns: Optional[List[str]] = None,
    process_lib: str = "pandas",
    filters: Optional[List[Any]] = None,
    spark: Optional[Any] = None,
    schema: Optional[Any] = None,
):
    """
    Load dữ liệu từ file Parquet dựa trên thư viện xử lý.

    Args:
        file_paths (Union[Path, str, List[Path], List[str]]): Đường dẫn hoặc danh sách
            đường dẫn tới các file Parquet.
        with_columns (Optional[List[str]]): Danh sách các cột cần load (nếu có).
        process_lib (str): Thư viện xử lý dữ liệu, có thể là 'pandas' hoặc 'cudf'.
        filters (Optional[List[Any]]): Danh sách các bộ lọc áp dụng khi load.
        spark (Optional[Any]): Đối tượng Spark (nếu sử dụng Spark).
        schema (Optional[Any]): Schema để áp dụng khi load DataFrame.

    Returns:
        pandas.DataFrame hoặc cudf.DataFrame: DataFrame chứa dữ liệu đã được load.

    Raises:
        ValueError: Nếu không tìm thấy file phù hợp với các bộ lọc.
    """
    # Xác thực danh sách file và thư viện xử lý
    file_paths = validate_file_paths(file_paths, process_lib)

    if process_lib == "pandas":
        # Chuyển filters thành biểu thức Arrow (nếu có)
        filters = pq.filters_to_expression(filters) if filters else None

        # Đọc schema từ file Parquet đầu tiên
        table = pq.read_table(
            file_paths[0] if isinstance(file_paths, list) else file_paths
        )
        col_types_mapping = dict(zip(table.schema.names, table.schema.types))

        # Tạo dataset từ danh sách file hoặc file đơn lẻ
        dataset = ds.dataset(
            [
                ds.dataset(path, format="parquet", partitioning="hive")
                for path in file_paths
            ]
            if isinstance(file_paths, list)
            else file_paths,
            format="parquet",
            partitioning="hive",
        )

        # Load dữ liệu và trả kết quả sau khi chuyển schema
        data_frame = dataset.to_table(columns=with_columns, filter=filters).to_pandas()
        return convert_schema(data_frame, schema, col_types_mapping)

    elif process_lib == "cudf":
        # Load cấu hình từ file
        config = load_config("config.yaml")
        partition_cols = set(config.get("partition_columns", []))

        # Phân chia filters thành partition và non-partition
        filters = filters or []
        partition_filters = [f for f in filters if f[0] in partition_cols]
        filtered_file_paths = filter_partitioned_files(
            str(file_paths), partition_filters, partition_cols
        )

        if not filtered_file_paths:
            raise ValueError(
                f"No files match the filters: {partition_filters}. "
                f"Partition columns: {partition_cols}"
            )

        # Tính toán batch size dựa trên VRAM
        batch_size = estimate_batch_size(get_free_vram(), max_batch=1000)
        print(f"Estimated batch size: {batch_size}")

        # Lấy filters không thuộc partition
        non_partition_filters = [f for f in filters if f[0] not in partition_cols]

        # Lấy ánh xạ giá trị mã hóa
        mapping_df = get_mapping_from_config(config, "popularity_item_group")
        data_frames = []  # Danh sách DataFrame chứa dữ liệu đã xử lý

        # Load dữ liệu từng batch và áp dụng các xử lý
        for i in tqdm(
            range(0, len(filtered_file_paths), batch_size),
            desc="Processing batches",
            leave=True,
            dynamic_ncols=True,
        ):
            # Đọc file batch hiện tại
            batch_df = cudf.read_parquet(
                filtered_file_paths[i : i + batch_size],
                columns=with_columns,
                filters=non_partition_filters,
            )

            # Áp dụng tối ưu hóa và mã hóa nhãn
            batch_df = optimize_dataframe_types(
                apply_label_encoding_cudf(batch_df, mapping_df), config
            )

            # Làm tròn các cột dạng float để giảm độ chính xác không cần thiết
            for col in batch_df.columns:
                if batch_df[col].dtype.kind == "f":  # Float column
                    batch_df[col] = batch_df[col].round(6)

            data_frames.append(batch_df)  # Thêm batch hiện tại vào danh sách

        # Kết hợp tất cả các batches thành một DataFrame
        data_frame = cudf.concat(data_frames, ignore_index=True)
        gc.collect()  # Thu hồi bộ nhớ không sử dụng
        return data_frame


def filters_by_expression_in_pyspark(df, filters):
    """
    Check if filters are well-formed and convert to an ``Expression``.

    Parameters
    ----------
    filters : List[Tuple] or List[List[Tuple]]
    """

    for filter in filters:
        assert len(filter) == 3
        col = filter[0]
        op = filter[1]
        val = filter[2]
        if op == "in":
            if not isinstance(val, DataFrame):
                df = df.filter(F.col(col).isin(val))
            else:
                df = df.join(val, on=col, how="inner")
        elif op == "not in":
            if not isinstance(val, DataFrame):
                df = df.filter(~F.col(col).isin(val))
            else:
                df = df.join(val, on=col, how="leftanti")
        elif op in ["=", "=="]:
            df = df.filter(F.col(col) == val)
        elif op == "<":
            df = df.filter(F.col(col) < val)
        elif op == ">":
            df = df.filter(F.col(col) > val)
        elif op == "<=":
            df = df.filter(F.col(col) <= val)
        elif op == ">=":
            df = df.filter(F.col(col) >= val)
        elif op == "!=":
            df = df.filter(F.col(col) != val)
        else:
            raise ValueError(
                '"{0}" is not a valid operator in predicates.'.format((col, op, val))
            )
    return df


def save_parquet_data(
    df,
    save_path: Union[Path, str],
    partition_cols: Optional[List[str]] = None,
    process_lib: str = "pandas",
    overwrite: bool = True,
    existing_data_behavior: str = "delete_matching",
    schema: Optional[Any] = None,
):
    """save parquet data.

    Args:
        df: dataframe to save
        save_path: path to save
        partition_cols: list of partition columns
        process_lib: process library, only support pandas currently
        overwrite: overwrite if save_path exists
        existing_data_behavior: Controls how the dataset will handle data that already
            exists in the destination. More details in
            https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset
    """
    if process_lib == "pandas":
        if overwrite and Path(save_path).exists():
            shutil.rmtree(save_path)
        if schema:
            df = df[list(schema.keys())].astype(schema)
        pa_table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            pa_table,
            root_path=save_path,
            existing_data_behavior=existing_data_behavior,
            partition_cols=partition_cols,
        )
    elif process_lib == "cudf":
        if overwrite and Path(save_path).exists():
            shutil.rmtree(save_path)
        if schema:
            print(f"we have yet to implement this {schema}")
        df.to_parquet(save_path, partition_cols=partition_cols, index=None)

    else:
        mode = "overwrite"
        if not overwrite:
            mode = "append"
        if isinstance(save_path, Path):
            to_save_path = save_path.as_posix()
        else:
            to_save_path = save_path

        if schema:
            spark = SparkOperations().get_spark_session()
            df = spark.createDataFrame(df.select(schema.names).rdd, schema)

        if partition_cols is None:
            df.write.option("header", True).mode(mode).parquet(to_save_path)
        else:
            df.write.option("header", True).partitionBy(partition_cols).mode(
                mode
            ).parquet(to_save_path)
