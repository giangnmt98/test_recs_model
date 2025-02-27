import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from recmodel.base.utils.logger import logger
from recmodel.base.utils.spark import SparkOperations


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


def __convert_pyarrowschema_to_pandasschema(p, is_pass_null=False):
    if p == pa.string():
        return np.dtype("O")
    elif p == pa.int32():
        return "int32" if not is_pass_null else "Int32"
    elif p == pa.int64():
        return "int64" if not is_pass_null else "Int64"
    elif p == pa.float32():
        return np.dtype("float32")
    elif p == pa.float64():
        return np.dtype("float64")
    else:
        return None


def load_parquet_data(
    file_paths: Union[Path, str, List[Path], List[str]],
    with_columns: Optional[List[str]] = None,
    process_lib: str = "pandas",
    filters: Optional[List[Any]] = None,
    spark: Optional[Any] = None,
    schema: Optional[Any] = None,
):
    """load parquet data."""

    if process_lib == "pandas":
        # Chuyển filters sang expression nếu có
        filters_expr = pq.filters_to_expression(filters) if filters else None

        # Chuẩn hóa file_paths thành list
        if isinstance(file_paths, (str, Path)):
            file_paths = [str(file_paths)]
        elif isinstance(file_paths, list):
            if len(file_paths) == 0:
                raise ValueError("file_paths list must not be empty")
            file_paths = [str(p) for p in file_paths]
        else:
            raise ValueError(
                "file_paths must be a string, Path, or list of strings/Paths"
            )

        # Tạo dataset cơ bản
        if len(file_paths) == 1 and os.path.isdir(file_paths[0]):
            dataset = ds.dataset(file_paths[0], format="parquet", partitioning="hive")
        else:
            for fp in file_paths:
                if not os.path.isfile(fp):
                    raise FileNotFoundError(
                        f"Expected a file but found a directory or missing file: {fp}"
                    )
            dataset = ds.dataset(file_paths, format="parquet", partitioning="hive")

        # Sử dụng Scanner để chọn cột và áp dụng bộ lọc ngay khi quét
        scanner = ds.Scanner.from_dataset(
            dataset,
            columns=with_columns,  # Chọn cột ngay từ đầu
            filter=filters_expr,  # Áp dụng bộ lọc ngay từ đầu
        )

        # Chuyển trực tiếp sang Pandas
        df = scanner.to_table().to_pandas()

        # Xử lý schema và ép kiểu dữ liệu
        if schema:
            df = df.astype(schema)
        else:
            # Dùng dataset thay vì scanner để lấy schema
            map_key_values = dict(zip(dataset.schema.names, dataset.schema.types))
            for col in df.columns:
                try:
                    np_type = __convert_pyarrowschema_to_pandasschema(
                        map_key_values[col]
                    )
                    if np_type:
                        df[col] = df[col].astype(np_type)
                except Exception:
                    np_type = __convert_pyarrowschema_to_pandasschema(
                        map_key_values[col], is_pass_null=True
                    )
                    if np_type:
                        df[col] = df[col].astype(np_type)

        return df
    elif process_lib == "cudf":
        import cudf

        pdf = load_parquet_data(
            file_paths=file_paths,
            with_columns=with_columns,
            process_lib="pandas",
            filters=filters,
            spark=spark,
            schema=schema,
        )
        return cudf.from_pandas(pdf)
    else:
        return load_parquet_data_by_pyspark(
            file_paths=file_paths,
            with_columns=with_columns,
            filters=filters,
            spark=spark,
            schema=schema,
        )


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
            logger.warning(f"we have yet to implement this {schema}")
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
