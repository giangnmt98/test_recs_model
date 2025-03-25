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
    schema: Optional[Any] = None,
    should_optimize: bool = True,
    config=None,
    spark=None,
):
    """
    Load data from Parquet files using the specified processing library.

    Args:
        file_paths (Union[Path, str, List[Path], List[str]]): Path or list of paths
            to the Parquet files.
        with_columns (Optional[List[str]]): List of columns to load from the dataset.
            If None, all columns are loaded.
        process_lib (str): The library to use for processing ('pandas' or 'cudf').
        filters (Optional[List[Any]]): List of filters to apply while loading data.
        schema (Optional[Any]): Schema to apply on the loaded DataFrame.
        should_optimize (bool): If True, performs data optimization for cudf.
        config (dict): Configuration for optimization and partition settings.
        spark (Optional[Any]): Spark session object for loading via Spark.

    Returns:
        DataFrame: A DataFrame containing the loaded and optionally processed data.
    """
    # Validate file paths and ensure the processing library is supported
    file_paths = validate_file_paths(file_paths, process_lib)

    if process_lib == "pandas":
        filters = pq.filters_to_expression(filters) if filters else None

        # Read schema from the first Parquet file
        table = pq.read_table(
            file_paths[0] if isinstance(file_paths, list) else file_paths
        )
        col_types_mapping = dict(zip(table.schema.names, table.schema.types))

        # Create a dataset for processing multiple Parquet files
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

        # Convert dataset to pandas DataFrame
        data_frame = dataset.to_table(columns=with_columns, filter=filters).to_pandas()

        # Convert schema of the data frame based on schema and column types
        return convert_schema(data_frame, schema, col_types_mapping)

    elif process_lib == "cudf" and should_optimize:
        # Extract partition columns from the configuration
        partition_cols = set(config.get("partition_columns", []))

        # Process filters into partition and non-partition filters
        filtered_file_paths, non_partition_filters = process_partition_filters(
            filters, file_paths, partition_cols
        )

        # Calculate batch size based on VRAM availability
        batch_size = get_batch_size_vram(config["max_batch_size_data_loader"])

        # Get column mapping for label encoding
        mapping_df = get_mapping_from_config(config, "popularity_item_group")
        data_frames = []

        # Process data in batches to optimize memory usage
        for i in tqdm(
            range(0, len(filtered_file_paths), batch_size),
            desc="Processing batches",
            leave=True,
            dynamic_ncols=True,
        ):
            # Load current batch of data
            batch_df = cudf.read_parquet(
                filtered_file_paths[i : i + batch_size],
                columns=with_columns,
                filters=non_partition_filters,
            )

            # Optimize data types and apply label encoding
            batch_df = optimize_dataframe_types(
                apply_label_encoding_cudf(batch_df, mapping_df), config
            )

            # Round float columns to reduce unnecessary precision
            for col in batch_df.columns:
                if batch_df[col].dtype.kind == "f":  # Float column
                    batch_df[col] = batch_df[col].round(6)

            data_frames.append(batch_df)

        # Concatenate all processed batches into a single DataFrame
        data_frame = cudf.concat(data_frames, ignore_index=True)

        # Optimize specific column if it exists
        if "filename_date" in data_frame.columns:
            data_frame["filename_date"] = data_frame["filename_date"].astype("int32")

        # Clean up unused objects to free memory
        del batch_df, mapping_df
        gc.collect()

        return data_frame

    elif process_lib == "cudf":
        # Load data using pandas first if optimization is not required
        pdf = load_parquet_data(
            file_paths=file_paths,
            with_columns=with_columns,
            process_lib="pandas",
            filters=filters,
            schema=schema,
        )
        return cudf.from_pandas(pdf)

    else:
        # Use Spark for processing if specified
        return load_parquet_data_by_pyspark(
            file_paths=file_paths,
            with_columns=with_columns,
            filters=filters,
            spark=spark,
            schema=schema,
        )


def process_partition_filters(filters, file_paths, partition_cols):
    """
    Process filters into partition and non-partition filters.

    Args:
        filters (list): List of filters to apply on data.
        file_paths (list): List of paths to Parquet files.
        partition_cols (set): Set of partition column names.

    Returns:
        tuple: Filtered file paths and a list of non-partition filters.
    """
    # Default filters to an empty list if not provided
    filters = filters or []

    # Separate filters into partition and non-partition filters
    partition_filters = [f for f in filters if f[0] in partition_cols]
    filtered_file_paths = (
        filter_partitioned_files(str(file_paths), partition_filters, partition_cols)
        if partition_filters
        else file_paths
    )
    non_partition_filters = [f for f in filters if f[0] not in partition_cols]

    return filtered_file_paths, non_partition_filters


def get_batch_size_vram(max_batch):
    """
    Calculate the batch size based on available VRAM.

    Args:
        max_batch (int): Maximum batch size allowed.

    Returns:
        int: The calculated batch size.
    """
    # Estimate batch size based on free VRAM
    return estimate_batch_size(get_free_vram(), max_batch=max_batch)


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
