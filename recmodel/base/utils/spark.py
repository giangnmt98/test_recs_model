import atexit
import os
import random
import shutil
import threading
from pathlib import Path

import psutil
import pyarrow.dataset as ds
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.logger import logger
from recmodel.base.utils.singleton import SingletonMeta


class AtomicCounter:
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value


class SparkOperations(metaclass=SingletonMeta):
    """
    Tập hợp funcs xử lý cho sparks
    """

    def __init__(self, config_spark=None):
        self.atom = AtomicCounter()
        if config_spark is None:
            gpu_loading = GpuLoading()
            gpu_loading.set_gpu_use()
            if gpu_loading.is_gpu_available():
                cpu_factor = 0.8
            else:
                cpu_factor = 0.5
            app_name = "spark-application"
            num_cores = max(int((os.cpu_count() - 1) * cpu_factor), 1)
            master = f"local[{num_cores}]"
            partitions = num_cores
            memory = psutil.virtual_memory()[1] / 1e9
            driver_memory = f"{int(0.7*  memory )}g"
            executor_memory = f"{int(0.7* memory)}g"
            auto_broadcast_join_threshold = 10485760
            checkpoint_dir = (
                f"/tmp/pyspark/"
                f"tmp_{self.atom.increment()}"
                f"_{random.randint(10000, 100000)}"
                f"_{random.randint(100, 5000000)}"
            )
            logger.info(
                f"app_name={app_name} master={master} "
                f"partitions={partitions} driver_memory={driver_memory} "
                f"num_cores={num_cores} driver_memory={driver_memory} "
                f"executor_memory={executor_memory} checkpoint_dir={checkpoint_dir}"
            )
        else:
            app_name = config_spark.name
            master = config_spark.master
            params = config_spark.params
            partitions = params.sql_shuffle_partitions
            driver_memory = params.driver_memory
            num_cores = params.num_cores
            executor_memory = params.executor_memory
            auto_broadcast_join_threshold = params.auto_broadcast_join_threshold
            checkpoint_dir = params.checkpoint_dir
        self.checkpoint_dir = checkpoint_dir
        self.spark_config = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", partitions)
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.memory", executor_memory)
            .config("spark.sql.execution.arrow.pyspark.enabled", True)
            .config("spark.sql.files.ignoreCorruptFiles", True)
            .config("spark.sql.files.ignoreMissingFiles", True)
            .config("spark.executor.cores", num_cores)
            .config("spark.driver.cores", num_cores)
            .config("spark.driver.maxResultSize", "100g")
            .config(
                "spark.sql.autoBroadcastJoinThreshold",
                auto_broadcast_join_threshold,
            )
            .config("spark.local.dir", self.checkpoint_dir)
        )
        self.partitions = partitions
        # https://spark.apache.org/docs/latest/sql-data-sources-generic-options.html
        self.__spark = self.__init_spark_session()
        atexit.register(self.clean_tmp_data)

    def get_spark_session(self):
        """Init spark session, set log level to warn"""
        if self.__spark is None:
            self.__spark = self.__init_spark_session()
        return self.__spark

    def toPandas(self, df):
        dir = (
            f"{self.checkpoint_dir}/"
            f"tmp_{self.atom.increment()}"
            f"_{random.randint(10000, 100000)}"
            f"_{random.randint(100, 5000000)}"
        )

        df.coalesce(self.partitions).write.option("header", True).mode(
            "overwrite"
        ).parquet(dir)

        df = (
            ds.dataset(dir, format="parquet", partitioning="hive")
            .to_table()
            .to_pandas()
        )
        shutil.rmtree(dir)
        return df

    def get_checkpoint_dir(self):
        dir = (
            f"{self.checkpoint_dir}/"
            f"tmp_{random.randint(0, 1000000)}"
            f"_{self.atom.increment()}"
        )
        return dir

    def get_partitions(self):
        return self.partitions

    def __init_spark_session(self):
        """Init spark session, set log level to warn"""
        spark = self.spark_config.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        spark.catalog.clearCache()
        spark.sparkContext.setCheckpointDir(self.checkpoint_dir)
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        return spark

    def stop_spark_session(self):
        if self.__spark:
            logger.info("stop pyspark & clean cache in ram!")
            self.__spark.catalog.clearCache()
            self.__spark.stop()
            self.__spark = None
            self.clean_tmp_data()

    def norm_min_max_cols(self, df, cols, target_cols):
        # UDF for converting column type from vector to double type
        unlist = udf(lambda x: float(list(x)[0]), DoubleType())
        assert len(cols) == len(target_cols)
        # Iterating over columns to be scaled
        for i, col_name in enumerate(cols):
            # VectorAssembler Transformation - Converting column to vector type
            assembler = VectorAssembler(
                inputCols=[col_name], outputCol=col_name + "_Vect"
            )

            # MinMaxScaler Transformation
            scaler = MinMaxScaler(
                inputCol=col_name + "_Vect", outputCol=col_name + "_Scaled"
            )

            # Pipeline of VectorAssembler and MinMaxScaler
            pipeline = Pipeline(stages=[assembler, scaler])

            # Fitting pipeline on dataframe
            df = (
                pipeline.fit(df)
                .transform(df)
                .withColumn(
                    col_name + "_Scaled",
                    unlist(col_name + "_Scaled"),
                )
                .drop(col_name + "_Vect")
                .drop(col_name)
                .withColumnRenamed(col_name + "_Scaled", target_cols[i])
            )

        return df

    def clean_tmp_data(self):
        if Path(self.checkpoint_dir).exists():
            logger.opt(depth=-1).info(
                "pyspark: cleaning all checkpoints to release disk cache"
            )
            try:
                shutil.rmtree(Path(self.checkpoint_dir))
            except Exception:
                logger.opt(depth=-1).info("checkpoint may not be removed clearly")
