import shutil
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import TrainerFn

from model_configs import constant as const
from recmodel.base.utils import logger
from recmodel.base.utils.fileops import load_parquet_data


class DLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        label_col,
        weight_col,
        procesed_train_path,
        procesed_val_path,
        features_order,
        shuffle,
        pin_memory,
        pin_memory_device,
        num_workers,
        train_cache={},
        val_cache={},
        turn_off_teardown=False,
    ):
        super().__init__()
        self.label_col = label_col
        self.weight_col = weight_col
        self.procesed_train_path = procesed_train_path
        self.procesed_val_path = procesed_val_path
        self.features_order = features_order
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.train_cache = train_cache
        self.val_cache = val_cache
        self.num_workers = num_workers
        self.turn_off_teardown = turn_off_teardown

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            BatchDataset(
                self.label_col,
                self.weight_col,
                self.features_order,
                self.procesed_train_path,
                shuffle=self.shuffle,
                _cache=self.train_cache,
            ),
            batch_size=None,
            shuffle=False,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            BatchDataset(
                self.label_col,
                self.weight_col,
                self.features_order,
                self.procesed_val_path,
                shuffle=False,
                _cache=self.val_cache,
            ),
            batch_size=None,
            shuffle=False,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
        )

    def teardown(self, stage):
        """
        Cleans up resources after the run is finished.
        Args:
            stage
        Returns:
            None
        """

        if stage == TrainerFn.FITTING:
            if not const.IS_USE_MEMORY_CACHE:
                logger.logger.info("teardown: clean temp folders")
                self.clean_after_fit_for_pyspark()
            else:
                if not self.turn_off_teardown:
                    logger.logger.info("teardown: cleaning cache in memory!")
                    keys = list(self.val_cache.keys())
                    for k in keys:
                        del self.val_cache[k]
                    del self.val_cache
                    keys = list(self.train_cache.keys())
                    for k in keys:
                        del self.train_cache[k]
                    del self.train_cache
                import gc

                gc.collect()
                torch.cuda.empty_cache()

    def clean_after_fit_for_pyspark(
        self,
    ):
        if isinstance(self.procesed_train_path, list):
            for p in self.procesed_train_path:
                if Path(p).exists():
                    shutil.rmtree(Path(p))
        else:
            if Path(self.procesed_train_path).exists():
                shutil.rmtree(Path(self.procesed_train_path))
        if isinstance(self.procesed_val_path, list):
            for p in self.procesed_val_path:
                if Path(p).exists():
                    shutil.rmtree(Path(p))
        else:
            if Path(self.procesed_val_path).exists():
                shutil.rmtree(Path(self.procesed_val_path))


class BatchDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        label_col,
        weight_col,
        features_order,
        interacted_parquet_path,
        shuffle=True,
        _cache={},
    ):
        """
        Initialize a BatchDataset.
        :param label_col:
        :param features_order:
        :param interacted_parquet_path : feature parquet dir
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A ParquetTensorDataGenerator.
        """

        self.features_order = features_order
        self.label_col = label_col
        self.weight_col = weight_col
        self.interacted_parquet_path = interacted_parquet_path
        self.shuffle = shuffle
        self._cache = _cache

        if (not const.IS_USE_MEMORY_CACHE) or (not const.IS_USE_CUDF_AT_DATALOADER):
            self.map_id_to_batch_dict = self._gen_on_file_map_id_to_batch_dict()
            # self.map_id_to_batch_dict = (
            #     self._old_version_gen_on_file_map_id_to_batch_dict()
            # )
        else:
            self.map_id_to_batch_dict = self._gen_on_cache_map_id_to_batch_dict()

        self.n_batches = len(self.map_id_to_batch_dict)

    def _old_version_gen_on_file_map_id_to_batch_dict(
        self,
    ):
        import glob

        fs = sorted(glob.glob(f"{self.interacted_parquet_path }/*/*"))
        map_id_to_batch_dict = dict(zip(range(len(fs)), fs))
        return map_id_to_batch_dict

    def _gen_on_file_map_id_to_batch_dict(self):
        import glob

        if isinstance(self.interacted_parquet_path, list):
            fs = []
            for path in self.interacted_parquet_path:
                fs += list(glob.glob(f"{path }/*/*"))
        else:
            fs = list(glob.glob(f"{self.interacted_parquet_path }/*/*"))
        df = pd.DataFrame(fs, columns=["fs"])
        df["filename_date"] = (
            df.fs.str.split("filename_date=", expand=True)
            .iloc[:, 1]
            .str.split("/")
            .str[0]
        )
        df["batch_idx"] = (
            df.fs.str.split("filename_date=", expand=True)
            .iloc[:, 1]
            .str.split("batch_idx=")
            .str[1]
        )
        df = (
            df.groupby(["filename_date", "batch_idx"])["fs"]
            .agg(list)
            .reset_index(drop=False)
        )
        df["week"] = pd.to_datetime(df["filename_date"], format="%Y%m%d")
        df["week"] = (
            df["week"].dt.isocalendar().year.astype(str)
            + "-"
            + df["week"].dt.isocalendar().week.astype(str)
        )

        # Alway shuffle batch of each day when it inits
        if const.IS_ENABLE_SHUFFLE_BASE_ON_WEEK:
            df = (
                df.groupby("week")
                .sample(frac=1, replace=False)
                .sort_values("week")
                .reset_index(drop=True)
            )
        else:
            df = (
                df.groupby("filename_date")
                .sample(frac=1, replace=False)
                .sort_values("filename_date")
                .reset_index(drop=True)
            )
        del df["week"]
        return dict(zip(df.index, df.fs))

    def _gen_on_cache_map_id_to_batch_dict(self):
        dfs = []
        if len(self._cache) == 0:
            return {}
        for part_idx in self._cache.keys():
            dfs += list(self._cache[part_idx].keys())
        df = pd.DataFrame(dfs, columns=["filename_date", "batch_idx"])
        df = df.drop_duplicates().reset_index(drop=True)

        df["week"] = pd.to_datetime(df["filename_date"], format="%Y%m%d")
        df["week"] = (
            df["week"].dt.isocalendar().year.astype(str)
            + "-"
            + df["week"].dt.isocalendar().week.astype(str)
        )

        # Alway shuffle batch of each day when it inits
        if const.IS_ENABLE_SHUFFLE_BASE_ON_WEEK:
            df = (
                df.groupby("week")
                .sample(frac=1, replace=False)
                .sort_values("week")
                .reset_index(drop=True)
            )
        else:
            df = (
                df.groupby("filename_date")
                .sample(frac=1, replace=False)
                .sort_values("filename_date")
                .reset_index(drop=True)
            )
        del df["week"]

        return dict(zip(df.index, df.values))

    def __len__(self):
        "Denotes the total number of samples"
        return self.n_batches

    def __getitem__(self, index):
        "Generates one batch of data"
        # Select sample

        ts = self._load_parquet_data_or_cache(index)
        if self.shuffle:
            ts = ts[torch.randperm(ts.shape[0])]
        # the order: self.features_order + [self.weight_col, self.label_col]
        tensor_label = ts[:, -1]
        tensor_value = ts[:, :-2]
        tensor_weight = ts[:, -2]
        batch = (tensor_value, tensor_label, tensor_weight)

        return batch

    def _load_parquet_data_or_cache(self, index):
        if (not const.IS_USE_MEMORY_CACHE) or (not const.IS_USE_CUDF_AT_DATALOADER):
            return torch.tensor(
                load_parquet_data(
                    self.map_id_to_batch_dict[index],
                    with_columns=self.features_order
                    + [self.weight_col, self.label_col],
                    process_lib="pandas",
                )
                .reset_index(drop=True)
                .values
            )
        else:
            list_ts = []
            idx = self.map_id_to_batch_dict[index]
            idx = tuple(idx)
            for part_idx in self._cache.keys():
                if idx in self._cache[part_idx]:
                    list_ts.append(self._cache[part_idx].get(idx))
            return torch.cat(list_ts, axis=0)
