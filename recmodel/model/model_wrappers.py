import gc
import glob
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from fastshap import KernelExplainer as numpy_explainer
from torch.utils.dlpack import from_dlpack

from model_configs.constant import IS_ENABLE_SHUFFLE_EACH_BATCH, MAX_FULL_TRAINED_EPOCHS
from recmodel.base.data_loaders import BaseDataLoader
from recmodel.base.model_wrappers import BaseModelWrapper
from recmodel.base.schemas.pipeline_config import DLModelParams, ModelWrapper
from recmodel.base.utils import logger
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.mlflow import MLflowMaster
from recmodel.base.utils.utils import suppress_warnings
from recmodel.model.model_core import DHashDLRM, DHashTwoTower
from recmodel.model.model_mlflow import CustomModelCheckpiont


class DeepLearningModelWrapper(BaseModelWrapper):
    def __init__(
        self,
        params: ModelWrapper,
        checkpoint_path: Union[str, Path, None] = None,
        enable_progress_bar: bool = True,
        model_date: str = "",
    ):
        pl.seed_everything(42, workers=True)
        self.mlflow_master = MLflowMaster()
        super(DeepLearningModelWrapper, self).__init__(params)
        self.enable_progress_bar = enable_progress_bar
        self.checkpoint_path = checkpoint_path
        if not IS_ENABLE_SHUFFLE_EACH_BATCH:
            self.fit_params["reload_dataloaders_every_n_epochs"] = 0
        self.trainer = self.build_trainer()
        self.feature_order = self._get_feature_order()
        self.gpu_loading = GpuLoading()
        self.gpu_loading.set_gpu_use()
        self.model_name = (
            params.cls_name.split(".")[-1].lower().replace("modelwrapper", "")
            + "_"
            + model_date
        )

    def build_model(self):
        params_dict = {
            **self.params.dict()["fit_params"],
            **self.params.dict()["model_params"],
        }
        params_dict["cls_name"] = self.params.cls_name
        params_dict.pop("max_epochs", None)
        self.mlflow_master.mlflow.log_params(params_dict)

    def _build_callback(self, ckpt_type="finetuned") -> List:
        if self.checkpoint_path is not None:
            self._remove_duplicate_model_checkpoint(ckpt_type)
            checkpoint_callback = CustomModelCheckpiont(
                monitor="auc",
                mode="max",
                dirpath=self.checkpoint_path,
                filename=self.model_name + "_" + ckpt_type,
                verbose=False,
            )
            return [checkpoint_callback]
        return []

    def build_trainer(self, enable_checkpointing=False, logger=True):
        return pl.Trainer(
            enable_progress_bar=self.enable_progress_bar,
            **self.fit_params,
            enable_checkpointing=enable_checkpointing,
            logger=logger,
        )

    def _remove_duplicate_model_checkpoint(self, ckpt_type):
        model_checkpoint_dirs = glob.glob(f"{self.checkpoint_path}/*{ckpt_type}*.ckpt")
        if len(model_checkpoint_dirs) > 0:
            for d in model_checkpoint_dirs:
                os.remove(d)

    def _get_feature_order(self):
        """Get the order of the features in the model."""
        return [
            sparse_feature.name for sparse_feature in self.model_params.sparse_features
        ] + [dense_feature.name for dense_feature in self.model_params.dense_features]

    @suppress_warnings(UserWarning)
    def get_feature_importance(
        self, input_data, num_samples: int = 1000
    ) -> Dict[str, float]:
        model = self.model.eval()

        def _numpy_predict_fn(x):
            x = torch.tensor(
                x,
                device=torch.device(
                    "cuda" if ("cuda" in str(self.model.device)) else "cpu"
                ),
            )
            positive_class = model.predict_step([x], 0)
            return positive_class.detach().cpu().numpy()

        def _cupy_predict_fn(x):
            # moder input is tensor.int because of embedding layer
            x = from_dlpack(x.toDlpack()).int()
            return model.predict_step([x], 0).detach()

        def _get_shap_values_one_sample(shap_values, index: int):
            if len(shap_values[index].shape) == 2:
                return shap_values[index][:, 0]
            assert len(shap_values[index].shape) == 1, len(shap_values[index])
            return shap_values[index]

        with torch.no_grad():
            if self.gpu_loading.is_gpu_available():
                import cupy as cp
                from cuml.explainer import KernelExplainer as cupy_explainer

                with cp.cuda.Device(self.gpu_loading.get_gpu_device_id()):
                    input_data = cp.array(input_data)
                background_data = input_data[:num_samples]
                test_data = input_data[-num_samples:]
                explainer = cupy_explainer(
                    model=_cupy_predict_fn,
                    data=background_data.copy(),
                    is_gpu_model=True,
                )
                shap_values = explainer.shap_values(test_data)
            else:
                background_data = input_data[:num_samples]
                test_data = input_data[-num_samples:]
                explainer = numpy_explainer(_numpy_predict_fn, background_data)
                shap_values = explainer.calculate_shap_values(test_data, verbose=True)

        feature_names = [feature_name for feature_name in self.feature_order]
        feature_importances = (
            np.mean(
                [
                    np.abs(_get_shap_values_one_sample(shap_values, i))
                    for i in range(len(shap_values))
                ],
                axis=0,
            )
            .astype(float)
            .tolist()
        )
        res = dict(zip(feature_names, feature_importances))
        self.mlflow_master.mlflow.log_metrics(res)
        return res

    def fit(self, data_loader: BaseDataLoader, train_full_data: bool = False):
        # train_ds = data_loader.get_train_ds(features_order=self.feature_order)
        if train_full_data:
            # val_ds = train_ds
            ckpt_type = "finetuned"
        else:
            # val_ds = data_loader.get_val_ds(features_order=self.feature_order)
            ckpt_type = "fulltrained"

        checkpoint_callback = self._build_callback(ckpt_type=ckpt_type)
        self.trainer.callbacks = self.trainer.callbacks + checkpoint_callback
        self.trainer.fit(
            self.model,
            datamodule=data_loader.get_datamodel(
                features_order=self.feature_order, train_full_data=train_full_data
            ),
        )
        self.mlflow_master.mlflow.log_metric("best_auc", self.model.best_auc)

    def predict(self, data, **kwargs):
        model = self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in data:
                preds.append(model.predict_step(batch, 0))
        if "cuda" in str(self.model.device):
            torch.cuda.empty_cache()
        else:
            gc.collect()
        return torch.concat(preds)

    def predict_rank(self, pairs, data_loader, train_pairs=None):
        pass

    def _group_tuning_params(self, tuning_params):
        tuning_dict = {}
        seletion_category_dict = {}
        for key, value in OrderedDict(sorted(tuning_params.items())).items():
            if "#list#" in key:
                common_key, _ = key.split("#list#")
                if common_key not in tuning_dict:
                    tuning_dict[common_key] = []
                tuning_dict[common_key].append(round(value))
            elif "#cate#" in key:
                common_key, category = key.split("#cate#")
                if value > seletion_category_dict.get(common_key, 0):
                    seletion_category_dict[common_key] = value
                    tuning_dict[common_key] = category
            elif "num" in key:
                tuning_dict[key] = round(value)
            elif "enable" in key:
                tuning_dict[key] = bool(round(value))
            else:
                tuning_dict[key] = value
        return tuning_dict

    def _update_model_params(self, tuning_params):
        # apply num_factors to all features
        if "num_factors" in list(tuning_params.keys()):
            for feature in self.model_params.sparse_features:
                feature.num_factors = round(tuning_params["num_factors"])

        tem_model_params_dict = self.model_params.dict()
        tuning_params = self._group_tuning_params(tuning_params)

        for key, value in tuning_params.items():
            keys = key.split(".")
            current_dict = tem_model_params_dict
            for k in keys[:-1]:
                if k not in current_dict and isinstance(current_dict, list):
                    for feature in current_dict:
                        if feature["name"] == k:
                            current_dict = feature
                else:
                    current_dict = current_dict[k]
            current_dict[keys[-1]] = value

        return DLModelParams(**tem_model_params_dict)

    def tuning_function(self, datamodule, **tuning_params):
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logger.logger.disable("")
        self.fit_params["max_epochs"] = MAX_FULL_TRAINED_EPOCHS
        if not IS_ENABLE_SHUFFLE_EACH_BATCH:
            self.fit_params["reload_dataloaders_every_n_epochs"] = 0
        self.model_params = self._update_model_params(tuning_params)

        with pl.utilities.seed.isolate_rng():
            self.model = self.build_model()
            self.trainer = self.build_trainer(enable_checkpointing=False, logger=False)
            self.trainer.fit(
                self.model,
                datamodule=datamodule,
            )
        torch.cuda.empty_cache()
        gc.collect()
        return self.model.best_auc

    def load_model(self, model_path: str):
        self.model = self.mlflow_master.mlflow.pytorch.load_model(model_uri=model_path)
        if self.gpu_loading.get_gpu_device_id() != -1:
            self.model = self.model.to(f"cuda:{self.gpu_loading.get_gpu_device_id()}")


class DHashDLRMModelWrapper(DeepLearningModelWrapper):
    def build_model(self):
        super().build_model()
        return DHashDLRM(self.model_params)


class DHashTwoTowerModelWrapper(DeepLearningModelWrapper):
    def __init__(
        self,
        params: ModelWrapper,
        checkpoint_path: Union[str, Path, None] = None,
        enable_progress_bar: bool = True,
        model_date: str = "",
    ):
        self.model_params = params.model_params
        self.user_feature_order = self._get_user_feature_order()
        self.item_feature_order = self._get_item_feature_order()
        super(DHashTwoTowerModelWrapper, self).__init__(
            params=params,
            checkpoint_path=checkpoint_path,
            enable_progress_bar=enable_progress_bar,
            model_date=model_date,
        )

    def _get_user_feature_order(self):
        """Get the order of the item features for item tower."""
        return [
            sparse_feature.name
            for sparse_feature in self.model_params.sparse_features
            if sparse_feature.is_user_feature
        ]

    def _get_item_feature_order(self):
        """Get the order of the user features for user tower."""
        return [
            sparse_feature.name
            for sparse_feature in self.model_params.sparse_features
            if not sparse_feature.is_user_feature
        ]

    def build_model(self):
        super().build_model()
        return DHashTwoTower(
            self.model_params, self.user_feature_order, self.item_feature_order
        )

    @suppress_warnings(UserWarning)
    def get_feature_importance(
        self, input_data, num_samples: int = 1000
    ) -> Dict[str, float]:
        model = self.model.eval()

        def _numpy_predict_fn(x):
            x = torch.tensor(
                x,
                device=torch.device(
                    "cuda" if ("cuda" in str(self.model.device)) else "cpu"
                ),
            )
            positive_class = model.predict_step([x], 0)
            return positive_class.detach().cpu().numpy()

        def _cupy_predict_fn(x):
            # moder input is tensor.int because of embedding layer
            x = from_dlpack(x.toDlpack()).int()
            return model.predict_step([x], 0)

        def _get_shap_values_one_sample(shap_values, index: int):
            if len(shap_values[index].shape) == 2:
                return shap_values[index][:, 0]
            assert len(shap_values[index].shape) == 1, len(shap_values[index])
            return shap_values[index]

        with torch.no_grad():
            if self.gpu_loading.is_gpu_available():
                import cupy as cp
                from cuml.explainer import KernelExplainer as cupy_explainer

                with cp.cuda.Device(self.gpu_loading.get_gpu_device_id()):
                    input_data = cp.array(input_data)
                background_data = input_data[:num_samples]
                test_data = input_data[-num_samples:]
                explainer = cupy_explainer(
                    model=_cupy_predict_fn, data=background_data, is_gpu_model=True
                )
                shap_values = explainer.shap_values(test_data)
            else:
                background_data = input_data[:num_samples]
                test_data = input_data[-num_samples:]
                explainer = numpy_explainer(_numpy_predict_fn, background_data)
                shap_values = explainer.calculate_shap_values(test_data, verbose=True)

        feature_names = [feature_name for feature_name in self.feature_order]
        feature_importances = (
            np.mean(
                [
                    np.abs(_get_shap_values_one_sample(shap_values, i))
                    for i in range(len(shap_values))
                ],
                axis=0,
            )
            .astype(float)
            .tolist()
        )
        res = dict(zip(feature_names, feature_importances))
        self.mlflow_master.mlflow.log_metrics(res)
        return res
