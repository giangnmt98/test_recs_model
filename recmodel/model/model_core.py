from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn

from model_configs.constant import ENABLE_LOGGING_LOSS
from recmodel.base.schemas.pipeline_config import DLModelParams
from recmodel.base.utils import factory, logger
from recmodel.base.utils.mlflow import MLflowMaster
from recmodel.model.model_modules import (
    CombineDoubleHashingModule,
    DLRMModule,
    EmbeddingModule,
    LightSE,
)


class BaseDeepLearningModel(pl.LightningModule):
    def __init__(self, config: DLModelParams):
        super().__init__()
        self.config = config
        self.criterion: Any = torchmetrics.classification.BinaryAUROC()
        self.validation_step_outputs: List[Any] = []
        self.validation_step_targets: List[Any] = []
        self.training_step_outputs: List[Any] = []
        self.training_step_targets: List[Any] = []
        self.best_auc = 0
        self.mlflow_master = MLflowMaster()
        self.feature_order = [
            sparse_feature.name
            for sparse_feature in self.config.sparse_features
            if sparse_feature.is_user_feature
        ]

    def forward(self, x):
        raise NotImplementedError

    def forward_for_training(self, x):
        return self(x)

    def compute_loss(self, output, target, weight=None):
        return torch.mean(
            torch.nn.BCEWithLogitsLoss()(
                torch.flatten(output.float()), (target > 1).float()
            )
            * weight
        )

    def training_step(self, batch, batch_idx):
        data, target, weight = batch
        output = self.forward_for_training(data.long())
        loss = self.compute_loss(output, target, weight)
        if ENABLE_LOGGING_LOSS:
            self.mlflow_master.mlflow.log_metric("loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        output = nn.Sigmoid()(self(data.long()))
        self.validation_step_outputs.append(torch.flatten(output.float()))
        self.validation_step_targets.append((target > 1).float())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return nn.Sigmoid()(self(batch[0].long()))

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        all_targets = torch.cat(self.validation_step_targets)
        auc = self.criterion(all_preds, all_targets).item()
        logger.logger.info("=============================================")
        logger.logger.info("Validation AUC: {:.4f}".format(auc))
        logger.logger.info("=============================================")
        self.mlflow_master.mlflow.log_metric("auc", auc)
        self.log("auc", auc)
        if self.best_auc < auc:
            self.best_auc = auc
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_targets.clear()  # free memory

    def configure_optimizers(self):
        assert self.config.optimizer, "Optimizer is not defined"
        optimizer = factory.create(self.config.optimizer.cls_name)(
            self.parameters(), **self.config.optimizer.params
        )
        return optimizer


class DHashDLRM(BaseDeepLearningModel):
    def __init__(self, config: DLModelParams):
        super().__init__(config)
        enable_sparse_feature = [
            feature for feature in config.sparse_features if feature.enable
        ]
        self.num_input_feature = sum(
            [1 for feature in enable_sparse_feature if feature.num_inputs != -1]
        )
        self.concatenated_embedding_size_for_spare_features = int(
            (self.num_input_feature * (self.num_input_feature - 1) / 2)
        )
        hidden_layers_size_list = [
            self.concatenated_embedding_size_for_spare_features
        ] + config.hidden_layers
        self.core = nn.Sequential(
            CombineDoubleHashingModule(enable_sparse_feature),
            EmbeddingModule(
                [
                    feature
                    for feature in enable_sparse_feature
                    if "_v2" not in feature.name
                ]
            ),
            DLRMModule(
                hidden_layers_size_list,
                config.activation_function,
            ),
        )

    def forward(self, x):
        x = self.core(x)
        return x


class TwoTower(pl.LightningModule):
    def __init__(
        self,
        config: DLModelParams,
        user_feature_order: list[str],
        item_feature_order: list[str],
    ):
        super().__init__()
        self.config = config
        self.criterion = torchmetrics.classification.BinaryAUROC()
        self.user_feature_order = user_feature_order
        self.item_feature_order = item_feature_order
        # if we place the embedding layers inside the dict, they are not moved to GPU
        # automatically with model.to(device)
        self.embedding_dict = nn.ModuleDict(
            {
                sparse_feature.name: self.create_embedding(
                    sparse_feature.num_inputs, sparse_feature.num_factors
                )
                for sparse_feature in config.sparse_features
                if sparse_feature.enable and sparse_feature.num_inputs != -1
            }
        )
        self.concatenated_embedding_size = sum(
            self.embedding_dict[sparse_feature.name].embedding_dim
            for sparse_feature in config.sparse_features
            if sparse_feature.enable and sparse_feature.num_inputs != -1
        )
        _layers = [self.concatenated_embedding_size] + config.hidden_layers
        self.hidden_layers = self.create_mlp(_layers)
        self.validation_step_outputs: List[Any] = []
        self.validation_step_targets: List[Any] = []
        self.training_step_outputs: List[Any] = []
        self.training_step_targets: List[Any] = []
        self.best_auc = 0
        self.mlflow_master = MLflowMaster()

    def create_embedding(self, num_inputs, num_factors):
        embedding = nn.Embedding(num_inputs, num_factors)
        embedding.weight.data.uniform_(
            -np.sqrt(1 / num_inputs), np.sqrt(1 / num_inputs)
        )
        return embedding

    def create_mlp(self, _layers):
        hidden_layers = nn.ModuleList()
        for i in range(len(_layers) - 1):
            hidden_layers.append(nn.Linear(_layers[i], _layers[i + 1]))
            if i == len(_layers) - 2:
                # no activation for last layer
                continue
            else:
                hidden_layers.append(factory.create(self.config.activation_function)())
        return hidden_layers

    def forward(self, x):
        output = torch.cat(
            [
                self.embedding_dict[sparse_feature.name](x[:, i])
                for i, sparse_feature in enumerate(self.config.sparse_features)
                if sparse_feature.enable
            ],
            axis=1,
        )
        for layer in self.hidden_layers:
            output = layer(output)
        return output

    def training_step(self, batch, batch_idx):
        data, target, weight = batch
        output = self(data.long())
        loss = torch.mean(
            torch.nn.BCELoss()(output.float(), (target > 1).float()) * weight
        )
        if ENABLE_LOGGING_LOSS:
            self.mlflow_master.mlflow.log_metric("loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        output = self(data.long())
        self.validation_step_outputs.append(output.float())
        self.validation_step_targets.append((target > 1).float())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0].long())

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs)
        all_targets = torch.cat(self.validation_step_targets)
        auc = self.criterion(all_preds, all_targets).item()
        logger.logger.info("=============================================")
        logger.logger.info("Validation AUC: {:.4f}".format(auc))
        logger.logger.info("=============================================")
        self.mlflow_master.mlflow.log_metric("auc", auc)
        self.log("auc", auc)
        if self.best_auc < auc:
            self.best_auc = auc
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_targets.clear()  # free memory

    def configure_optimizers(self):
        assert self.config.optimizer, "Optimizer is not defined"
        optimizer = factory.create(self.config.optimizer.cls_name)(
            self.parameters(), **self.config.optimizer.params
        )
        return optimizer


class DHashTwoTower(TwoTower):
    def __init__(
        self,
        config: DLModelParams,
        user_feature_order: list[str],
        item_feature_order: list[str],
    ):
        super().__init__(config, user_feature_order, item_feature_order)
        num_factors_user_feature_list = [
            sparse_feature.num_factors
            for sparse_feature in config.sparse_features
            if sparse_feature.enable
            and sparse_feature.name in self.user_feature_order
            and "v2" not in sparse_feature.name
        ]
        self.concatenated_embedding_size_user_tower = sum(num_factors_user_feature_list)
        num_factors_item_feature_list = [
            sparse_feature.num_factors
            for sparse_feature in config.sparse_features
            if sparse_feature.enable
            and sparse_feature.name in self.item_feature_order
            and "v2" not in sparse_feature.name
        ]
        self.concatenated_embedding_size_item_tower = sum(num_factors_item_feature_list)
        _layers_user_tower = [
            self.concatenated_embedding_size_user_tower
        ] + config.hidden_layers
        self.hidden_layers_user_tower = self.create_mlp(_layers_user_tower)
        _layers_item_tower = [
            self.concatenated_embedding_size_item_tower
        ] + config.hidden_layers
        self.hidden_layers_item_tower = self.create_mlp(_layers_item_tower)

        self.refined_user_emb_module = LightSE(
            len(num_factors_user_feature_list), num_factors_user_feature_list[0]
        )
        self.refined_item_emb_module = LightSE(
            len(num_factors_item_feature_list), num_factors_item_feature_list[0]
        )

    def apply_emb(self, x, feature_order, refined_emb_module):
        index_hashed_user_id_v2 = None
        index_hashed_item_id_v2 = None
        for i, sparse_feature in enumerate(feature_order):
            if sparse_feature == "hashed_item_id_v2":
                index_hashed_item_id_v2 = i
            elif sparse_feature == "hashed_user_id_v2":
                index_hashed_user_id_v2 = i

        emb_dict = {}
        for i, sparse_feature in enumerate(feature_order):
            if sparse_feature == "hashed_item_id":
                emb_dict[sparse_feature] = self.embedding_dict[sparse_feature](
                    x[:, i] + x[:, index_hashed_item_id_v2]
                )
            elif sparse_feature == "hashed_user_id":
                emb_dict[sparse_feature] = self.embedding_dict[sparse_feature](
                    x[:, i] + x[:, index_hashed_user_id_v2]
                )
            elif "v2" not in sparse_feature:
                emb_dict[sparse_feature] = self.embedding_dict[sparse_feature](x[:, i])

        stack_emb = torch.cat(
            [
                emb_dict[sparse_feature].unsqueeze(1)
                for sparse_feature in feature_order
                if "v2" not in sparse_feature
            ],
            axis=1,
        )
        output_vector = torch.flatten(refined_emb_module(stack_emb), start_dim=1)
        return output_vector

    def apply_mlp(self, x, hidden_layers):
        for layer in hidden_layers:
            x = layer(x)
        return x

    def split_tower_input(self, x):
        user_index = []
        item_index = []
        for i, sparse_feature in enumerate(self.config.sparse_features):
            if sparse_feature.name in self.user_feature_order:
                user_index.append(i)
            elif sparse_feature.name in self.item_feature_order:
                item_index.append(i)
        x_user = x[:, user_index]
        x_item = x[:, item_index]
        return x_user, x_item

    def user_tower(self, x):
        concated_user_emb = self.apply_emb(
            x.long(), self.user_feature_order, self.refined_user_emb_module
        )
        user_vector = self.apply_mlp(concated_user_emb, self.hidden_layers_user_tower)
        user_vector = torch.nn.functional.normalize(user_vector, p=2, dim=1)
        return user_vector

    def item_tower(self, x):
        concated_item_emb = self.apply_emb(
            x.long(), self.item_feature_order, self.refined_item_emb_module
        )
        item_vector = self.apply_mlp(concated_item_emb, self.hidden_layers_item_tower)
        item_vector = torch.nn.functional.normalize(item_vector, p=2, dim=1)
        return item_vector

    def interacted_top(self, user_vector, item_vector):
        # print(torch.sum(user_vector * item_vector, dim=-1))
        return (torch.nn.functional.cosine_similarity(user_vector, item_vector) + 1) / 2

    def forward(self, x):
        x_user, x_item = self.split_tower_input(x)
        user_vector = self.user_tower(x_user)
        item_vector = self.item_tower(x_item)
        output = self.interacted_top(user_vector, item_vector)
        return output
