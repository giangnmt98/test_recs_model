import numpy as np
import torch
from torch import nn

from recmodel.base.utils import factory


class EmbeddingModule(nn.Module):
    def __init__(self, sparse_features):
        super(EmbeddingModule, self).__init__()
        self.parent_feature_dict = {}
        sparse_features_name_list = [
            feature.name for feature in sparse_features if feature.parent_feature == ""
        ]
        self.embedding_table_list = nn.ModuleList(
            [
                self.create_embedding(feature.num_inputs, feature.num_factors)
                for feature in sparse_features
                if feature.parent_feature == ""
            ]
        )
        for i, sparse_feature in enumerate(sparse_features):
            if sparse_feature.parent_feature != "":
                self.parent_feature_dict[i] = sparse_features_name_list.index(
                    sparse_feature.parent_feature
                )
            else:
                self.parent_feature_dict[i] = sparse_features_name_list.index(
                    sparse_feature.name
                )

    def create_embedding(self, num_inputs, num_factors):
        embedding = nn.Embedding(num_inputs, num_factors)
        embedding.weight.data.uniform_(
            -np.sqrt(1 / num_inputs), np.sqrt(1 / num_inputs)
        )
        return embedding

    def forward(self, x):
        all_embeddings = []
        for i in range(x.shape[0]):
            all_embeddings.append(
                self.embedding_table_list[self.parent_feature_dict[i]](x[i].squeeze())
            )
        stack_emb = torch.cat([emb.unsqueeze(1) for emb in all_embeddings], axis=1)
        return stack_emb


class CombineDoubleHashingModule(nn.Module):
    def __init__(self, sparse_features):
        super(CombineDoubleHashingModule, self).__init__()
        self.sparse_features = sparse_features
        self.v2_dict = self._get_embedding_index_v2()

    def _get_embedding_index_v2(self):
        v2_dict = {}
        for i, sparse_feature in enumerate(self.sparse_features):
            if "_v2" in sparse_feature.name:
                v2_dict[sparse_feature.name] = i
        return v2_dict

    def _get_embedding_index(self, x, index, sparse_feature, v2_dict):
        embedding_index = x[:, index]
        if v2_dict.get(sparse_feature.name + "_v2", -1) != -1:
            embedding_index += x[:, v2_dict[sparse_feature.name + "_v2"]]
        return embedding_index

    def forward(self, x):
        pre_emb_list = []
        for i, sparse_feature in enumerate(self.sparse_features):
            if "_v2" not in sparse_feature.name:
                embedding_index = self._get_embedding_index(
                    x, i, sparse_feature, self.v2_dict
                )
                pre_emb_list.append(embedding_index.unsqueeze(0))
        return torch.cat(pre_emb_list, axis=0)


class MLPModule(nn.Module):
    def __init__(
        self,
        hidden_layers_size_list,
        activation_function,
        last_layer_activation_func=False,
    ):
        super(MLPModule, self).__init__()
        self.hidden_layers = self.create_mlp(
            hidden_layers_size_list, activation_function, last_layer_activation_func
        )

    def create_mlp(self, _layers, activation_function, last_layer_activation_func):
        hidden_layers = nn.ModuleList()
        for i in range(len(_layers) - 1):
            hidden_layers.append(nn.Linear(_layers[i], _layers[i + 1]))
            if i == len(_layers) - 2:
                if last_layer_activation_func:
                    hidden_layers.append(factory.create(activation_function)())
                else:
                    # no activation for last layer
                    continue
            else:
                hidden_layers.append(factory.create(activation_function)())
        return hidden_layers

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x


class NCFModule(nn.Module):
    def __init__(
        self,
        hidden_layers_size_list,
        activation_function,
        last_layer_activation_func=False,
    ):
        super(NCFModule, self).__init__()
        self.mlp = MLPModule(
            hidden_layers_size_list, activation_function, last_layer_activation_func
        )

    def forward(self, all_embeddings):
        x = torch.cat(all_embeddings, axis=1)
        x = self.mlp(x)
        return x


class DLRMModule(nn.Module):
    def __init__(
        self,
        hidden_layers_size_list,
        activation_function,
        last_layer_activation_func=False,
    ):
        super(DLRMModule, self).__init__()
        self.mlp = MLPModule(
            hidden_layers_size_list, activation_function, last_layer_activation_func
        )

    def intertact_features(self, x):
        interaction_matrix = torch.bmm(x, torch.transpose(x, 1, 2))
        _, ni, nj = interaction_matrix.shape
        # li, lj = torch.tril_indices(ni, nj, offset=-1)
        row_indices = torch.arange(ni).unsqueeze(1).expand(ni, nj)
        col_indices = torch.arange(nj).expand(ni, nj)
        mask = row_indices > col_indices
        li = row_indices[mask]
        lj = col_indices[mask]
        interaction_values = interaction_matrix[:, li, lj]
        interaction_vectors = torch.cat([interaction_values], dim=1)
        return interaction_vectors

    def forward(self, stack_emb):
        interaction_vectors = self.intertact_features(stack_emb)
        output = self.mlp(interaction_vectors)
        return output


class LightSE(nn.Module):
    """Lightweight SELayer to refine embedding."""

    def __init__(self, field_size, embedding_size=32):
        super(LightSE, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.excitation = nn.Sequential(nn.Linear(self.field_size, self.field_size))

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, \
                expect to be 3 dimensions"
                % (len(inputs.shape))
            )
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        A = self.softmax(A)
        out = inputs * torch.unsqueeze(A, dim=2)
        return inputs + out
