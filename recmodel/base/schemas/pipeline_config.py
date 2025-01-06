from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class DataLoader(BaseModel):
    cls_name: str  # name of DataLoader class
    # list of features going in to the model
    user_id_col: str
    item_id_col: str
    # time order column event
    time_order_event_col: str = ""
    popularity_item_group_col: str = ""
    label_col: str = ""
    duration_col: str = ""
    weight_col: str = ""
    batch_size: int = 0
    shuffle: bool = False
    debug: bool = False
    account_feature_names: List[str] = []
    user_feature_names: List[str] = []
    item_feature_names: List[str] = []
    context_feature_names: List[str] = []
    user_context_feature_names: List[str] = []
    item_context_feature_names: List[str] = []
    popularity_sample_groups: List[str] = []
    factor_popularity_sample_groups: List[float] = []
    tail_sample_groups: List[str] = []
    factor_tail_sample_groups: List[float] = []

    # filters to determine training, validation, and test dataset. For each
    # dataset, rows that are met all conditions will be selected.
    train_filters: List[str] = []
    validation_filters: List[str] = []
    content_filters: List[str] = []


class ClusterDataLoader(BaseModel):
    cls_name: str  # name of DataLoader class
    feature_manager_name: str = ""
    feature_manager_class: str = ""
    feature_config_path: str = ""
    feature_names: List[str] = []
    additional_feature_names: List[str] = []


class SparseFeature(BaseModel):
    name: str
    num_inputs: int  # -1 for define in code.
    num_factors: int
    enable: bool = True
    is_user_feature: bool = False
    is_interaction_feature: bool = False
    parent_feature: str = ""


class DenseFeature(BaseModel):
    name: str
    num_factors: int = 1
    enable: bool = True
    is_user_feature: bool = False


class Optimizer(BaseModel):
    cls_name: str
    params: Dict = {}


class DLModelParams(BaseModel):
    sparse_features: List[SparseFeature]
    dense_features: List[DenseFeature] = []
    hidden_layers: List[int]
    activation_function: str = "torch.nn.LeakyReLU"
    interaction_layers: List[int] = []
    optimizer: Optional[Optimizer] = None


class ModelWrapper(BaseModel):
    cls_name: str  # name of model_wrapper class
    model_params: Union[DLModelParams, Dict] = {}
    fit_params: Dict = {}


class ModelAnalysis(BaseModel):
    # list of metrics to compute.
    # Each element must be the name of a subclass of vk.metrics.BaseMetric
    metrics: List[str] = []

    # list of features for analysis
    by_features: List[str] = []

    # list of external user features for analysis
    by_external_user_features: List[str] = []

    # list of external item features for analysis
    by_external_item_features: List[str] = []

    # to separate dataset by column
    sep_col: str = ""


class PipelineConfig(BaseModel):
    config_name: str
    with_train_full_data: bool = False
    data_loader: DataLoader
    model_wrapper: ModelWrapper
    model_analysis: Optional[ModelAnalysis] = None


class ClusterPipelineConfig(BaseModel):
    config_name: str
    data_loader: ClusterDataLoader
    model_wrapper: ModelWrapper
    save_data_path: str


class DataLoaderFilter(BaseModel):
    # filters to determine training, validation, and test dataset. For each
    # dataset, rows that are met all conditions will be selected.
    train_filters: List[str] = []
    validation_filters: List[str] = []


class SearcherWrapper(BaseModel):
    searcher_cls: str
    estimated_num_user: int
    searcher_params: Dict = {}
    data_loader_filter: DataLoaderFilter
    tuning_params: Dict = {}
