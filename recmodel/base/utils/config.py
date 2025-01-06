from pathlib import Path
from typing import Union

import yaml

from recmodel.base.schemas import pipeline_config


def parse_pipeline_config(yaml_path: str) -> pipeline_config.PipelineConfig:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return pipeline_config.PipelineConfig(**config)


def parse_tuning_config(yaml_path: str) -> pipeline_config.SearcherWrapper:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        return pipeline_config.SearcherWrapper(**config)


def load_simple_dict_config(path_config: Union[str, Path]) -> dict:
    with open(path_config) as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config_to_file(
    yaml_config,
    yaml_path: Union[str, Path],
) -> None:
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_config, yaml_file, default_flow_style=False)
