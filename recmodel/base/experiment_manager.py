import re
import shutil
from pathlib import Path
from typing import Optional, Union

from recmodel.base.schemas.pipeline_config import PipelineConfig, SearcherWrapper
from recmodel.base.utils.config import parse_pipeline_config, save_yaml_config_to_file
from recmodel.base.utils.utils import get_current_time_stamp, return_or_load


class ExperimentManager:
    """Class managing folder structure of an experiment.

    For each experiment, there will be one run_dir under _root_dir that contains
    all related information about the run, e.g. run log, config file, submission
    csv file, trained model, and a model_analysis folder.

    Attributes:
        exp_root_dir: root directory of all experiments.
        config_name: config name
        config: a pipeline config object.
        run_dir: path to the run directory, will be a subdir of _root_dir.
            This run_dir string also contains time stamp.
    """

    _log_filename = "run.log"
    _model_analysis_dir = "model_analysis"
    config_filename = "config.yaml"
    pipeline_bundle_filename = "pipeline_bundle"

    def __init__(
        self,
        config: Union[str, Path, PipelineConfig],
        custom_run_dir: Optional[str] = None,
        exp_root_dir: Path = Path("./experiments"),
        should_create_new_run_dir: bool = True,
    ):
        self.exp_root_dir = exp_root_dir
        self.config = return_or_load(config, PipelineConfig, parse_pipeline_config)
        self.config_name = self.config.config_name
        self._should_create_new_run_dir = should_create_new_run_dir
        self.run_dir = (
            Path(custom_run_dir) if custom_run_dir is not None else self._get_run_dir()
        )

    def _get_run_dir(self) -> Path:
        if not self._should_create_new_run_dir:
            return self.get_most_recent_run_dir()
        return self.exp_root_dir / f"{self.config_name}" / f"{get_current_time_stamp()}"

    def create_new_run_dir(self):
        _make_dir_if_needed(self.run_dir)
        self._save_config_to_file()

    def _save_config_to_file(self):
        save_yaml_config_to_file(self.config, self.get_config_path())

    def get_log_path(self):
        return self._make_path(self._log_filename)

    def get_config_path(self):
        return self._make_path(self.config_filename)

    def get_pipeline_bundle_path(self):
        return self._make_path(self.pipeline_bundle_filename)

    def _make_path(self, filename: str) -> Path:
        return self.run_dir / filename

    def get_model_analysis_dir(self):
        res = self._make_path(self._model_analysis_dir)
        _make_dir_if_needed(res)
        return res

    def get_most_recent_run_dir(self):
        """Returns the run_dir corresponding to the most recent timestamp.
        Raises:
            IOError if there is no such folder
        """
        subfolders = sorted(
            [
                sub
                for sub in (self.exp_root_dir / self.config_name).iterdir()
                if sub.is_dir() and bool(re.match("[0-9]{6}_[0-9]{6}", sub.name[-13:]))
            ],
        )
        if not subfolders:
            raise IOError(
                "Could not find any run directory starting with "
                f"{self.exp_root_dir.joinpath(self.config_name)}"
            )
        return subfolders[-1]

    def get_most_recent_saved_pipeline_bundle(self) -> Path:
        return self.get_most_recent_run_dir() / self.pipeline_bundle_filename


def _make_dir_if_needed(dir_path: Path):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def get_config_path_from_model_path(model_path: str) -> str:
    run_dir = Path(model_path).parent
    config_full_path = run_dir / ExperimentManager.config_filename
    assert config_full_path.exists(), (
        f"{config_full_path.as_posix()} does not exists in the same directory with "
        f"{model_path}"
    )
    return config_full_path.as_posix()


def get_saved_model_path_from_run_dir(run_dir: Path) -> Path:
    model_filename = sorted(
        [
            model_.name.split(".")[0]  # keras models end with .index, .data-
            for model_ in run_dir.iterdir()
            if model_.name.startswith("model_")
            and re.split("[_.]", model_.name)[1].isnumeric()
        ],
        key=lambda model_epoch: int(model_epoch.split("_")[-1]),
    )[-1]
    return run_dir / model_filename


class TuningExperimentManager(ExperimentManager):
    _root_dir = Path("./experiments") / "tuning"

    def __init__(
        self,
        path_to_config: Union[str, Path, PipelineConfig],
        path_to_tuning_config: Union[str, Path, SearcherWrapper],
        exp_root_dir: Path = Path("./experiments/tuning"),
        should_create_new_run_dir: bool = True,
    ):
        super(TuningExperimentManager, self).__init__(
            path_to_config,
            exp_root_dir=exp_root_dir,
            should_create_new_run_dir=should_create_new_run_dir,
        )
        self.path_to_tuning_config = path_to_tuning_config

    def get_tuning_config_path(self):
        return self._make_path("tuning_config.yaml")

    def _copy_tuning_config_file(self):
        shutil.copyfile(self.path_to_tuning_config, self.get_tuning_config_path())

    def create_new_run_dir(self):
        super(TuningExperimentManager, self).create_new_run_dir()
        self._copy_tuning_config_file()
