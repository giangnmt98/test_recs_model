import shutil
from datetime import datetime
from pathlib import Path

import pytest

from recmodel.base.utils.mlflow import MLflowMaster
from recmodel.src import TestRecModel


class TestDailyPipeline:
    @pytest.fixture(autouse=True)
    def copy_config_folder(self):
        config_folder = Path("model_configs")
        if not config_folder.exists():
            shutil.copytree(
                "../model_configs",
                config_folder,
            )

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        test_dir = request.fspath.dirname
        monkeypatch.chdir(request.fspath.dirname)
        return test_dir

    def test_clean_up_at_the_start(self):
        self.clean_folder()

    def test_finetune_and_evaluate_pandas(self):
        infer_date = 20230430
        model_name = "dhashdlrm"
        with MLflowMaster().mlflow.start_run(
            run_name=str(infer_date) + "_" + datetime.now().strftime("%H%M%S")
        ):
            test_rec_model = TestRecModel(
                infer_date=infer_date,
                num_days_to_train=90,
                config_path=f"model_configs/models/{model_name}.yaml",
                device_id=-1,
                data_path="data/features",
                cpu_process_lib="pandas",
            )
            test_rec_model.run()
            test_rec_model.evaluate()

    def test_clean_up_at_the_end(self):
        self.clean_folder()

    def clean_folder(self):
        paths = [
            "experiments/",
            "lightning_logs/",
            "mlruns.db",
            "mlruns/",
        ]
        import os

        for f in paths:
            if "/" not in f and os.path.exists(f):
                os.remove(f)
            else:
                shutil.rmtree(f, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def remove_test_dir():
        paths = [
            "experiments/",
            "lightning_logs/",
            "model_configs/",
            "mlruns.db",
            "mlruns/",
        ]
        import os

        for f in paths:
            if "/" not in f and os.path.exists(f):
                os.remove(f)
            else:
                shutil.rmtree(
                    f"{os.path.dirname(os.path.realpath(__file__))}/{f}",
                    ignore_errors=True,
                )

    request.addfinalizer(remove_test_dir)
