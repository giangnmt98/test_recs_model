import argparse
from datetime import datetime

from recmodel.base.utils.config import load_simple_dict_config
from recmodel.base.utils.gpu import GpuLoading
from recmodel.base.utils.mlflow import MLflowMaster
from recmodel.src import TestRecModel
import logging
from sqlalchemy import create_engine

# Tắt log từ py4j
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("onnxscript").setLevel(logging.ERROR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate recommendation model."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        nargs="?",
        default="config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_simple_dict_config(args.config_path)
    model_name = config["model_name"]
    use_tracking_server = config["use_mlflow_tracking_server"]
    print("AAAAAAAAAAAA")

    engine = create_engine("mysql+mysqldb://root:DsteamIC2024@localhost:3306/mlflow")
    connection = engine.connect()
    logging.info("Successfully connected to the MySQL database.")
    
    with MLflowMaster(experiment_name=model_name, use_tracking_server=use_tracking_server).mlflow.start_run(
        run_name=str(config["infer_date"]) + "_" + datetime.now().strftime("%H%M%S")
    ):
        if GpuLoading().is_gpu_available():
            device_id = 0
        else:
            device_id = -1

        # test_rec_model = TestRecModel(
        #     infer_date=config["infer_date"],
        #     num_days_to_train=config["num_days_to_train"],
        #     config_path=config["model_config_path"] + f"{model_name}.yaml",
        #     device_id=device_id,
        #     data_path=config["data_path"],
        #     cpu_process_lib=config["cpu_process_lib"],
        # )
        # test_rec_model.run()
        # test_rec_model.evaluate()
