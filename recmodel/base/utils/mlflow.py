import mlflow

from recmodel.base.utils.singleton import SingletonMeta


class MLflowMaster(metaclass=SingletonMeta):
    def __init__(self, experiment_name="first_experiment", uri="sqlite:///mlruns.db"):
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(uri=uri)
        self.mlflow.set_experiment(experiment_name)
