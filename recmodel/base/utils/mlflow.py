import os

import mlflow

from recmodel.base.utils.singleton import SingletonMeta

# Extract constants for environment variable defaults
DEFAULT_MYSQL_USER = "root"
DEFAULT_MYSQL_PASSWORD = "DsteamIC2024"
DEFAULT_MYSQL_HOST = "0.0.0.0"
DEFAULT_MYSQL_PORT = "3306"
DEFAULT_MYSQL_MLFLOW_DATABASE = "mlflow"
DEFAULT_SQLITE_URI = "sqlite:///mlflow.db"  # Path to SQLite database file


class MLflowMaster(metaclass=SingletonMeta):
    def __init__(
        self,
        experiment_name="first_experiment",
        use_tracking_server=False,
        tracking_uri=None,
    ):
        """
        Initialize MLflowMaster.

        Args:
            experiment_name (str): Name of the MLflow experiment.
            use_sqlite (bool): If True, use SQLite as the tracking URI.
             Otherwise, use MySQL.
            tracking_uri (str): Custom tracking URI if provided.
        """
        self.mysql_user = os.getenv("MYSQL_USER", DEFAULT_MYSQL_USER)
        self.mysql_password = os.getenv("MYSQL_PASSWORD", DEFAULT_MYSQL_PASSWORD)
        self.mysql_host = os.getenv("MYSQL_HOST", DEFAULT_MYSQL_HOST)
        self.mysql_port = os.getenv("MYSQL_PORT", DEFAULT_MYSQL_PORT)
        self.mysql_database = os.getenv("MYSQL_MLFLOW_DATABASE", DEFAULT_MYSQL_MLFLOW_DATABASE)
        self.mlflow = mlflow
        self.use_tracking_server = (
            use_tracking_server  # New attribute to toggle between SQLite and MySQL
        )
        # Initialize MLflow
        self.initialize_mlflow(tracking_uri, experiment_name)

    def initialize_mlflow(self, tracking_uri, experiment_name):
        """
        Set up the MLflow tracking URI and experiment.

        Args:
            tracking_uri (str): Custom tracking URI if provided.
            experiment_name (str): Name of the MLflow experiment to set.
        """
        # Determine tracking URI based on the storage type
        if tracking_uri is None:
            if self.use_tracking_server:
                tracking_uri = (
                    f"mysql+mysqldb://{self.mysql_user}:{self.mysql_password}"
                    f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
                )
            else:
                tracking_uri = DEFAULT_SQLITE_URI

        # Set MLflow tracking URI and experiment
        self.mlflow.set_tracking_uri(uri=tracking_uri)
        self.mlflow.set_experiment(experiment_name)
