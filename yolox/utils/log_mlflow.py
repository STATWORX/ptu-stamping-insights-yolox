from azureml.core import Run
import mlflow
from mlflow.tracking import MlflowClient


class MLFlowLogger():
    """Logs parameters and metrics to MLFlow for use in AzureML.
    """
    def __init__(self):
        run = Run.get_context()
        mlflow_tracking_uri = run.experiment.workspace.get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        if mlflow.active_run() is None:
            self.mlflow_run = mlflow.start_run()
        else:
            self.mlflow_run = mlflow.active_run()

        self.mlflow_run_id = self.mlflow_run.info.run_id

    def log_param(self, key, value):
        MlflowClient().log_param(
                run_id=self.mlflow_run_id,
                key=key,
                value=value,
        )

    def log_metric(self, key, value, step=0):
        MlflowClient().log_metric(
                run_id=self.mlflow_run_id,
                key=key,
                value=value,
                step=step
        )

    def log_metrics(self, metrics, step=0):
        MlflowClient().log_metrics(
                run_id=self.mlflow_run_id,
                metrics=metrics,
                step=step
        )

    def log_text(self, text, artifact_file):
        MlflowClient().log_text(
                run_id=self.mlflow_run_id,
                text=text,
                artifact_file=artifact_file
        )