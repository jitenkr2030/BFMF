"""
MLflow integration for BharatFM experiment tracking
"""

import os
import json
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import shutil


class BharatMLflowTracker:
    """MLflow-based experiment tracking for BharatFM"""
    
    def __init__(
        self,
        tracking_uri: str = None,
        experiment_name: str = "bharatfm",
        registry_uri: str = None,
        artifact_location: str = None
    ):
        self.tracking_uri = tracking_uri or "file:///tmp/mlruns"
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri
        self.artifact_location = artifact_location
        
        self.logger = logging.getLogger(__name__)
        self.experiment_id = None
        self.run_id = None
        
        # Setup MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set registry URI
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
            
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location
            )
        else:
            self.experiment_id = experiment.experiment_id
            
        self.logger.info(f"MLflow tracking setup complete. Experiment ID: {self.experiment_id}")
        
    def start_run(
        self,
        run_name: str = None,
        description: str = None,
        tags: Dict[str, str] = None,
        nested: bool = False
    ) -> str:
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Start run
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            description=description,
            tags=tags,
            nested=nested
        )
        
        self.run_id = mlflow.active_run().info.run_id
        self.logger.info(f"Started MLflow run: {self.run_id}")
        
        return self.run_id
        
    def end_run(self):
        """End the current MLflow run"""
        if mlflow.active_run():
            mlflow.end_run()
            self.logger.info(f"Ended MLflow run: {self.run_id}")
            self.run_id = None
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        mlflow.log_params(params)
        self.logger.info(f"Logged {len(params)} parameters")
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics, step=step)
        self.logger.info(f"Logged {len(metrics)} metrics")
        
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact to MLflow"""
        mlflow.log_artifact(local_path, artifact_path)
        self.logger.info(f"Logged artifact: {local_path}")
        
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """Log directory of artifacts to MLflow"""
        mlflow.log_artifacts(local_dir, artifact_path)
        self.logger.info(f"Logged artifacts from: {local_dir}")
        
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: str = None,
        signature: Any = None,
        input_example: Any = None
    ):
        """Log model to MLflow"""
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
        self.logger.info(f"Logged model to: {artifact_path}")
        
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, artifact_file)
        self.logger.info(f"Logged dictionary to: {artifact_file}")
        
    def log_text(self, text: str, artifact_file: str):
        """Log text as artifact"""
        mlflow.log_text(text, artifact_file)
        self.logger.info(f"Logged text to: {artifact_file}")
        
    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure as artifact"""
        mlflow.log_figure(figure, artifact_file)
        self.logger.info(f"Logged figure to: {artifact_file}")
        
    def log_image(self, image, artifact_file: str):
        """Log image as artifact"""
        mlflow.log_image(image, artifact_file)
        self.logger.info(f"Logged image to: {artifact_file}")
        
    def set_tag(self, key: str, value: str):
        """Set tag on current run"""
        mlflow.set_tag(key, value)
        
    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags on current run"""
        mlflow.set_tags(tags)
        
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        self.log_params({
            f"dataset_{key}": value for key, value in dataset_info.items()
        })
        
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information"""
        self.log_params({
            f"model_{key}": value for key, value in model_info.items()
        })
        
    def log_training_info(self, training_info: Dict[str, Any]):
        """Log training information"""
        self.log_params({
            f"training_{key}": value for key, value in training_info.items()
        })
        
    def log_evaluation_results(self, eval_results: Dict[str, Any]):
        """Log evaluation results"""
        # Log metrics
        metrics = {}
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                metrics[f"eval_{key}"] = value
                
        if metrics:
            self.log_metrics(metrics)
            
        # Log detailed results as artifact
        self.log_dict(eval_results, "evaluation_results.json")
        
    def log_system_info(self):
        """Log system information"""
        import platform
        import torch
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        self.log_params(system_info)
        
    def register_model(
        self,
        model_uri: str,
        name: str,
        description: str = None,
        tags: Dict[str, str] = None
    ):
        """Register model in MLflow Model Registry"""
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            description=description,
            tags=tags
        )
        
        self.logger.info(f"Registered model {name} version {model_version.version}")
        return model_version
        
    def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ):
        """Transition model version to different stage"""
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        
        self.logger.info(f"Transitioned model {name} version {version} to stage {stage}")
        
    def delete_model_version(self, name: str, version: str):
        """Delete model version"""
        client = mlflow.MlflowClient()
        client.delete_model_version(name=name, version=version)
        
        self.logger.info(f"Deleted model {name} version {version}")
        
    def search_runs(
        self,
        experiment_ids: List[str] = None,
        filter_string: str = "",
        run_view_type: str = "ACTIVE_ONLY",
        max_results: int = 1000
    ) -> List:
        """Search for runs"""
        if experiment_ids is None:
            experiment_ids = [self.experiment_id]
            
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            run_view_type=run_view_type,
            max_results=max_results
        )
        
        return runs
        
    def get_run(self, run_id: str) -> Dict:
        """Get run information"""
        run = mlflow.get_run(run_id)
        return run.to_dict()
        
    def list_artifacts(self, run_id: str = None, path: str = "") -> List[str]:
        """List artifacts for a run"""
        if run_id is None:
            run_id = self.run_id
            
        if run_id is None:
            raise ValueError("No active run or run_id provided")
            
        artifacts = mlflow.list_artifacts(run_id, path)
        return [artifact.path for artifact in artifacts]
        
    def download_artifacts(self, run_id: str = None, path: str = "", dst_path: str = None):
        """Download artifacts from a run"""
        if run_id is None:
            run_id = self.run_id
            
        if run_id is None:
            raise ValueError("No active run or run_id provided")
            
        if dst_path is None:
            dst_path = tempfile.mkdtemp()
            
        mlflow.download_artifacts(run_id, path, dst_path)
        return dst_path
        
    def get_experiment(self, experiment_id: str = None) -> Dict:
        """Get experiment information"""
        if experiment_id is None:
            experiment_id = self.experiment_id
            
        experiment = mlflow.get_experiment(experiment_id)
        return experiment.to_dict()
        
    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        experiments = mlflow.list_experiments()
        return [exp.to_dict() for exp in experiments]
        
    def delete_experiment(self, experiment_id: str):
        """Delete experiment"""
        mlflow.delete_experiment(experiment_id)
        self.logger.info(f"Deleted experiment: {experiment_id}")
        
    def restore_experiment(self, experiment_id: str):
        """Restore deleted experiment"""
        mlflow.restore_experiment(experiment_id)
        self.logger.info(f"Restored experiment: {experiment_id}")
        
    def create_registered_model(self, name: str, description: str = None, tags: Dict[str, str] = None):
        """Create registered model"""
        client = mlflow.MlflowClient()
        client.create_registered_model(name, description, tags)
        self.logger.info(f"Created registered model: {name}")
        
    def get_registered_model(self, name: str) -> Dict:
        """Get registered model information"""
        client = mlflow.MlflowClient()
        model = client.get_registered_model(name)
        return model.to_dict()
        
    def list_registered_models(self, max_results: int = 100) -> List[Dict]:
        """List registered models"""
        client = mlflow.MlflowClient()
        models = client.list_registered_models(max_results=max_results)
        return [model.to_dict() for model in models]
        
    def search_model_versions(self, filter_string: str = "", max_results: int = 100) -> List[Dict]:
        """Search model versions"""
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(filter_string, max_results)
        return [version.to_dict() for version in versions]
        
    def get_model_version(self, name: str, version: str) -> Dict:
        """Get model version information"""
        client = mlflow.MlflowClient()
        model_version = client.get_model_version(name, version)
        return model_version.to_dict()
        
    def get_model_version_stages(self, name: str, version: str) -> List[str]:
        """Get available stages for model version"""
        client = mlflow.MlflowClient()
        model_version = client.get_model_version(name, version)
        return model_version.current_stage
        
    def add_model_version_tag(self, name: str, version: str, key: str, value: str):
        """Add tag to model version"""
        client = mlflow.MlflowClient()
        client.set_model_version_tag(name, version, key, value)
        
    def delete_model_version_tag(self, name: str, version: str, key: str):
        """Delete tag from model version"""
        client = mlflow.MlflowClient()
        client.delete_model_version_tag(name, version, key)
        
    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """Get download URI for model version"""
        client = mlflow.MlflowClient()
        model_version = client.get_model_version(name, version)
        return f"models:/{name}/{version}"


class ExperimentTracker:
    """High-level experiment tracking interface"""
    
    def __init__(self, tracker: BharatMLflowTracker):
        self.tracker = tracker
        self.current_run_info = {}
        
    def start_experiment(
        self,
        experiment_name: str,
        run_name: str = None,
        description: str = None,
        tags: Dict[str, str] = None,
        config: Dict[str, Any] = None
    ) -> str:
        """Start a new experiment"""
        # Set experiment name
        self.tracker.experiment_name = experiment_name
        self.tracker.setup_mlflow()
        
        # Start run
        run_id = self.tracker.start_run(run_name, description, tags)
        
        # Log configuration
        if config:
            self.tracker.log_params(config)
            self.tracker.log_dict(config, "config.json")
            
        # Log system info
        self.tracker.log_system_info()
        
        # Store run info
        self.current_run_info = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "config": config
        }
        
        return run_id
        
    def log_training_step(self, step: int, metrics: Dict[str, float]):
        """Log training step metrics"""
        self.tracker.log_metrics(metrics, step=step)
        
    def log_epoch_results(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch results"""
        epoch_metrics = {f"epoch_{key}": value for key, value in metrics.items()}
        self.tracker.log_metrics(epoch_metrics, step=epoch)
        
    def log_model_checkpoint(self, model, checkpoint_name: str, metrics: Dict[str, float] = None):
        """Log model checkpoint"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / checkpoint_name
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), checkpoint_path / "model.pt")
            
            # Save metrics if provided
            if metrics:
                with open(checkpoint_path / "metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                    
            # Log as artifact
            self.tracker.log_artifact(str(checkpoint_path), f"checkpoints/{checkpoint_name}")
            
    def log_evaluation_results(self, eval_results: Dict[str, Any], step: int = None):
        """Log evaluation results"""
        self.tracker.log_evaluation_results(eval_results)
        
        if step is not None:
            # Log summary metrics
            summary_metrics = {}
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    summary_metrics[f"eval_{key}"] = value
                    
            if summary_metrics:
                self.tracker.log_metrics(summary_metrics, step=step)
                
    def log_final_model(self, model, model_name: str, metrics: Dict[str, float] = None):
        """Log final trained model"""
        # Log model to MLflow
        self.tracker.log_model(model, "model", registered_model_name=model_name)
        
        # Log metrics
        if metrics:
            self.tracker.log_metrics(metrics)
            
        # Log model info
        model_info = {
            "model_name": model_name,
            "model_type": "pytorch",
            "framework": "bharatfm",
            "timestamp": datetime.now().isoformat()
        }
        self.tracker.log_dict(model_info, "model_info.json")
        
    def finish_experiment(self, status: str = "FINISHED"):
        """Finish the experiment"""
        # Set final status
        self.tracker.set_tag("status", status)
        self.tracker.set_tag("end_time", datetime.now().isoformat())
        
        # End run
        self.tracker.end_run()
        
        # Clear current run info
        self.current_run_info = {}
        
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Dict:
        """Get the best run based on a metric"""
        runs = self.tracker.search_runs()
        
        if not runs:
            return None
            
        # Filter runs that have the metric
        valid_runs = []
        for run in runs:
            metrics = run.data.metrics
            if metric_name in metrics:
                valid_runs.append((run, metrics[metric_name]))
                
        if not valid_runs:
            return None
            
        # Sort by metric value
        valid_runs.sort(key=lambda x: x[1], reverse=not ascending)
        
        return valid_runs[0][0].to_dict()
        
    def compare_runs(self, run_ids: List[str], metrics: List[str] = None) -> Dict[str, Any]:
        """Compare multiple runs"""
        comparison = {}
        
        for run_id in run_ids:
            run = self.tracker.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run["info"]["run_name"],
                "start_time": run["info"]["start_time"],
                "metrics": run["data"]["metrics"],
                "params": run["data"]["params"]
            }
            
            if metrics:
                run_data["selected_metrics"] = {
                    key: run["data"]["metrics"].get(key) for key in metrics
                }
                
            comparison[run_id] = run_data
            
        return comparison