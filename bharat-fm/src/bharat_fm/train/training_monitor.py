"""
Distributed Training Monitoring and Logging System
Real-time monitoring, metrics collection, and distributed logging for Bharat-FM
"""

import os
import json
import time
import threading
import queue
import psutil
import GPUtil
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for training monitoring"""
    # Monitoring settings
    enable_monitoring: bool = True
    monitor_interval: float = 1.0  # seconds
    
    # Metrics collection
    collect_gpu_metrics: bool = True
    collect_cpu_metrics: bool = True
    collect_memory_metrics: bool = True
    collect_network_metrics: bool = True
    
    # Logging settings
    log_to_tensorboard: bool = True
    log_to_wandb: bool = False
    log_to_file: bool = True
    log_to_console: bool = True
    
    # TensorBoard settings
    tensorboard_dir: str = "./tensorboard"
    tensorboard_flush_secs: int = 30
    
    # WandB settings
    wandb_project: str = "bharat-fm"
    wandb_entity: str = None
    wandb_run_name: str = None
    
    # File logging
    log_dir: str = "./logs"
    metrics_file: str = "training_metrics.json"
    log_file: str = "training_monitor.log"
    
    # Alerting
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "gpu_memory_usage": 0.95,
        "cpu_memory_usage": 0.95,
        "gpu_utilization": 0.95,
        "loss_explosion": 10.0,
        "gradient_nan": True
    })
    
    # Visualization
    create_plots: bool = True
    plot_interval: int = 100
    plot_dir: str = "./plots"

@dataclass
class TrainingMetrics:
    """Training metrics data structure"""
    timestamp: float
    step: int
    epoch: int
    
    # Loss metrics
    train_loss: float
    val_loss: Optional[float] = None
    test_loss: Optional[float] = None
    
    # Accuracy metrics
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    
    # Learning rate
    learning_rate: float
    
    # Performance metrics
    throughput: Optional[float] = None
    latency: Optional[float] = None
    
    # Memory metrics
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_memory_used: Optional[float] = None
    cpu_memory_total: Optional[float] = None
    
    # GPU metrics
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_power_usage: Optional[float] = None
    
    # CPU metrics
    cpu_utilization: Optional[float] = None
    cpu_temperature: Optional[float] = None
    
    # Gradient metrics
    gradient_norm: Optional[float] = None
    gradient_max: Optional[float] = None
    gradient_min: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.is_running = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        # GPU monitoring
        self.gpus = GPUtil.getGPUs() if GPUtil avail and len(GPUtil.getGPUs()) > 0 else []
        
        # Process monitoring
        self.process = psutil.Process()
        
        # Metrics history
        self.metrics_history = deque(maxlen=1000)
        
    def start_monitoring(self):
        """Start system monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_queue.put(metrics)
                self.metrics_history.append(metrics)
                
                # Check alerts
                if self.config.enable_alerts:
                    self._check_alerts(metrics)
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        metrics = {
            "timestamp": time.time(),
            "cpu": {},
            "memory": {},
            "gpu": {}
        }
        
        # CPU metrics
        if self.config.collect_cpu_metrics:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics["cpu"]["utilization"] = cpu_percent
            
            # CPU temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            metrics["cpu"]["temperature"] = entries[0].current
                            break
            except:
                pass
        
        # Memory metrics
        if self.config.collect_memory_metrics:
            memory = psutil.virtual_memory()
            metrics["memory"]["used"] = memory.used
            metrics["memory"]["total"] = memory.total
            metrics["memory"]["percent"] = memory.percent
            
            # Process memory
            process_memory = self.process.memory_info()
            metrics["memory"]["process_used"] = process_memory.rss
        
        # GPU metrics
        if self.config.collect_gpu_metrics and self.gpus:
            for i, gpu in enumerate(self.gpus):
                gpu_metrics = {
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_percent": gpu.memoryUtil * 100,
                    "utilization": gpu.load * 100,
                    "temperature": gpu.temperature,
                    "power_usage": gpu.powerDraw if hasattr(gpu, 'powerDraw') else None
                }
                metrics["gpu"][f"gpu_{i}"] = gpu_metrics
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        
        # GPU memory alerts
        if "gpu" in metrics:
            for gpu_name, gpu_metrics in metrics["gpu"].items():
                if gpu_metrics.get("memory_percent", 0) > self.config.alert_thresholds["gpu_memory_usage"] * 100:
                    alerts.append(f"High GPU memory usage on {gpu_name}: {gpu_metrics['memory_percent']:.1f}%")
                
                if gpu_metrics.get("utilization", 0) > self.config.alert_thresholds["gpu_utilization"] * 100:
                    alerts.append(f"High GPU utilization on {gpu_name}: {gpu_metrics['utilization']:.1f}%")
        
        # CPU memory alerts
        if "memory" in metrics:
            cpu_memory_percent = metrics["memory"].get("percent", 0)
            if cpu_memory_percent > self.config.alert_thresholds["cpu_memory_usage"] * 100:
                alerts.append(f"High CPU memory usage: {cpu_memory_percent:.1f}%")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest system metrics"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history"""
        return list(self.metrics_history)


class DistributedMetricsCollector:
    """Distributed metrics collection and aggregation"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Metrics storage
        self.local_metrics = []
        self.global_metrics = []
        
        # Synchronization
        self.sync_interval = 10  # steps
        self.step_counter = 0
    
    def collect_metrics(self, metrics: TrainingMetrics):
        """Collect metrics from training step"""
        # Add rank information
        metrics.rank = self.rank
        
        # Store locally
        self.local_metrics.append(metrics)
        
        # Synchronize periodically
        self.step_counter += 1
        if self.step_counter % self.sync_interval == 0:
            self._synchronize_metrics()
    
    def _synchronize_metrics(self):
        """Synchronize metrics across distributed processes"""
        if self.world_size == 1:
            self.global_metrics.extend(self.local_metrics)
            self.local_metrics.clear()
            return
        
        try:
            # Gather metrics from all processes
            local_metrics_data = [asdict(m) for m in self.local_metrics]
            
            # Serialize metrics
            metrics_json = json.dumps(local_metrics_data)
            metrics_bytes = metrics_json.encode('utf-8')
            
            # Get max size
            local_size = torch.tensor([len(metrics_bytes)], dtype=torch.int64)
            max_size = [torch.tensor([0], dtype=torch.int64) for _ in range(self.world_size)]
            dist.all_gather(max_size, local_size)
            max_size = max(max_size).item()
            
            # Pad and gather
            padded_bytes = metrics_bytes + b'\x00' * (max_size - len(metrics_bytes))
            padded_tensor = torch.frombuffer(padded_bytes, dtype=torch.uint8)
            
            gathered_tensors = [torch.empty_like(padded_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered_tensors, padded_tensor)
            
            # Deserialize and combine
            all_metrics = []
            for tensor in gathered_tensors:
                bytes_data = tensor.numpy().tobytes()
                json_data = bytes_data.decode('utf-8').rstrip('\x00')
                if json_data:
                    metrics_list = json.loads(json_data)
                    all_metrics.extend([TrainingMetrics(**m) for m in metrics_list])
            
            # Only rank 0 stores global metrics
            if self.rank == 0:
                self.global_metrics.extend(all_metrics)
            
            # Clear local metrics
            self.local_metrics.clear()
            
        except Exception as e:
            logger.error(f"Error synchronizing metrics: {e}")
    
    def get_global_metrics(self) -> List[TrainingMetrics]:
        """Get global metrics (only available on rank 0)"""
        if self.rank == 0:
            return self.global_metrics
        else:
            return []


class TrainingMonitor:
    """Main training monitor with distributed support"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.is_running = False
        
        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Components
        self.system_monitor = SystemMonitor(config)
        self.metrics_collector = DistributedMetricsCollector(config)
        
        # Logging
        self.tensorboard_writer = None
        self.wandb_run = None
        self.file_logger = None
        
        # Metrics storage
        self.training_metrics = []
        self.system_metrics = []
        
        # Visualization
        self.plot_data = defaultdict(list)
        
        # Initialize logging
        self._initialize_logging()
    
    def _initialize_logging(self):
        """Initialize logging components"""
        # Create directories
        if self.rank == 0:
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.plot_dir).mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.log_to_tensorboard and self.rank == 0:
            self.tensorboard_writer = SummaryWriter(
                log_dir=self.config.tensorboard_dir,
                flush_secs=self.config.tensorboard_flush_secs
            )
        
        # WandB
        if self.config.log_to_wandb and WANDB_AVAILABLE and self.rank == 0:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                config=asdict(self.config)
            )
            self.wandb_run = wandb
        
        # File logging
        if self.config.log_to_file and self.rank == 0:
            log_file = Path(self.config.log_dir) / self.config.log_file
            self.file_logger = logging.getLogger("training_monitor")
            self.file_logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)
    
    def start_monitoring(self):
        """Start monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        logger.info("Training monitor started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        # Close logging
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        # Save final metrics
        if self.rank == 0:
            self._save_metrics()
            self._create_final_plots()
        
        logger.info("Training monitor stopped")
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        if not self.is_running:
            return
        
        # Collect metrics
        self.metrics_collector.collect_metrics(metrics)
        
        # Only rank 0 logs to external systems
        if self.rank == 0:
            self.training_metrics.append(metrics)
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                self._log_to_tensorboard(metrics)
            
            # Log to WandB
            if self.wandb_run:
                self._log_to_wandb(metrics)
            
            # Log to file
            if self.file_logger:
                self._log_to_file(metrics)
            
            # Log to console
            if self.config.log_to_console:
                self._log_to_console(metrics)
            
            # Update plot data
            self._update_plot_data(metrics)
            
            # Create periodic plots
            if self.config.create_plots and len(self.training_metrics) % self.config.plot_interval == 0:
                self._create_plots()
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard"""
        step = metrics.step
        
        # Loss metrics
        self.tensorboard_writer.add_scalar("Loss/train", metrics.train_loss, step)
        if metrics.val_loss is not None:
            self.tensorboard_writer.add_scalar("Loss/val", metrics.val_loss, step)
        if metrics.test_loss is not None:
            self.tensorboard_writer.add_scalar("Loss/test", metrics.test_loss, step)
        
        # Accuracy metrics
        if metrics.train_accuracy is not None:
            self.tensorboard_writer.add_scalar("Accuracy/train", metrics.train_accuracy, step)
        if metrics.val_accuracy is not None:
            self.tensorboard_writer.add_scalar("Accuracy/val", metrics.val_accuracy, step)
        if metrics.test_accuracy is not None:
            self.tensorboard_writer.add_scalar("Accuracy/test", metrics.test_accuracy, step)
        
        # Learning rate
        self.tensorboard_writer.add_scalar("Learning_Rate", metrics.learning_rate, step)
        
        # Performance metrics
        if metrics.throughput is not None:
            self.tensorboard_writer.add_scalar("Performance/throughput", metrics.throughput, step)
        if metrics.latency is not None:
            self.tensorboard_writer.add_scalar("Performance/latency", metrics.latency, step)
        
        # Memory metrics
        if metrics.gpu_memory_used is not None:
            self.tensorboard_writer.add_scalar("Memory/GPU_Used", metrics.gpu_memory_used, step)
        if metrics.cpu_memory_used is not None:
            self.tensorboard_writer.add_scalar("Memory/CPU_Used", metrics.cpu_memory_used, step)
        
        # GPU metrics
        if metrics.gpu_utilization is not None:
            self.tensorboard_writer.add_scalar("GPU/Utilization", metrics.gpu_utilization, step)
        if metrics.gpu_temperature is not None:
            self.tensorboard_writer.add_scalar("GPU/Temperature", metrics.gpu_temperature, step)
        
        # CPU metrics
        if metrics.cpu_utilization is not None:
            self.tensorboard_writer.add_scalar("CPU/Utilization", metrics.cpu_utilization, step)
        
        # Gradient metrics
        if metrics.gradient_norm is not None:
            self.tensorboard_writer.add_scalar("Gradients/Norm", metrics.gradient_norm, step)
        if metrics.gradient_max is not None:
            self.tensorboard_writer.add_scalar("Gradients/Max", metrics.gradient_max, step)
        if metrics.gradient_min is not None:
            self.tensorboard_writer.add_scalar("Gradients/Min", metrics.gradient_min, step)
        
        # Custom metrics
        for name, value in metrics.custom_metrics.items():
            self.tensorboard_writer.add_scalar(f"Custom/{name}", value, step)
    
    def _log_to_wandb(self, metrics: TrainingMetrics):
        """Log metrics to WandB"""
        log_dict = {
            "step": metrics.step,
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
            "learning_rate": metrics.learning_rate
        }
        
        # Add optional metrics
        if metrics.val_loss is not None:
            log_dict["val_loss"] = metrics.val_loss
        if metrics.test_loss is not None:
            log_dict["test_loss"] = metrics.test_loss
        
        if metrics.train_accuracy is not None:
            log_dict["train_accuracy"] = metrics.train_accuracy
        if metrics.val_accuracy is not None:
            log_dict["val_accuracy"] = metrics.val_accuracy
        if metrics.test_accuracy is not None:
            log_dict["test_accuracy"] = metrics.test_accuracy
        
        if metrics.throughput is not None:
            log_dict["throughput"] = metrics.throughput
        if metrics.latency is not None:
            log_dict["latency"] = metrics.latency
        
        # System metrics
        if metrics.gpu_memory_used is not None:
            log_dict["gpu_memory_used"] = metrics.gpu_memory_used
        if metrics.cpu_memory_used is not None:
            log_dict["cpu_memory_used"] = metrics.cpu_memory_used
        if metrics.gpu_utilization is not None:
            log_dict["gpu_utilization"] = metrics.gpu_utilization
        if metrics.cpu_utilization is not None:
            log_dict["cpu_utilization"] = metrics.cpu_utilization
        
        # Gradient metrics
        if metrics.gradient_norm is not None:
            log_dict["gradient_norm"] = metrics.gradient_norm
        
        # Custom metrics
        log_dict.update(metrics.custom_metrics)
        
        self.wandb_run.log(log_dict)
    
    def _log_to_file(self, metrics: TrainingMetrics):
        """Log metrics to file"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(metrics.timestamp).isoformat(),
            "step": metrics.step,
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
            "learning_rate": metrics.learning_rate
        }
        
        # Add optional metrics
        if metrics.val_loss is not None:
            log_entry["val_loss"] = metrics.val_loss
        if metrics.train_accuracy is not None:
            log_entry["train_accuracy"] = metrics.train_accuracy
        if metrics.throughput is not None:
            log_entry["throughput"] = metrics.throughput
        
        self.file_logger.info(json.dumps(log_entry))
    
    def _log_to_console(self, metrics: TrainingMetrics):
        """Log metrics to console"""
        log_msg = (f"Step {metrics.step:6d} | "
                  f"Epoch {metrics.epoch:3d} | "
                  f"Loss: {metrics.train_loss:8.4f}")
        
        if metrics.val_loss is not None:
            log_msg += f" | Val Loss: {metrics.val_loss:8.4f}"
        
        if metrics.train_accuracy is not None:
            log_msg += f" | Acc: {metrics.train_accuracy:6.2f}%"
        
        if metrics.learning_rate is not None:
            log_msg += f" | LR: {metrics.learning_rate:8.2e}"
        
        if metrics.throughput is not None:
            log_msg += f" | Throughput: {metrics.throughput:6.1f}"
        
        print(log_msg)
    
    def _update_plot_data(self, metrics: TrainingMetrics):
        """Update plot data"""
        self.plot_data["steps"].append(metrics.step)
        self.plot_data["train_loss"].append(metrics.train_loss)
        
        if metrics.val_loss is not None:
            self.plot_data["val_loss"].append(metrics.val_loss)
        
        if metrics.train_accuracy is not None:
            self.plot_data["train_accuracy"].append(metrics.train_accuracy)
        
        if metrics.val_accuracy is not None:
            self.plot_data["val_accuracy"].append(metrics.val_accuracy)
        
        if metrics.learning_rate is not None:
            self.plot_data["learning_rate"].append(metrics.learning_rate)
        
        if metrics.gpu_utilization is not None:
            self.plot_data["gpu_utilization"].append(metrics.gpu_utilization)
        
        if metrics.gradient_norm is not None:
            self.plot_data["gradient_norm"].append(metrics.gradient_norm)
    
    def _create_plots(self):
        """Create monitoring plots"""
        if not self.config.create_plots or self.rank != 0:
            return
        
        # Create loss plot
        if len(self.plot_data["train_loss"]) > 1:
            plt.figure(figsize=(12, 8))
            
            # Loss plot
            plt.subplot(2, 2, 1)
            plt.plot(self.plot_data["steps"], self.plot_data["train_loss"], label="Train Loss")
            if self.plot_data["val_loss"]:
                plt.plot(self.plot_data["steps"], self.plot_data["val_loss"], label="Val Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Accuracy plot
            if self.plot_data["train_accuracy"]:
                plt.subplot(2, 2, 2)
                plt.plot(self.plot_data["steps"], self.plot_data["train_accuracy"], label="Train Acc")
                if self.plot_data["val_accuracy"]:
                    plt.plot(self.plot_data["steps"], self.plot_data["val_accuracy"], label="Val Acc")
                plt.xlabel("Step")
                plt.ylabel("Accuracy (%)")
                plt.title("Training Accuracy")
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Learning rate plot
            if self.plot_data["learning_rate"]:
                plt.subplot(2, 2, 3)
                plt.plot(self.plot_data["steps"], self.plot_data["learning_rate"])
                plt.xlabel("Step")
                plt.ylabel("Learning Rate")
                plt.title("Learning Rate Schedule")
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
            
            # GPU utilization plot
            if self.plot_data["gpu_utilization"]:
                plt.subplot(2, 2, 4)
                plt.plot(self.plot_data["steps"], self.plot_data["gpu_utilization"])
                plt.xlabel("Step")
                plt.ylabel("GPU Utilization (%)")
                plt.title("GPU Utilization")
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = Path(self.config.plot_dir) / f"training_plot_step_{self.plot_data['steps'][-1]}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def _create_final_plots(self):
        """Create final comprehensive plots"""
        if not self.config.create_plots or self.rank != 0:
            return
        
        # Create comprehensive report
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss evolution
        axes[0, 0].plot(self.plot_data["steps"], self.plot_data["train_loss"], label="Train")
        if self.plot_data["val_loss"]:
            axes[0, 0].plot(self.plot_data["steps"], self.plot_data["val_loss"], label="Validation")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss Evolution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if self.plot_data["learning_rate"]:
            axes[0, 1].plot(self.plot_data["steps"], self.plot_data["learning_rate"])
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].set_title("Learning Rate Schedule")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        
        # GPU utilization
        if self.plot_data["gpu_utilization"]:
            axes[0, 2].plot(self.plot_data["steps"], self.plot_data["gpu_utilization"])
            axes[0, 2].set_xlabel("Step")
            axes[0, 2].set_ylabel("GPU Utilization (%)")
            axes[0, 2].set_title("GPU Utilization")
            axes[0, 2].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.plot_data["gradient_norm"]:
            axes[1, 0].plot(self.plot_data["steps"], self.plot_data["gradient_norm"])
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Gradient Norm")
            axes[1, 0].set_title("Gradient Norms")
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training speed
        if self.plot_data["steps"] and len(self.plot_data["steps"]) > 1:
            steps_per_hour = []
            timestamps = []
            for i in range(1, len(self.plot_data["steps"])):
                steps_per_hour.append(3600 / (self.plot_data["steps"][i] - self.plot_data["steps"][i-1]))
                timestamps.append(self.plot_data["steps"][i])
            
            axes[1, 1].plot(timestamps, steps_per_hour)
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Steps per Hour")
            axes[1, 1].set_title("Training Speed")
            axes[1, 1].grid(True, alpha=0.3)
        
        # Memory usage
        if self.plot_data["gpu_utilization"]:
            axes[1, 2].plot(self.plot_data["steps"], self.plot_data["gpu_utilization"], label="GPU Util")
            axes[1, 2].set_xlabel("Step")
            axes[1, 2].set_ylabel("Utilization (%)")
            axes[1, 2].set_title("Resource Usage")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        final_plot_path = Path(self.config.plot_dir) / "final_training_report.png"
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Final training report saved to {final_plot_path}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        if self.rank != 0:
            return
        
        # Save training metrics
        metrics_file = Path(self.config.log_dir) / self.config.metrics_file
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.training_metrics], f, indent=2)
        
        # Save system metrics
        system_metrics_file = Path(self.config.log_dir) / "system_metrics.json"
        system_metrics_list = list(self.system_monitor.get_metrics_history())
        with open(system_metrics_file, 'w') as f:
            json.dump(system_metrics_list, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if self.rank != 0 or not self.training_metrics:
            return {}
        
        # Calculate summary statistics
        train_losses = [m.train_loss for m in self.training_metrics]
        val_losses = [m.val_loss for m in self.training_metrics if m.val_loss is not None]
        train_accuracies = [m.train_accuracy for m in self.training_metrics if m.train_accuracy is not None]
        val_accuracies = [m.val_accuracy for m in self.training_metrics if m.val_accuracy is not None]
        
        summary = {
            "total_steps": len(self.training_metrics),
            "final_step": self.training_metrics[-1].step,
            "final_epoch": self.training_metrics[-1].epoch,
            "best_train_loss": min(train_losses),
            "final_train_loss": train_losses[-1],
            "best_val_loss": min(val_losses) if val_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_train_accuracy": max(train_accuracies) if train_accuracies else None,
            "final_train_accuracy": train_accuracies[-1] if train_accuracies else None,
            "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
            "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
            "total_training_time": self.training_metrics[-1].timestamp - self.training_metrics[0].timestamp,
            "average_step_time": (self.training_metrics[-1].timestamp - self.training_metrics[0].timestamp) / len(self.training_metrics)
        }
        
        return summary


def create_monitoring_config(**kwargs) -> MonitoringConfig:
    """Create monitoring configuration"""
    return MonitoringConfig(**kwargs)


def main():
    """Main function for testing monitoring system"""
    
    # Create monitoring configuration
    config = create_monitoring_config(
        enable_monitoring=True,
        monitor_interval=0.5,
        log_to_tensorboard=True,
        log_to_console=True,
        create_plots=True
    )
    
    # Create monitor
    monitor = TrainingMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate training metrics
    import random
    for step in range(100):
        metrics = TrainingMetrics(
            timestamp=time.time(),
            step=step,
            epoch=step // 10,
            train_loss=2.0 * (1 - step / 100) + 0.1 * random.random(),
            val_loss=1.8 * (1 - step / 100) + 0.1 * random.random(),
            train_accuracy=80 + 15 * (step / 100) + 2 * random.random(),
            val_accuracy=75 + 20 * (step / 100) + 2 * random.random(),
            learning_rate=1e-3 * (1 - step / 100),
            throughput=100 + 50 * random.random(),
            gpu_utilization=80 + 10 * random.random(),
            gradient_norm=1.0 + 0.5 * random.random()
        )
        
        monitor.log_training_metrics(metrics)
        time.sleep(0.1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get summary
    summary = monitor.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()