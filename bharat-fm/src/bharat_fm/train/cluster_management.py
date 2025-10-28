"""
Cluster Management Integration for Distributed Training
Kubernetes and Slurm integration for Bharat-FM distributed training
"""

import os
import json
import yaml
import subprocess
import socket
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import kubernetes
from kubernetes import client, config as k8s_config
from kubernetes.client.rest import ApiException
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

@dataclass
class ClusterConfig:
    """Configuration for cluster management"""
    cluster_type: str = "local"  # local, kubernetes, slurm
    cluster_name: str = "bharat-fm-cluster"
    
    # Kubernetes settings
    k8s_namespace: str = "bharat-fm"
    k8s_image: str = "bharat-fm/training:latest"
    k8s_service_account: str = "bharat-fm-sa"
    k8s_node_selector: Dict[str, str] = None
    
    # Slurm settings
    slurm_partition: str = "gpu"
    slurm_account: str = "bharat-fm"
    slurm_qos: str = "normal"
    slurm_time_limit: str = "24:00:00"
    
    # Resource settings
    num_nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_node: int = 8
    memory_per_node: str = "32Gi"
    
    # Network settings
    network_plugin: str = "nccl"
    rdma_enabled: bool = False
    
    # Storage settings
    storage_class: str = "standard"
    storage_size: str = "100Gi"
    shared_storage: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_collection: bool = True
    
    def __post_init__(self):
        if self.k8s_node_selector is None:
            self.k8s_node_selector = {"accelerator": "nvidia-tesla"}


class KubernetesManager:
    """Kubernetes cluster management for distributed training"""
    
    def __init__(self, cluster_config: ClusterConfig):
        self.config = cluster_config
        self.api_client = None
        self.batch_api = None
        self.core_api = None
        
        # Initialize Kubernetes client
        self._init_kubernetes_client()
    
    def _init_kubernetes_client(self):
        """Initialize Kubernetes client"""
        try:
            # Try to load in-cluster config first
            k8s_config.load_incluster_config()
        except:
            # Fall back to kubeconfig
            try:
                k8s_config.load_kube_config()
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise
        
        self.api_client = k8s_config.new_client_from_config()
        self.batch_api = client.BatchV1Api(self.api_client)
        self.core_api = client.CoreV1Api(self.api_client)
        
        logger.info("Kubernetes client initialized successfully")
    
    def create_namespace(self) -> bool:
        """Create Kubernetes namespace"""
        try:
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=self.config.k8s_namespace)
            )
            
            self.core_api.create_namespace(namespace)
            logger.info(f"Created namespace: {self.config.k8s_namespace}")
            return True
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"Namespace {self.config.k8s_namespace} already exists")
                return True
            else:
                logger.error(f"Failed to create namespace: {e}")
                return False
    
    def create_service_account(self) -> bool:
        """Create service account for training jobs"""
        try:
            # Service account
            sa = client.V1ServiceAccount(
                metadata=client.V1ObjectMeta(
                    name=self.config.k8s_service_account,
                    namespace=self.config.k8s_namespace
                )
            )
            
            self.core_api.create_namespaced_service_account(
                namespace=self.config.k8s_namespace,
                body=sa
            )
            
            # RBAC role binding
            role_binding = client.V1RoleBinding(
                metadata=client.V1ObjectMeta(
                    name=f"{self.config.k8s_service_account}-binding",
                    namespace=self.config.k8s_namespace
                ),
                subjects=[client.V1Subject(
                    kind="ServiceAccount",
                    name=self.config.k8s_service_account,
                    namespace=self.config.k8s_namespace
                )],
                role_ref=client.V1RoleRef(
                    kind="ClusterRole",
                    name="edit",
                    api_group="rbac.authorization.k8s.io"
                )
            )
            
            self.core_api.create_namespaced_role_binding(
                namespace=self.config.k8s_namespace,
                body=role_binding
            )
            
            logger.info(f"Created service account: {self.config.k8s_service_account}")
            return True
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"Service account {self.config.k8s_service_account} already exists")
                return True
            else:
                logger.error(f"Failed to create service account: {e}")
                return False
    
    def create_pvc(self, name: str, size: str = None) -> bool:
        """Create Persistent Volume Claim for shared storage"""
        size = size or self.config.storage_size
        
        try:
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(
                    name=name,
                    namespace=self.config.k8s_namespace
                ),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=["ReadWriteMany"],
                    resources=client.V1ResourceRequirements(
                        requests={"storage": size}
                    ),
                    storage_class_name=self.config.storage_class
                )
            )
            
            self.core_api.create_namespaced_persistent_volume_claim(
                namespace=self.config.k8s_namespace,
                body=pvc
            )
            
            logger.info(f"Created PVC: {name} with size {size}")
            return True
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"PVC {name} already exists")
                return True
            else:
                logger.error(f"Failed to create PVC: {e}")
                return False
    
    def create_training_job(self, 
                           job_name: str,
                           command: List[str],
                           args: List[str] = None,
                           env_vars: Dict[str, str] = None,
                           volumes: List[Dict] = None) -> Optional[str]:
        """Create distributed training job"""
        
        # Environment variables
        env_list = []
        if env_vars:
            for key, value in env_vars.items():
                env_list.append(client.V1EnvVar(name=key, value=value))
        
        # Add distributed training environment variables
        env_list.extend([
            client.V1EnvVar(name="WORLD_SIZE", value=str(self.config.num_nodes * self.config.gpus_per_node)),
            client.V1EnvVar(name="MASTER_ADDR", value=f"{job_name}-master-0.{job_name}.{self.config.k8s_namespace}.svc.cluster.local"),
            client.V1EnvVar(name="MASTER_PORT", value="12355"),
            client.V1EnvVar(name="NCCL_DEBUG", value="INFO"),
            client.V1EnvVar(name="NCCL_SOCKET_IFNAME", value="eth0"),
        ])
        
        # Volume mounts
        volume_mounts = []
        if volumes:
            for volume in volumes:
                volume_mounts.append(client.V1VolumeMount(
                    name=volume["name"],
                    mount_path=volume["mount_path"]
                ))
        
        # Resource requirements
        resources = client.V1ResourceRequirements(
            limits={
                "nvidia.com/gpu": str(self.config.gpus_per_node),
                "memory": self.config.memory_per_node,
                "cpu": str(self.config.cpus_per_node)
            },
            requests={
                "memory": self.config.memory_per_node,
                "cpu": str(self.config.cpus_per_node)
            }
        )
        
        # Container spec
        container = client.V1Container(
            name="trainer",
            image=self.config.k8s_image,
            command=command,
            args=args,
            env=env_list,
            volume_mounts=volume_mounts,
            resources=resources,
            security_context=client.V1SecurityContext(
                privileged=True,
                run_as_user=0
            )
        )
        
        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            node_selector=self.config.k8s_node_selector,
            service_account_name=self.config.k8s_service_account
        )
        
        # Add volumes if specified
        if volumes:
            pod_spec.volumes = []
            for volume in volumes:
                pod_spec.volumes.append(client.V1Volume(
                    name=volume["name"],
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=volume["pvc_name"]
                    )
                ))
        
        # Job spec
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": job_name, "component": "trainer"}
                ),
                spec=pod_spec
            ),
            backoff_limit=1,
            parallelism=self.config.num_nodes,
            completions=self.config.num_nodes
        )
        
        # Job
        job = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.config.k8s_namespace,
                labels={"app": job_name, "component": "training"}
            ),
            spec=job_spec
        )
        
        try:
            created_job = self.batch_api.create_namespaced_job(
                namespace=self.config.k8s_namespace,
                body=job
            )
            
            logger.info(f"Created training job: {job_name}")
            return created_job.metadata.name
            
        except ApiException as e:
            logger.error(f"Failed to create training job: {e}")
            return None
    
    def create_tensorboard_service(self, job_name: str) -> Optional[str]:
        """Create TensorBoard service for monitoring"""
        
        service = client.V1Service(
            metadata=client.V1ObjectMeta(
                name=f"{job_name}-tensorboard",
                namespace=self.config.k8s_namespace,
                labels={"app": job_name, "component": "tensorboard"}
            ),
            spec=client.V1ServiceSpec(
                selector={"app": job_name, "component": "tensorboard"},
                ports=[client.V1ServicePort(
                    port=6006,
                    target_port=6006,
                    protocol="TCP"
                )],
                type="LoadBalancer"
            )
        )
        
        try:
            created_service = self.core_api.create_namespaced_service(
                namespace=self.config.k8s_namespace,
                body=service
            )
            
            logger.info(f"Created TensorBoard service: {job_name}-tensorboard")
            return created_service.metadata.name
            
        except ApiException as e:
            logger.error(f"Failed to create TensorBoard service: {e}")
            return None
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get job status and pod information"""
        try:
            # Get job status
            job = self.batch_api.read_namespaced_job(
                name=job_name,
                namespace=self.config.k8s_namespace
            )
            
            # Get pods
            pods = self.core_api.list_namespaced_pod(
                namespace=self.config.k8s_namespace,
                label_selector=f"app={job_name}"
            )
            
            pod_status = {}
            for pod in pods.items:
                pod_status[pod.metadata.name] = {
                    "phase": pod.status.phase,
                    "node": pod.spec.node_name,
                    "start_time": pod.status.start_time.isoformat() if pod.status.start_time else None,
                    "conditions": [
                        {
                            "type": cond.type,
                            "status": cond.status,
                            "message": cond.message
                        }
                        for cond in pod.status.conditions or []
                    ]
                }
            
            return {
                "job_name": job_name,
                "status": job.status,
                "pod_status": pod_status,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0
            }
            
        except ApiException as e:
            logger.error(f"Failed to get job status: {e}")
            return {}
    
    def delete_job(self, job_name: str) -> bool:
        """Delete training job"""
        try:
            # Delete job
            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.config.k8s_namespace,
                body=client.V1DeleteOptions()
            )
            
            # Delete associated pods
            self.core_api.delete_collection_namespaced_pod(
                namespace=self.config.k8s_namespace,
                label_selector=f"app={job_name}"
            )
            
            logger.info(f"Deleted job: {job_name}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to delete job: {e}")
            return False
    
    def get_job_logs(self, job_name: str, pod_name: str = None) -> str:
        """Get job logs"""
        try:
            if pod_name:
                logs = self.core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.config.k8s_namespace
                )
            else:
                # Get logs from all pods
                pods = self.core_api.list_namespaced_pod(
                    namespace=self.config.k8s_namespace,
                    label_selector=f"app={job_name}"
                )
                
                logs = ""
                for pod in pods.items:
                    pod_logs = self.core_api.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=self.config.k8s_namespace
                    )
                    logs += f"=== Pod: {pod.metadata.name} ===\n{pod_logs}\n\n"
            
            return logs
            
        except ApiException as e:
            logger.error(f"Failed to get job logs: {e}")
            return ""


class SlurmManager:
    """Slurm cluster management for distributed training"""
    
    def __init__(self, cluster_config: ClusterConfig):
        self.config = cluster_config
        self.job_id = None
    
    def submit_job(self, 
                  script_path: str,
                  job_name: str = None,
                  env_vars: Dict[str, str] = None) -> Optional[str]:
        """Submit Slurm job"""
        
        job_name = job_name or f"bharat-fm-{int(time.time())}"
        
        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--partition={self.config.slurm_partition}",
            f"--account={self.config.slurm_account}",
            f"--qos={self.config.slurm_qos}",
            f"--time={self.config.slurm_time_limit}",
            f"--nodes={self.config.num_nodes}",
            f"--ntasks-per-node={self.config.gpus_per_node}",
            f"--cpus-per-task={self.config.cpus_per_node}",
            f"--mem={self.config.memory_per_node}",
            f"--gres=gpu:{self.config.gpus_per_node}",
            "--output=%j.out",
            "--error=%j.err"
        ]
        
        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                sbatch_cmd.append(f"--export={key}={value}")
        
        # Add script path
        sbatch_cmd.append(script_path)
        
        try:
            # Submit job
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract job ID from output
                job_id = result.stdout.strip().split()[-1]
                self.job_id = job_id
                
                logger.info(f"Submitted Slurm job: {job_name} (ID: {job_id})")
                return job_id
            else:
                logger.error(f"Failed to submit Slurm job: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting Slurm job: {e}")
            return None
    
    def get_job_status(self, job_id: str = None) -> Dict[str, Any]:
        """Get Slurm job status"""
        job_id = job_id or self.job_id
        
        if not job_id:
            return {}
        
        try:
            # Get job status using scontrol
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse job info
                job_info = {}
                for line in result.stdout.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        job_info[key.strip()] = value.strip()
                
                return {
                    "job_id": job_id,
                    "status": job_info.get("JobState", "UNKNOWN"),
                    "nodes": job_info.get("NodeList", ""),
                    "start_time": job_info.get("StartTime", ""),
                    "end_time": job_info.get("EndTime", ""),
                    "runtime": job_info.get("RunTime", ""),
                    "partition": job_info.get("Partition", ""),
                    "account": job_info.get("Account", "")
                }
            else:
                logger.error(f"Failed to get job status: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {}
    
    def cancel_job(self, job_id: str = None) -> bool:
        """Cancel Slurm job"""
        job_id = job_id or self.job_id
        
        if not job_id:
            return False
        
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Cancelled Slurm job: {job_id}")
                return True
            else:
                logger.error(f"Failed to cancel job: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False
    
    def get_job_logs(self, job_id: str = None) -> Dict[str, str]:
        """Get Slurm job logs"""
        job_id = job_id or self.job_id
        
        if not job_id:
            return {}
        
        try:
            # Get output and error files
            output_file = f"{job_id}.out"
            error_file = f"{job_id}.err"
            
            logs = {}
            
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    logs["output"] = f.read()
            
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    logs["error"] = f.read()
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting job logs: {e}")
            return {}


class ClusterManager:
    """Unified cluster management interface"""
    
    def __init__(self, cluster_config: ClusterConfig):
        self.config = cluster_config
        
        # Initialize appropriate cluster manager
        if cluster_config.cluster_type == "kubernetes":
            self.manager = KubernetesManager(cluster_config)
        elif cluster_config.cluster_type == "slurm":
            self.manager = SlurmManager(cluster_config)
        else:
            self.manager = None
    
    def setup_cluster(self) -> bool:
        """Setup cluster environment"""
        if self.config.cluster_type == "kubernetes":
            return self._setup_kubernetes_cluster()
        elif self.config.cluster_type == "slurm":
            return self._setup_slurm_cluster()
        else:
            logger.info("Local cluster setup - no additional setup required")
            return True
    
    def _setup_kubernetes_cluster(self) -> bool:
        """Setup Kubernetes cluster"""
        k8s_manager = self.manager
        
        # Create namespace
        if not k8s_manager.create_namespace():
            return False
        
        # Create service account
        if not k8s_manager.create_service_account():
            return False
        
        # Create PVC for shared storage
        if self.config.shared_storage:
            if not k8s_manager.create_pvc("training-storage"):
                return False
        
        logger.info("Kubernetes cluster setup complete")
        return True
    
    def _setup_slurm_cluster(self) -> bool:
        """Setup Slurm cluster"""
        # Slurm setup is typically done by cluster administrators
        # Here we just verify that Slurm commands are available
        
        try:
            result = subprocess.run(["sinfo"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Slurm cluster is accessible")
                return True
            else:
                logger.error("Slurm cluster is not accessible")
                return False
        except Exception as e:
            logger.error(f"Error accessing Slurm cluster: {e}")
            return False
    
    def submit_training_job(self, 
                           job_name: str,
                           script_path: str = None,
                           command: List[str] = None,
                           args: List[str] = None,
                           env_vars: Dict[str, str] = None) -> Optional[str]:
        """Submit training job to cluster"""
        
        if self.config.cluster_type == "kubernetes":
            return self.manager.create_training_job(
                job_name=job_name,
                command=command,
                args=args,
                env_vars=env_vars,
                volumes=[{
                    "name": "training-storage",
                    "mount_path": "/shared",
                    "pvc_name": "training-storage"
                }] if self.config.shared_storage else []
            )
        elif self.config.cluster_type == "slurm":
            return self.manager.submit_job(
                script_path=script_path,
                job_name=job_name,
                env_vars=env_vars
            )
        else:
            logger.error("Unsupported cluster type")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        return self.manager.get_job_status(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel job"""
        return self.manager.cancel_job(job_id)
    
    def get_job_logs(self, job_id: str) -> Union[str, Dict[str, str]]:
        """Get job logs"""
        return self.manager.get_job_logs(job_id)


def create_cluster_config(cluster_type: str = "local", **kwargs) -> ClusterConfig:
    """Create cluster configuration"""
    config_dict = {
        "cluster_type": cluster_type,
        **kwargs
    }
    
    return ClusterConfig(**config_dict)


def main():
    """Main function for cluster management testing"""
    
    # Test Kubernetes cluster
    k8s_config = create_cluster_config(
        cluster_type="kubernetes",
        num_nodes=2,
        gpus_per_node=2,
        k8s_namespace="bharat-fm-test"
    )
    
    k8s_manager = ClusterManager(k8s_config)
    
    if k8s_manager.setup_cluster():
        logger.info("Kubernetes cluster setup successful")
        
        # Submit test job
        job_id = k8s_manager.submit_training_job(
            job_name="test-job",
            command=["python", "train.py"],
            args=["--config", "config.json"],
            env_vars={"WORLD_SIZE": "4"}
        )
        
        if job_id:
            logger.info(f"Submitted job: {job_id}")
            
            # Monitor job status
            time.sleep(5)
            status = k8s_manager.get_job_status(job_id)
            logger.info(f"Job status: {status}")
    
    # Test Slurm cluster
    slurm_config = create_cluster_config(
        cluster_type="slurm",
        num_nodes=2,
        gpus_per_node=2,
        slurm_partition="gpu"
    )
    
    slurm_manager = ClusterManager(slurm_config)
    
    if slurm_manager.setup_cluster():
        logger.info("Slurm cluster setup successful")


if __name__ == "__main__":
    main()