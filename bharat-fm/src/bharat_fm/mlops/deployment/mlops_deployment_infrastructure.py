"""
MLOps Deployment Infrastructure for Bharat-FM

This module provides comprehensive deployment infrastructure support for machine learning models
across multiple platforms including Kubernetes, Docker, and Serverless environments. It includes
blue-green deployments, canary deployments, and infrastructure as code capabilities.

Author: Bharat-FM Team
Version: 1.0.0
"""

import os
import sys
import json
import yaml
import time
import logging
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import docker
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.web import WebSiteManagementClient
import google.cloud.functions_v1
import google.cloud.run_v2
import requests
from pydantic import BaseModel, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeploymentPlatform(Enum):
    """Supported deployment platforms"""
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    AWS_LAMBDA = "aws_lambda"
    AZURE_FUNCTIONS = "azure_functions"
    GCP_FUNCTIONS = "gcp_functions"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    AZURE_CONTAINER_INSTANCES = "azure_container_instances"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    model_version: str
    platform: DeploymentPlatform
    strategy: DeploymentStrategy
    image_name: str
    image_tag: str
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_request: Optional[str] = None
    gpu_limit: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    health_check_port: int = 8000
    canary_percentage: int = 10
    blue_green_switch_delay: int = 300  # 5 minutes
    auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80


@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    deployment_id: str
    status: str  # pending, running, failed, completed
    platform: DeploymentPlatform
    strategy: DeploymentStrategy
    created_at: datetime
    updated_at: datetime
    replicas: int
    available_replicas: int
    url: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class KubernetesDeployer:
    """Kubernetes deployment manager"""
    
    def __init__(self, kube_config_path: Optional[str] = None):
        """Initialize Kubernetes client"""
        try:
            if kube_config_path:
                config.load_kube_config(config_file=kube_config_path)
            else:
                config.load_kube_config()
            
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.autoscaling_v1 = client.AutoscalingV1Api()
            self.networking_v1 = client.NetworkingV1Api()
            logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def create_deployment(self, config: DeploymentConfig) -> DeploymentStatus:
        """Create Kubernetes deployment"""
        try:
            deployment_id = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            # Create deployment manifest
            deployment_manifest = self._create_deployment_manifest(config, deployment_id)
            
            # Apply deployment
            self.apps_v1.create_namespaced_deployment(
                namespace="default",
                body=deployment_manifest
            )
            
            # Create service if needed
            service_manifest = self._create_service_manifest(config, deployment_id)
            self.core_v1.create_namespaced_service(
                namespace="default",
                body=service_manifest
            )
            
            # Create HPA if auto-scaling enabled
            if config.auto_scaling:
                hpa_manifest = self._create_hpa_manifest(config, deployment_id)
                self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                    namespace="default",
                    body=hpa_manifest
                )
            
            # Wait for deployment to be ready
            status = self._wait_for_deployment_ready(deployment_id, config)
            
            logger.info(f"Kubernetes deployment created successfully: {deployment_id}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes deployment: {e}")
            return DeploymentStatus(
                deployment_id=deployment_id,
                status="failed",
                platform=DeploymentPlatform.KUBERNETES,
                strategy=config.strategy,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                replicas=0,
                available_replicas=0,
                error_message=str(e)
            )
    
    def _create_deployment_manifest(self, config: DeploymentConfig, deployment_id: str) -> Dict:
        """Create Kubernetes deployment manifest"""
        containers = [{
            "name": config.model_name,
            "image": f"{config.image_name}:{config.image_tag}",
            "ports": [{"containerPort": config.health_check_port}],
            "resources": {
                "requests": {
                    "cpu": config.cpu_request,
                    "memory": config.memory_request
                },
                "limits": {
                    "cpu": config.cpu_limit,
                    "memory": config.memory_limit
                }
            },
            "env": [{"name": k, "value": v} for k, v in config.environment.items()],
            "livenessProbe": {
                "httpGet": {
                    "path": config.health_check_path,
                    "port": config.health_check_port
                },
                "initialDelaySeconds": 30,
                "periodSeconds": 10
            },
            "readinessProbe": {
                "httpGet": {
                    "path": config.health_check_path,
                    "port": config.health_check_port
                },
                "initialDelaySeconds": 5,
                "periodSeconds": 5
            }
        }]
        
        # Add GPU resources if specified
        if config.gpu_request:
            containers[0]["resources"]["requests"]["nvidia.com/gpu"] = config.gpu_request
            containers[0]["resources"]["limits"]["nvidia.com/gpu"] = config.gpu_limit
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_id,
                "labels": {
                    "app": config.model_name,
                    "version": config.model_version,
                    "deployment-id": deployment_id
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.model_name,
                        "deployment-id": deployment_id
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.model_name,
                            "version": config.model_version,
                            "deployment-id": deployment_id
                        }
                    },
                    "spec": {
                        "containers": containers
                    }
                }
            }
        }
    
    def _create_service_manifest(self, config: DeploymentConfig, deployment_id: str) -> Dict:
        """Create Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{deployment_id}-service",
                "labels": {
                    "app": config.model_name,
                    "deployment-id": deployment_id
                }
            },
            "spec": {
                "selector": {
                    "app": config.model_name,
                    "deployment-id": deployment_id
                },
                "ports": [{
                    "port": 80,
                    "targetPort": config.health_check_port,
                    "protocol": "TCP"
                }],
                "type": "LoadBalancer"
            }
        }
    
    def _create_hpa_manifest(self, config: DeploymentConfig, deployment_id: str) -> Dict:
        """Create Kubernetes HPA manifest"""
        return {
            "apiVersion": "autoscaling/v1",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{deployment_id}-hpa",
                "labels": {
                    "app": config.model_name,
                    "deployment-id": deployment_id
                }
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": deployment_id
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "targetCPUUtilizationPercentage": config.target_cpu_utilization
            }
        }
    
    def _wait_for_deployment_ready(self, deployment_id: str, config: DeploymentConfig, timeout: int = 300) -> DeploymentStatus:
        """Wait for deployment to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_id,
                    namespace="default"
                )
                
                replicas = deployment.spec.replicas or 0
                available_replicas = deployment.status.available_replicas or 0
                
                if available_replicas == replicas:
                    # Get service URL
                    service = self.core_v1.read_namespaced_service(
                        name=f"{deployment_id}-service",
                        namespace="default"
                    )
                    
                    url = None
                    if service.status.load_balancer.ingress:
                        url = f"http://{service.status.load_balancer.ingress[0].ip or service.status.load_balancer.ingress[0].hostname}"
                    
                    return DeploymentStatus(
                        deployment_id=deployment_id,
                        status="running",
                        platform=DeploymentPlatform.KUBERNETES,
                        strategy=config.strategy,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        replicas=replicas,
                        available_replicas=available_replicas,
                        url=url
                    )
                
                time.sleep(10)
                
            except Exception as e:
                logger.warning(f"Error checking deployment status: {e}")
                time.sleep(10)
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="failed",
            platform=DeploymentPlatform.KUBERNETES,
            strategy=config.strategy,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            replicas=0,
            available_replicas=0,
            error_message="Timeout waiting for deployment to be ready"
        )
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete Kubernetes deployment"""
        try:
            # Delete HPA
            try:
                self.autoscaling_v1.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"{deployment_id}-hpa",
                    namespace="default"
                )
            except ApiException as e:
                if e.status != 404:
                    logger.warning(f"Failed to delete HPA: {e}")
            
            # Delete service
            try:
                self.core_v1.delete_namespaced_service(
                    name=f"{deployment_id}-service",
                    namespace="default"
                )
            except ApiException as e:
                if e.status != 404:
                    logger.warning(f"Failed to delete service: {e}")
            
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_id,
                namespace="default"
            )
            
            logger.info(f"Kubernetes deployment deleted successfully: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete Kubernetes deployment: {e}")
            return False


class DockerDeployer:
    """Docker deployment manager"""
    
    def __init__(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def create_deployment(self, config: DeploymentConfig) -> DeploymentStatus:
        """Create Docker container deployment"""
        try:
            deployment_id = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            # Pull image
            image = self.client.images.pull(f"{config.image_name}:{config.image_tag}")
            
            # Create containers
            containers = []
            for i in range(config.replicas):
                container = self.client.containers.run(
                    image=f"{config.image_name}:{config.image_tag}",
                    name=f"{deployment_id}-{i}",
                    environment=config.environment,
                    ports={f"{config.health_check_port}/tcp": None},
                    detach=True,
                    mem_limit=config.memory_limit,
                    nano_cpus=int(float(config.cpu_limit.replace("m", "")) * 1e6)
                )
                containers.append(container)
            
            # Wait for containers to be ready
            status = self._wait_for_containers_ready(deployment_id, containers, config)
            
            logger.info(f"Docker deployment created successfully: {deployment_id}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to create Docker deployment: {e}")
            return DeploymentStatus(
                deployment_id=deployment_id,
                status="failed",
                platform=DeploymentPlatform.DOCKER,
                strategy=config.strategy,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                replicas=0,
                available_replicas=0,
                error_message=str(e)
            )
    
    def _wait_for_containers_ready(self, deployment_id: str, containers: List, config: DeploymentConfig, timeout: int = 300) -> DeploymentStatus:
        """Wait for containers to be ready"""
        start_time = time.time()
        ready_containers = 0
        
        while time.time() - start_time < timeout:
            ready_containers = 0
            
            for container in containers:
                try:
                    container.reload()
                    if container.status == "running":
                        # Check health endpoint
                        try:
                            container_ip = container.attrs["NetworkSettings"]["IPAddress"]
                            if container_ip:
                                response = requests.get(
                                    f"http://{container_ip}:{config.health_check_port}{config.health_check_path}",
                                    timeout=5
                                )
                                if response.status_code == 200:
                                    ready_containers += 1
                        except:
                            pass
                except:
                    pass
            
            if ready_containers == len(containers):
                return DeploymentStatus(
                    deployment_id=deployment_id,
                    status="running",
                    platform=DeploymentPlatform.DOCKER,
                    strategy=config.strategy,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    replicas=len(containers),
                    available_replicas=ready_containers
                )
            
            time.sleep(10)
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="failed",
            platform=DeploymentPlatform.DOCKER,
            strategy=config.strategy,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            replicas=len(containers),
            available_replicas=ready_containers,
            error_message="Timeout waiting for containers to be ready"
        )
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete Docker containers"""
        try:
            containers = self.client.containers.list(all=True, filters={"name": deployment_id})
            
            for container in containers:
                container.remove(force=True)
            
            logger.info(f"Docker deployment deleted successfully: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete Docker deployment: {e}")
            return False


class ServerlessDeployer:
    """Serverless deployment manager"""
    
    def __init__(self, platform: DeploymentPlatform):
        """Initialize serverless client"""
        self.platform = platform
        
        if platform == DeploymentPlatform.AWS_LAMBDA:
            self.lambda_client = boto3.client('lambda')
        elif platform == DeploymentPlatform.AZURE_FUNCTIONS:
            self.credential = DefaultAzureCredential()
            self.web_client = WebSiteManagementClient(
                credential=self.credential,
                subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID', '')
            )
        elif platform == DeploymentPlatform.GCP_FUNCTIONS:
            self.functions_client = google.cloud.functions_v1.CloudFunctionsServiceClient()
        elif platform == DeploymentPlatform.GCP_CLOUD_RUN:
            self.run_client = google.cloud.run_v2.ServicesClient()
        
        logger.info(f"{platform.value} client initialized successfully")
    
    def create_deployment(self, config: DeploymentConfig) -> DeploymentStatus:
        """Create serverless deployment"""
        try:
            deployment_id = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            if self.platform == DeploymentPlatform.AWS_LAMBDA:
                return self._create_aws_lambda_deployment(config, deployment_id)
            elif self.platform == DeploymentPlatform.AZURE_FUNCTIONS:
                return self._create_azure_function_deployment(config, deployment_id)
            elif self.platform == DeploymentPlatform.GCP_FUNCTIONS:
                return self._create_gcp_function_deployment(config, deployment_id)
            elif self.platform == DeploymentPlatform.GCP_CLOUD_RUN:
                return self._create_gcp_cloud_run_deployment(config, deployment_id)
            else:
                raise ValueError(f"Unsupported platform: {self.platform}")
                
        except Exception as e:
            logger.error(f"Failed to create serverless deployment: {e}")
            return DeploymentStatus(
                deployment_id=deployment_id,
                status="failed",
                platform=self.platform,
                strategy=config.strategy,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                replicas=0,
                available_replicas=0,
                error_message=str(e)
            )
    
    def _create_aws_lambda_deployment(self, config: DeploymentConfig, deployment_id: str) -> DeploymentStatus:
        """Create AWS Lambda deployment"""
        # Create Lambda function
        response = self.lambda_client.create_function(
            FunctionName=deployment_id,
            Runtime='python3.9',
            Role=os.getenv('AWS_LAMBDA_ROLE', ''),
            Handler='lambda_function.lambda_handler',
            Code={
                'ImageUri': f"{config.image_name}:{config.image_tag}"
            },
            Environment={
                'Variables': config.environment
            },
            MemorySize=int(config.memory_limit.replace('Mi', '')),
            Timeout=30
        )
        
        # Wait for function to be active
        self._wait_for_lambda_active(deployment_id)
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="running",
            platform=DeploymentPlatform.AWS_LAMBDA,
            strategy=config.strategy,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            replicas=1,
            available_replicas=1,
            url=f"https://{response['FunctionArn']}.lambda-url.{os.getenv('AWS_REGION', 'us-east-1')}.on.aws"
        )
    
    def _create_azure_function_deployment(self, config: DeploymentConfig, deployment_id: str) -> DeploymentStatus:
        """Create Azure Function deployment"""
        # Implementation for Azure Functions
        # This is a simplified version
        function_name = deployment_id
        
        # Create function app
        poller = self.web_client.web_apps.begin_create_or_update(
            resource_group_name=os.getenv('AZURE_RESOURCE_GROUP', ''),
            name=function_name,
            site_envelope={
                'location': os.getenv('AZURE_LOCATION', 'East US'),
                'kind': 'functionapp,linux',
                'reserved': True,
                'site_config': {
                    'linux_fx_version': f'DOCKER|{config.image_name}:{config.image_tag}',
                    'app_settings': [
                        {'name': k, 'value': v} for k, v in config.environment.items()
                    ]
                }
            }
        )
        
        function_app = poller.result()
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="running",
            platform=DeploymentPlatform.AZURE_FUNCTIONS,
            strategy=config.strategy,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            replicas=1,
            available_replicas=1,
            url=f"https://{function_name}.azurewebsites.net"
        )
    
    def _create_gcp_function_deployment(self, config: DeploymentConfig, deployment_id: str) -> DeploymentStatus:
        """Create GCP Function deployment"""
        # Implementation for GCP Functions
        project_id = os.getenv('GCP_PROJECT_ID', '')
        location = os.getenv('GCP_LOCATION', 'us-central1')
        
        function = {
            'name': f'projects/{project_id}/locations/{location}/functions/{deployment_id}',
            'description': f'{config.model_name} v{config.model_version}',
            'runtime': 'python39',
            'entry_point': 'lambda_handler',
            'environment_variables': config.environment,
            'docker_repository': f'{config.image_name}:{config.image_tag}'
        }
        
        operation = self.functions_client.create_function(
            parent=f'projects/{project_id}/locations/{location}',
            function=function,
            function_id=deployment_id
        )
        
        # Wait for operation to complete
        operation.result()
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="running",
            platform=DeploymentPlatform.GCP_FUNCTIONS,
            strategy=config.strategy,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            replicas=1,
            available_replicas=1,
            url=f"https://{location}-{project_id}.cloudfunctions.net/{deployment_id}"
        )
    
    def _create_gcp_cloud_run_deployment(self, config: DeploymentConfig, deployment_id: str) -> DeploymentStatus:
        """Create GCP Cloud Run deployment"""
        project_id = os.getenv('GCP_PROJECT_ID', '')
        location = os.getenv('GCP_LOCATION', 'us-central1')
        
        service = {
            'template': {
                'containers': [{
                    'image': f'{config.image_name}:{config.image_tag}',
                    'env': [{'name': k, 'value': v} for k, v in config.environment.items()],
                    'resources': {
                        'limits': {
                            'cpu': config.cpu_limit,
                            'memory': config.memory_limit
                        }
                    }
                }]
            }
        }
        
        operation = self.run_client.create_service(
            parent=f'projects/{project_id}/locations/{location}',
            service=service,
            service_id=deployment_id
        )
        
        # Wait for operation to complete
        result = operation.result()
        
        return DeploymentStatus(
            deployment_id=deployment_id,
            status="running",
            platform=DeploymentPlatform.GCP_CLOUD_RUN,
            strategy=config.strategy,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            replicas=1,
            available_replicas=1,
            url=result.uri
        )
    
    def _wait_for_lambda_active(self, function_name: str, timeout: int = 300):
        """Wait for Lambda function to be active"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.lambda_client.get_function(FunctionName=function_name)
                if response['Configuration']['State'] == 'Active':
                    return
                time.sleep(10)
            except Exception as e:
                logger.warning(f"Error checking Lambda status: {e}")
                time.sleep(10)
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete serverless deployment"""
        try:
            if self.platform == DeploymentPlatform.AWS_LAMBDA:
                self.lambda_client.delete_function(FunctionName=deployment_id)
            elif self.platform == DeploymentPlatform.AZURE_FUNCTIONS:
                self.web_client.web_apps.delete(
                    resource_group_name=os.getenv('AZURE_RESOURCE_GROUP', ''),
                    name=deployment_id
                )
            elif self.platform == DeploymentPlatform.GCP_FUNCTIONS:
                project_id = os.getenv('GCP_PROJECT_ID', '')
                location = os.getenv('GCP_LOCATION', 'us-central1')
                self.functions_client.delete_function(
                    name=f'projects/{project_id}/locations/{location}/functions/{deployment_id}'
                )
            elif self.platform == DeploymentPlatform.GCP_CLOUD_RUN:
                project_id = os.getenv('GCP_PROJECT_ID', '')
                location = os.getenv('GCP_LOCATION', 'us-central1')
                self.run_client.delete_service(
                    name=f'projects/{project_id}/locations/{location}/services/{deployment_id}'
                )
            
            logger.info(f"Serverless deployment deleted successfully: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete serverless deployment: {e}")
            return False


class BlueGreenDeployer:
    """Blue-green deployment strategy manager"""
    
    def __init__(self, deployer: Union[KubernetesDeployer, DockerDeployer]):
        """Initialize with base deployer"""
        self.deployer = deployer
        self.active_deployments: Dict[str, str] = {}  # model_name -> active_deployment_id
    
    def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        """Execute blue-green deployment"""
        try:
            deployment_id = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            # Get current active deployment
            current_active = self.active_deployments.get(config.model_name)
            
            # Deploy new version (green)
            logger.info(f"Deploying new version (green): {deployment_id}")
            green_status = self.deployer.create_deployment(config)
            
            if green_status.status != "running":
                logger.error("Failed to deploy green version")
                return green_status
            
            # Wait for health checks
            logger.info("Waiting for health checks on green version...")
            time.sleep(config.blue_green_switch_delay)
            
            # Switch traffic to green
            if current_active:
                logger.info(f"Switching traffic from {current_active} to {deployment_id}")
                self._switch_traffic(current_active, deployment_id)
                
                # Keep blue version for rollback
                logger.info(f"Keeping blue version {current_active} for rollback")
            else:
                logger.info(f"Setting {deployment_id} as active deployment")
            
            # Update active deployment
            self.active_deployments[config.model_name] = deployment_id
            
            logger.info(f"Blue-green deployment completed successfully: {deployment_id}")
            return green_status
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return DeploymentStatus(
                deployment_id=deployment_id,
                status="failed",
                platform=config.platform,
                strategy=DeploymentStrategy.BLUE_GREEN,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                replicas=0,
                available_replicas=0,
                error_message=str(e)
            )
    
    def rollback(self, model_name: str) -> bool:
        """Rollback to previous version"""
        try:
            current_active = self.active_deployments.get(model_name)
            if not current_active:
                logger.warning(f"No active deployment found for {model_name}")
                return False
            
            # Find previous deployment (this is simplified)
            # In practice, you'd maintain a deployment history
            logger.info(f"Rolling back deployment for {model_name}")
            
            # Implementation would involve:
            # 1. Finding the previous stable version
            # 2. Switching traffic back
            # 3. Cleaning up failed deployment
            
            logger.info(f"Rollback completed for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _switch_traffic(self, from_deployment: str, to_deployment: str):
        """Switch traffic between deployments"""
        # Implementation depends on the platform
        # For Kubernetes: update service selector
        # For Docker: update load balancer configuration
        # For cloud: update DNS or load balancer rules
        pass


class CanaryDeployer:
    """Canary deployment strategy manager"""
    
    def __init__(self, deployer: Union[KubernetesDeployer, DockerDeployer]):
        """Initialize with base deployer"""
        self.deployer = deployer
        self.canary_deployments: Dict[str, Dict[str, Any]] = {}
    
    def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        """Execute canary deployment"""
        try:
            deployment_id = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            # Deploy canary version
            logger.info(f"Deploying canary version: {deployment_id}")
            canary_config = self._create_canary_config(config)
            canary_status = self.deployer.create_deployment(canary_config)
            
            if canary_status.status != "running":
                logger.error("Failed to deploy canary version")
                return canary_status
            
            # Store canary deployment info
            self.canary_deployments[deployment_id] = {
                'config': config,
                'status': canary_status,
                'start_time': datetime.now(),
                'percentage': config.canary_percentage
            }
            
            # Start gradual rollout
            logger.info(f"Starting canary rollout with {config.canary_percentage}% traffic")
            
            logger.info(f"Canary deployment initiated: {deployment_id}")
            return canary_status
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return DeploymentStatus(
                deployment_id=deployment_id,
                status="failed",
                platform=config.platform,
                strategy=DeploymentStrategy.CANARY,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                replicas=0,
                available_replicas=0,
                error_message=str(e)
            )
    
    def promote_canary(self, deployment_id: str) -> bool:
        """Promote canary to full deployment"""
        try:
            canary_info = self.canary_deployments.get(deployment_id)
            if not canary_info:
                logger.warning(f"Canary deployment not found: {deployment_id}")
                return False
            
            config = canary_info['config']
            
            # Create full deployment with canary configuration
            full_config = self._create_full_config(config)
            full_status = self.deployer.create_deployment(full_config)
            
            if full_status.status == "running":
                # Clean up canary
                self.deployer.delete_deployment(deployment_id)
                del self.canary_deployments[deployment_id]
                
                logger.info(f"Canary promoted successfully: {deployment_id}")
                return True
            else:
                logger.error("Failed to promote canary to full deployment")
                return False
                
        except Exception as e:
            logger.error(f"Canary promotion failed: {e}")
            return False
    
    def rollback_canary(self, deployment_id: str) -> bool:
        """Rollback canary deployment"""
        try:
            canary_info = self.canary_deployments.get(deployment_id)
            if not canary_info:
                logger.warning(f"Canary deployment not found: {deployment_id}")
                return False
            
            # Delete canary deployment
            self.deployer.delete_deployment(deployment_id)
            del self.canary_deployments[deployment_id]
            
            logger.info(f"Canary rollback completed: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Canary rollback failed: {e}")
            return False
    
    def _create_canary_config(self, config: DeploymentConfig) -> DeploymentConfig:
        """Create canary configuration with reduced replicas"""
        canary_config = DeploymentConfig(
            model_name=config.model_name,
            model_version=config.model_version,
            platform=config.platform,
            strategy=config.strategy,
            image_name=config.image_name,
            image_tag=config.image_tag,
            replicas=max(1, config.replicas * config.canary_percentage // 100),
            cpu_request=config.cpu_request,
            cpu_limit=config.cpu_limit,
            memory_request=config.memory_request,
            memory_limit=config.memory_limit,
            gpu_request=config.gpu_request,
            gpu_limit=config.gpu_limit,
            environment=config.environment,
            secrets=config.secrets,
            health_check_path=config.health_check_path,
            health_check_port=config.health_check_port,
            canary_percentage=config.canary_percentage,
            blue_green_switch_delay=config.blue_green_switch_delay,
            auto_scaling=config.auto_scaling,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas,
            target_cpu_utilization=config.target_cpu_utilization,
            target_memory_utilization=config.target_memory_utilization
        )
        return canary_config
    
    def _create_full_config(self, config: DeploymentConfig) -> DeploymentConfig:
        """Create full deployment configuration"""
        full_config = DeploymentConfig(
            model_name=config.model_name,
            model_version=config.model_version,
            platform=config.platform,
            strategy=DeploymentStrategy.ROLLING,
            image_name=config.image_name,
            image_tag=config.image_tag,
            replicas=config.replicas,
            cpu_request=config.cpu_request,
            cpu_limit=config.cpu_limit,
            memory_request=config.memory_request,
            memory_limit=config.memory_limit,
            gpu_request=config.gpu_request,
            gpu_limit=config.gpu_limit,
            environment=config.environment,
            secrets=config.secrets,
            health_check_path=config.health_check_path,
            health_check_port=config.health_check_port,
            canary_percentage=100,
            blue_green_switch_delay=config.blue_green_switch_delay,
            auto_scaling=config.auto_scaling,
            min_replicas=config.min_replicas,
            max_replicas=config.max_replicas,
            target_cpu_utilization=config.target_cpu_utilization,
            target_memory_utilization=config.target_memory_utilization
        )
        return full_config


class InfrastructureAsCode:
    """Infrastructure as Code (IaC) manager"""
    
    def __init__(self):
        """Initialize IaC manager"""
        self.templates_dir = Path("infrastructure/templates")
        self.outputs_dir = Path("infrastructure/outputs")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_kubernetes_manifest(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes manifest as YAML"""
        k8s_deployer = KubernetesDeployer()
        
        deployment_manifest = k8s_deployer._create_deployment_manifest(
            config, f"{config.model_name}-{config.model_version}"
        )
        service_manifest = k8s_deployer._create_service_manifest(
            config, f"{config.model_name}-{config.model_version}"
        )
        
        if config.auto_scaling:
            hpa_manifest = k8s_deployer._create_hpa_manifest(
                config, f"{config.model_name}-{config.model_version}"
            )
            manifests = [deployment_manifest, service_manifest, hpa_manifest]
        else:
            manifests = [deployment_manifest, service_manifest]
        
        # Convert to YAML
        yaml_content = ""
        for manifest in manifests:
            yaml_content += yaml.dump(manifest, default_flow_style=False)
            yaml_content += "---\n"
        
        return yaml_content
    
    def generate_docker_compose(self, config: DeploymentConfig) -> str:
        """Generate Docker Compose configuration"""
        compose_config = {
            'version': '3.8',
            'services': {
                f"{config.model_name}-{config.model_version}": {
                    'image': f"{config.image_name}:{config.image_tag}",
                    'ports': [f"{config.health_check_port}:{config.health_check_port}"],
                    'environment': config.environment,
                    'deploy': {
                        'replicas': config.replicas,
                        'resources': {
                            'limits': {
                                'cpus': config.cpu_limit,
                                'memory': config.memory_limit
                            },
                            'reservations': {
                                'cpus': config.cpu_request,
                                'memory': config.memory_request
                            }
                        }
                    },
                    'healthcheck': {
                        'test': [f"CMD", "curl", "-f", f"http://localhost:{config.health_check_port}{config.health_check_path}"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            }
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def generate_terraform_config(self, config: DeploymentConfig) -> str:
        """Generate Terraform configuration"""
        # This is a simplified example
        terraform_config = f"""
terraform {{
  required_providers {{
    kubernetes = {{
      source = "hashicorp/kubernetes"
      version = ">= 2.0.0"
    }}
  }}
}}

provider "kubernetes" {{
  config_path = "~/.kube/config"
}}

resource "kubernetes_deployment" "{config.model_name}_{config.model_version}" {{
  metadata {{
    name = "{config.model_name}-{config.model_version}"
    labels {{
      app = "{config.model_name}"
      version = "{config.model_version}"
    }}
  }}
  
  spec {{
    replicas = {config.replicas}
    
    selector {{
      match_labels {{
        app = "{config.model_name}"
        version = "{config.model_version}"
      }}
    }}
    
    template {{
      metadata {{
        labels {{
          app = "{config.model_name}"
          version = "{config.model_version}"
        }}
      }}
      
      spec {{
        container {{
          name = "{config.model_name}"
          image = "{config.image_name}:{config.image_tag}"
          ports {{
            container_port = {config.health_check_port}
          }}
          
          resources {{
            limits {{
              cpu = "{config.cpu_limit}"
              memory = "{config.memory_limit}"
            }}
            requests {{
              cpu = "{config.cpu_request}"
              memory = "{config.memory_request}"
            }}
          }}
          
          env {{
            name = "MODEL_NAME"
            value = "{config.model_name}"
          }}
          
          liveness_probe {{
            http_get {{
              path = "{config.health_check_path}"
              port = {config.health_check_port}
            }}
            initial_delay_seconds = 30
            period_seconds = 10
          }}
          
          readiness_probe {{
            http_get {{
              path = "{config.health_check_path}"
              port = {config.health_check_port}
            }}
            initial_delay_seconds = 5
            period_seconds = 5
          }}
        }}
      }}
    }}
  }}
}}
"""
        return terraform_config
    
    def save_template(self, config: DeploymentConfig, template_type: str) -> str:
        """Save infrastructure template to file"""
        timestamp = int(time.time())
        filename = f"{config.model_name}_{config.model_version}_{template_type}_{timestamp}.yaml"
        filepath = self.templates_dir / filename
        
        if template_type == "kubernetes":
            content = self.generate_kubernetes_manifest(config)
        elif template_type == "docker-compose":
            content = self.generate_docker_compose(config)
        elif template_type == "terraform":
            content = self.generate_terraform_config(config)
        else:
            raise ValueError(f"Unsupported template type: {template_type}")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Template saved: {filepath}")
        return str(filepath)
    
    def apply_template(self, template_path: str) -> bool:
        """Apply infrastructure template"""
        try:
            if "kubernetes" in template_path:
                # Apply Kubernetes manifest
                result = subprocess.run(
                    ["kubectl", "apply", "-f", template_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Kubernetes template applied successfully: {template_path}")
                    return True
                else:
                    logger.error(f"Failed to apply Kubernetes template: {result.stderr}")
                    return False
            
            elif "docker-compose" in template_path:
                # Apply Docker Compose
                result = subprocess.run(
                    ["docker-compose", "-f", template_path, "up", "-d"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Docker Compose template applied successfully: {template_path}")
                    return True
                else:
                    logger.error(f"Failed to apply Docker Compose template: {result.stderr}")
                    return False
            
            elif "terraform" in template_path:
                # Apply Terraform
                # Initialize Terraform
                init_result = subprocess.run(
                    ["terraform", "init"],
                    cwd=template_path.parent,
                    capture_output=True,
                    text=True
                )
                
                if init_result.returncode != 0:
                    logger.error(f"Terraform init failed: {init_result.stderr}")
                    return False
                
                # Apply Terraform
                apply_result = subprocess.run(
                    ["terraform", "apply", "-auto-approve"],
                    cwd=template_path.parent,
                    capture_output=True,
                    text=True
                )
                
                if apply_result.returncode == 0:
                    logger.info(f"Terraform template applied successfully: {template_path}")
                    return True
                else:
                    logger.error(f"Failed to apply Terraform template: {apply_result.stderr}")
                    return False
            
            else:
                logger.error(f"Unknown template type: {template_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply template: {e}")
            return False


class MLOpsDeploymentInfrastructure:
    """Main MLOps Deployment Infrastructure class"""
    
    def __init__(self):
        """Initialize deployment infrastructure"""
        self.kubernetes_deployer = KubernetesDeployer()
        self.docker_deployer = DockerDeployer()
        self.blue_green_deployer = BlueGreenDeployer(self.kubernetes_deployer)
        self.canary_deployer = CanaryDeployer(self.kubernetes_deployer)
        self.iac = InfrastructureAsCode()
        self.deployments: Dict[str, DeploymentStatus] = {}
        
        logger.info("MLOps Deployment Infrastructure initialized")
    
    def deploy_model(self, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy model using specified platform and strategy"""
        try:
            logger.info(f"Starting deployment: {config.model_name} v{config.model_version}")
            
            # Select deployer based on platform
            if config.platform == DeploymentPlatform.KUBERNETES:
                base_deployer = self.kubernetes_deployer
            elif config.platform == DeploymentPlatform.DOCKER:
                base_deployer = self.docker_deployer
            else:
                serverless_deployer = ServerlessDeployer(config.platform)
                return serverless_deployer.create_deployment(config)
            
            # Apply deployment strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                blue_green_deployer = BlueGreenDeployer(base_deployer)
                status = blue_green_deployer.deploy(config)
            elif config.strategy == DeploymentStrategy.CANARY:
                canary_deployer = CanaryDeployer(base_deployer)
                status = canary_deployer.deploy(config)
            else:
                status = base_deployer.create_deployment(config)
            
            # Store deployment status
            self.deployments[status.deployment_id] = status
            
            logger.info(f"Deployment completed: {status.deployment_id} - {status.status}")
            return status
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return DeploymentStatus(
                deployment_id=f"{config.model_name}-{config.model_version}-{int(time.time())}",
                status="failed",
                platform=config.platform,
                strategy=config.strategy,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                replicas=0,
                available_replicas=0,
                error_message=str(e)
            )
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get deployment status"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self, model_name: Optional[str] = None) -> List[DeploymentStatus]:
        """List all deployments or deployments for specific model"""
        if model_name:
            return [status for status in self.deployments.values() 
                   if model_name in status.deployment_id]
        return list(self.deployments.values())
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete deployment"""
        try:
            status = self.deployments.get(deployment_id)
            if not status:
                logger.warning(f"Deployment not found: {deployment_id}")
                return False
            
            # Select appropriate deployer
            if status.platform == DeploymentPlatform.KUBERNETES:
                success = self.kubernetes_deployer.delete_deployment(deployment_id)
            elif status.platform == DeploymentPlatform.DOCKER:
                success = self.docker_deployer.delete_deployment(deployment_id)
            else:
                serverless_deployer = ServerlessDeployer(status.platform)
                success = serverless_deployer.delete_deployment(deployment_id)
            
            if success:
                del self.deployments[deployment_id]
                logger.info(f"Deployment deleted: {deployment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False
    
    def generate_infrastructure_template(self, config: DeploymentConfig, template_type: str) -> str:
        """Generate infrastructure template"""
        return self.iac.save_template(config, template_type)
    
    def apply_infrastructure_template(self, template_path: str) -> bool:
        """Apply infrastructure template"""
        return self.iac.apply_template(template_path)
    
    def promote_canary(self, deployment_id: str) -> bool:
        """Promote canary deployment to full deployment"""
        return self.canary_deployer.promote_canary(deployment_id)
    
    def rollback_canary(self, deployment_id: str) -> bool:
        """Rollback canary deployment"""
        return self.canary_deployer.rollback_canary(deployment_id)
    
    def rollback_blue_green(self, model_name: str) -> bool:
        """Rollback blue-green deployment"""
        return self.blue_green_deployer.rollback(model_name)


def main():
    """Main function for testing the deployment infrastructure"""
    # Example usage
    mlops_infra = MLOpsDeploymentInfrastructure()
    
    # Create a deployment configuration
    config = DeploymentConfig(
        model_name="bharat-language-model",
        model_version="v1.0.0",
        platform=DeploymentPlatform.KUBERNETES,
        strategy=DeploymentStrategy.BLUE_GREEN,
        image_name="bharat-fm/language-model",
        image_tag="v1.0.0",
        replicas=3,
        environment={
            "MODEL_NAME": "bharat-language-model",
            "MODEL_VERSION": "v1.0.0",
            "ENVIRONMENT": "production"
        },
        health_check_path="/health",
        health_check_port=8000,
        blue_green_switch_delay=60  # 1 minute for testing
    )
    
    # Deploy the model
    status = mlops_infra.deploy_model(config)
    print(f"Deployment status: {status.status}")
    print(f"Deployment ID: {status.deployment_id}")
    
    # Generate infrastructure template
    template_path = mlops_infra.generate_infrastructure_template(config, "kubernetes")
    print(f"Template generated: {template_path}")
    
    # List deployments
    deployments = mlops_infra.list_deployments()
    print(f"Total deployments: {len(deployments)}")


if __name__ == "__main__":
    main()