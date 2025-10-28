"""
Advanced Model Registry & Versioning System for Bharat-FM Phase 2

This module provides comprehensive model management capabilities including:
- Model registration and versioning
- Metadata management
- Performance tracking
- Model lifecycle management
- Deployment management
"""

import asyncio
import json
import hashlib
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Model type categories"""
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    GENERATION = "generation"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float = 0.0
    latency: float = 0.0  # in milliseconds
    throughput: float = 0.0  # requests per second
    memory_usage: float = 0.0  # in MB
    cost_per_request: float = 0.0
    f1_score: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelMetadata:
    """Model metadata information"""
    name: str
    description: str
    version: str
    model_type: ModelType
    framework: str  # pytorch, tensorflow, onnx, etc.
    architecture: str  # transformer, cnn, rnn, etc.
    parameters: int  # number of parameters
    file_size: int  # in bytes
    created_by: str
    created_at: datetime
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    input_format: str = "text"  # text, image, audio, multimodal
    output_format: str = "text"
    supported_languages: List[str] = field(default_factory=list)
    domain: str = "general"  # general, governance, education, finance, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['model_type'] = self.model_type.value
        data['created_at'] = self.created_at.isoformat() if self.created_at else datetime.now().isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['model_type'] = ModelType(data['model_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Individual model version"""
    version_id: str
    model_id: str
    metadata: ModelMetadata
    metrics: ModelMetrics
    file_path: str
    checksum: str
    status: ModelStatus
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['metadata'] = self.metadata.to_dict()
        data['metrics'] = self.metrics.to_dict()
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        data = data.copy()
        data['metadata'] = ModelMetadata.from_dict(data['metadata'])
        data['metrics'] = ModelMetrics.from_dict(data['metrics'])
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ModelDeployment:
    """Model deployment information"""
    deployment_id: str
    model_id: str
    version_id: str
    endpoint: str
    environment: str  # development, staging, production
    status: str  # running, stopped, error
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelDeployment':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class ModelRegistry:
    """Advanced Model Registry & Versioning System"""
    
    def __init__(self, storage_path: str = "./model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.models_file = self.storage_path / "models.pkl"
        self.versions_file = self.storage_path / "versions.pkl"
        self.deployments_file = self.storage_path / "deployments.pkl"
        
        # In-memory storage
        self.models: Dict[str, ModelMetadata] = {}
        self.versions: Dict[str, ModelVersion] = {}
        self.deployments: Dict[str, ModelDeployment] = {}
        
        # Load existing data
        self._load_data()
        
        logger.info(f"ModelRegistry initialized with storage path: {storage_path}")
    
    async def start(self):
        """Start the model registry"""
        logger.info("ModelRegistry started")
    
    async def stop(self):
        """Stop the model registry and save data"""
        self._save_data()
        logger.info("ModelRegistry stopped")
    
    def _load_data(self):
        """Load data from disk"""
        try:
            if self.models_file.exists():
                with open(self.models_file, 'rb') as f:
                    self.models = pickle.load(f)
            
            if self.versions_file.exists():
                with open(self.versions_file, 'rb') as f:
                    self.versions = pickle.load(f)
            
            if self.deployments_file.exists():
                with open(self.deployments_file, 'rb') as f:
                    self.deployments = pickle.load(f)
                    
            logger.info(f"Loaded {len(self.models)} models, {len(self.versions)} versions, {len(self.deployments)} deployments")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to disk"""
        try:
            with open(self.models_file, 'wb') as f:
                pickle.dump(self.models, f)
            
            with open(self.versions_file, 'wb') as f:
                pickle.dump(self.versions, f)
            
            with open(self.deployments_file, 'wb') as f:
                pickle.dump(self.deployments, f)
                
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = int(time.time())
        random_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_hash}"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    async def register_model(self, metadata: ModelMetadata, file_path: str) -> str:
        """Register a new model"""
        model_id = self._generate_id("model")
        metadata.created_at = datetime.now()
        
        # Store model metadata
        self.models[model_id] = metadata
        
        # Create initial version
        version_id = self._generate_id("version")
        checksum = self._calculate_checksum(file_path)
        
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            metadata=metadata,
            metrics=ModelMetrics(),
            file_path=file_path,
            checksum=checksum,
            status=ModelStatus.DEVELOPMENT,
            is_default=True
        )
        
        self.versions[version_id] = version
        
        self._save_data()
        logger.info(f"Registered model {metadata.name} with ID {model_id}")
        return model_id
    
    async def add_version(self, model_id: str, metadata: ModelMetadata, file_path: str, 
                         metrics: Optional[ModelMetrics] = None) -> str:
        """Add a new version to an existing model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        version_id = self._generate_id("version")
        checksum = self._calculate_checksum(file_path)
        
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            metadata=metadata,
            metrics=metrics or ModelMetrics(),
            file_path=file_path,
            checksum=checksum,
            status=ModelStatus.DEVELOPMENT
        )
        
        self.versions[version_id] = version
        self._save_data()
        
        logger.info(f"Added version {version_id} to model {model_id}")
        return version_id
    
    async def update_metrics(self, version_id: str, metrics: ModelMetrics):
        """Update model version metrics"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.versions[version_id].metrics = metrics
        self.versions[version_id].updated_at = datetime.now()
        self._save_data()
        
        logger.info(f"Updated metrics for version {version_id}")
    
    async def set_version_status(self, version_id: str, status: ModelStatus):
        """Update version status"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.versions[version_id].status = status
        self.versions[version_id].updated_at = datetime.now()
        self._save_data()
        
        logger.info(f"Updated status for version {version_id} to {status.value}")
    
    async def set_default_version(self, model_id: str, version_id: str):
        """Set default version for a model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Clear existing default
        for v in self.versions.values():
            if v.model_id == model_id and v.is_default:
                v.is_default = False
        
        # Set new default
        self.versions[version_id].is_default = True
        self.versions[version_id].updated_at = datetime.now()
        self._save_data()
        
        logger.info(f"Set version {version_id} as default for model {model_id}")
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self.models.get(model_id)
    
    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get model version"""
        return self.versions.get(version_id)
    
    async def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        return [v for v in self.versions.values() if v.model_id == model_id]
    
    async def get_default_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get default version of a model"""
        versions = await self.get_model_versions(model_id)
        return next((v for v in versions if v.is_default), None)
    
    async def list_models(self, model_type: Optional[ModelType] = None, 
                         domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models with optional filtering"""
        models = []
        for model_id, metadata in self.models.items():
            if model_type and metadata.model_type != model_type:
                continue
            if domain and metadata.domain != domain:
                continue
            
            versions = await self.get_model_versions(model_id)
            default_version = await self.get_default_version(model_id)
            
            models.append({
                "model_id": model_id,
                "metadata": metadata.to_dict(),
                "versions_count": len(versions),
                "default_version": default_version.to_dict() if default_version else None
            })
        
        return models
    
    async def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search models by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for model_id, metadata in self.models.items():
            # Search in name, description, and tags
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                
                versions = await self.get_model_versions(model_id)
                default_version = await self.get_default_version(model_id)
                
                results.append({
                    "model_id": model_id,
                    "metadata": metadata.to_dict(),
                    "versions_count": len(versions),
                    "default_version": default_version.to_dict() if default_version else None
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def deploy_model(self, model_id: str, version_id: str, endpoint: str, 
                         environment: str, config: Dict[str, Any] = None) -> str:
        """Deploy a model version"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        deployment_id = self._generate_id("deployment")
        
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            version_id=version_id,
            endpoint=endpoint,
            environment=environment,
            status="running",
            config=config or {}
        )
        
        self.deployments[deployment_id] = deployment
        self._save_data()
        
        logger.info(f"Deployed model {model_id} version {version_id} to {endpoint}")
        return deployment_id
    
    async def update_deployment_status(self, deployment_id: str, status: str, 
                                    metrics: Dict[str, Any] = None):
        """Update deployment status and metrics"""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        self.deployments[deployment_id].status = status
        self.deployments[deployment_id].updated_at = datetime.now()
        if metrics:
            self.deployments[deployment_id].metrics.update(metrics)
        
        self._save_data()
        logger.info(f"Updated deployment {deployment_id} status to {status}")
    
    async def get_deployments(self, environment: Optional[str] = None) -> List[ModelDeployment]:
        """Get deployments with optional filtering"""
        deployments = list(self.deployments.values())
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        return deployments
    
    async def get_model_deployments(self, model_id: str) -> List[ModelDeployment]:
        """Get deployments for a specific model"""
        return [d for d in self.deployments.values() if d.model_id == model_id]
    
    async def delete_model(self, model_id: str):
        """Delete a model and all its versions"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Delete versions
        versions_to_delete = [v_id for v_id, v in self.versions.items() if v.model_id == model_id]
        for version_id in versions_to_delete:
            del self.versions[version_id]
        
        # Delete deployments
        deployments_to_delete = [d_id for d_id, d in self.deployments.items() if d.model_id == model_id]
        for deployment_id in deployments_to_delete:
            del self.deployments[deployment_id]
        
        # Delete model
        del self.models[model_id]
        
        self._save_data()
        logger.info(f"Deleted model {model_id} and all its versions")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_models": len(self.models),
            "total_versions": len(self.versions),
            "total_deployments": len(self.deployments),
            "models_by_type": {},
            "models_by_domain": {},
            "versions_by_status": {},
            "deployments_by_environment": {}
        }
        
        # Models by type
        for metadata in self.models.values():
            model_type = metadata.model_type.value
            stats["models_by_type"][model_type] = stats["models_by_type"].get(model_type, 0) + 1
        
        # Models by domain
        for metadata in self.models.values():
            domain = metadata.domain
            stats["models_by_domain"][domain] = stats["models_by_domain"].get(domain, 0) + 1
        
        # Versions by status
        for version in self.versions.values():
            status = version.status.value
            stats["versions_by_status"][status] = stats["versions_by_status"].get(status, 0) + 1
        
        # Deployments by environment
        for deployment in self.deployments.values():
            env = deployment.environment
            stats["deployments_by_environment"][env] = stats["deployments_by_environment"].get(env, 0) + 1
        
        return stats


# Factory function for creating model registry
async def create_model_registry(config: Dict[str, Any] = None) -> ModelRegistry:
    """Create and initialize model registry"""
    config = config or {}
    storage_path = config.get("storage_path", "./model_registry")
    
    registry = ModelRegistry(storage_path)
    await registry.start()
    
    return registry