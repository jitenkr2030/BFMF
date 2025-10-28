"""
MLOps Configuration Management System for Bharat-FM Platform

This module provides centralized configuration management for all MLOps components,
including model training, deployment, monitoring, and infrastructure settings.
It supports environment-specific configurations, validation, and dynamic updates.

Features:
- Centralized configuration management
- Environment-specific settings
- Configuration validation and schema enforcement
- Dynamic configuration updates
- Configuration versioning and rollback
- Integration with external configuration sources
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import jsonschema
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigType(Enum):
    """Configuration types"""
    MODEL = "model"
    TRAINING = "training"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_type: str
    model_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    framework: str = "pytorch"
    device: str = "auto"
    batch_size: int = 32
    max_sequence_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

@dataclass
class TrainingConfig:
    """Training configuration"""
    experiment_name: str
    dataset_path: str
    output_dir: str
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    seed: int = 42
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_name: str
    model_path: str
    api_endpoint: str
    replicas: int = 1
    cpu_request: str = "1000m"
    memory_request: str = "1Gi"
    gpu_request: str = "0"
    autoscaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 5
    target_cpu_utilization: int = 70
    rolling_update_enabled: bool = True
    max_unavailable: int = 1
    max_surge: int = 1

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics_collection_enabled: bool = True
    metrics_interval_seconds: int = 30
    alerting_enabled: bool = True
    log_level: str = "INFO"
    tracing_enabled: bool = False
    custom_metrics: List[str] = field(default_factory=list)
    alert_channels: List[str] = field(default_factory=list)

@dataclass
class InfrastructureConfig:
    """Infrastructure configuration"""
    cluster_name: str
    region: str
    node_pools: Dict[str, Any] = field(default_factory=dict)
    storage_class: str = "standard"
    network_policy: str = "default"
    resource_quotas: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityConfig:
    """Security configuration"""
    authentication_enabled: bool = True
    authorization_enabled: bool = True
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    allowed_origins: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    rate_limiting: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MLOpsConfig:
    """Main MLOps configuration"""
    environment: Environment
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class MLOpsConfigManager:
    """
    MLOps Configuration Management System
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration storage
        self.configs: Dict[str, MLOpsConfig] = {}
        self.active_config: Optional[str] = None
        
        # Configuration schemas
        self.schemas = self._load_schemas()
        
        # Configuration history
        self.config_history: List[Dict[str, Any]] = []
        
        # Validation rules
        self.validation_rules = self._load_validation_rules()
        
    def create_config(self, config_id: str, environment: Environment, 
                     base_config: Dict[str, Any] = None) -> MLOpsConfig:
        """
        Create a new MLOps configuration
        
        Args:
            config_id: Unique identifier for the configuration
            environment: Deployment environment
            base_config: Optional base configuration dictionary
            
        Returns:
            MLOpsConfig object
        """
        if config_id in self.configs:
            raise ValueError(f"Configuration {config_id} already exists")
            
        # Create base configuration
        if base_config:
            config = self._dict_to_config(base_config, environment)
        else:
            config = MLOpsConfig(environment=environment, version="1.0.0")
            
        # Validate configuration
        self._validate_config(config)
        
        # Store configuration
        self.configs[config_id] = config
        
        # Set as active if first configuration
        if self.active_config is None:
            self.active_config = config_id
            
        logger.info(f"Created configuration {config_id} for {environment.value} environment")
        
        return config
        
    def update_config(self, config_id: str, updates: Dict[str, Any]) -> MLOpsConfig:
        """
        Update an existing configuration
        
        Args:
            config_id: Configuration identifier
            updates: Dictionary of updates to apply
            
        Returns:
            Updated MLOpsConfig object
        """
        if config_id not in self.configs:
            raise ValueError(f"Configuration {config_id} not found")
            
        # Get current configuration
        current_config = self.configs[config_id]
        
        # Create history entry
        history_entry = {
            'config_id': config_id,
            'timestamp': datetime.now(),
            'old_config': asdict(current_config),
            'update_type': 'full_update'
        }
        
        # Apply updates
        updated_config = self._apply_updates(current_config, updates)
        updated_config.updated_at = datetime.now()
        
        # Validate updated configuration
        self._validate_config(updated_config)
        
        # Store updated configuration
        self.configs[config_id] = updated_config
        
        # Add to history
        history_entry['new_config'] = asdict(updated_config)
        self.config_history.append(history_entry)
        
        logger.info(f"Updated configuration {config_id}")
        
        return updated_config
        
    def get_config(self, config_id: str = None) -> MLOpsConfig:
        """
        Get configuration by ID or active configuration
        
        Args:
            config_id: Optional configuration ID
            
        Returns:
            MLOpsConfig object
        """
        if config_id is None:
            config_id = self.active_config
            
        if config_id is None:
            raise ValueError("No active configuration found")
            
        if config_id not in self.configs:
            raise ValueError(f"Configuration {config_id} not found")
            
        return self.configs[config_id]
        
    def delete_config(self, config_id: str):
        """
        Delete a configuration
        
        Args:
            config_id: Configuration identifier
        """
        if config_id not in self.configs:
            raise ValueError(f"Configuration {config_id} not found")
            
        # Remove from active if it's the active config
        if self.active_config == config_id:
            self.active_config = None
            
        # Delete configuration
        del self.configs[config_id]
        
        logger.info(f"Deleted configuration {config_id}")
        
    def set_active_config(self, config_id: str):
        """
        Set the active configuration
        
        Args:
            config_id: Configuration identifier
        """
        if config_id not in self.configs:
            raise ValueError(f"Configuration {config_id} not found")
            
        self.active_config = config_id
        logger.info(f"Set active configuration to {config_id}")
        
    def list_configs(self) -> List[Dict[str, Any]]:
        """
        List all configurations
        
        Returns:
            List of configuration summaries
        """
        return [
            {
                'config_id': config_id,
                'environment': config.environment.value,
                'version': config.version,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat(),
                'is_active': config_id == self.active_config
            }
            for config_id, config in self.configs.items()
        ]
        
    def save_config(self, config_id: str, filename: str = None):
        """
        Save configuration to file
        
        Args:
            config_id: Configuration identifier
            filename: Optional filename (defaults to config_id.yaml)
        """
        config = self.get_config(config_id)
        
        if filename is None:
            filename = f"{config_id}.yaml"
            
        filepath = self.config_dir / filename
        
        # Convert to dictionary
        config_dict = asdict(config)
        config_dict['environment'] = config.environment.value
        
        # Save as YAML
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        logger.info(f"Saved configuration {config_id} to {filepath}")
        
    def load_config(self, filename: str, config_id: str = None) -> MLOpsConfig:
        """
        Load configuration from file
        
        Args:
            filename: Configuration filename
            config_id: Optional configuration ID (defaults to filename without extension)
            
        Returns:
            MLOpsConfig object
        """
        if config_id is None:
            config_id = Path(filename).stem
            
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file {filepath} not found")
            
        # Load YAML
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert environment string to enum
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
            
        # Convert datetime strings
        for field in ['created_at', 'updated_at']:
            if field in config_dict and isinstance(config_dict[field], str):
                config_dict[field] = datetime.fromisoformat(config_dict[field])
                
        # Create configuration object
        config = self._dict_to_config(config_dict, config_dict.get('environment'))
        
        # Validate configuration
        self._validate_config(config)
        
        # Store configuration
        self.configs[config_id] = config
        
        # Set as active if first configuration
        if self.active_config is None:
            self.active_config = config_id
            
        logger.info(f"Loaded configuration {config_id} from {filepath}")
        
        return config
        
    def get_config_history(self, config_id: str = None) -> List[Dict[str, Any]]:
        """
        Get configuration history
        
        Args:
            config_id: Optional configuration ID filter
            
        Returns:
            List of history entries
        """
        if config_id:
            return [
                entry for entry in self.config_history
                if entry['config_id'] == config_id
            ]
        return self.config_history.copy()
        
    def rollback_config(self, config_id: str, history_index: int) -> MLOpsConfig:
        """
        Rollback configuration to a previous version
        
        Args:
            config_id: Configuration identifier
            history_index: Index in history to rollback to
            
        Returns:
            MLOpsConfig object
        """
        history = self.get_config_history(config_id)
        
        if history_index >= len(history):
            raise IndexError("History index out of range")
            
        history_entry = history[history_index]
        old_config_dict = history_entry['old_config']
        
        # Restore configuration
        restored_config = self._dict_to_config(old_config_dict, old_config_dict.get('environment'))
        restored_config.updated_at = datetime.now()
        
        # Validate configuration
        self._validate_config(restored_config)
        
        # Store restored configuration
        self.configs[config_id] = restored_config
        
        # Add rollback entry to history
        rollback_entry = {
            'config_id': config_id,
            'timestamp': datetime.now(),
            'old_config': asdict(self.configs[config_id]),
            'new_config': asdict(restored_config),
            'update_type': 'rollback',
            'rollback_from_index': history_index
        }
        self.config_history.append(rollback_entry)
        
        logger.info(f"Rolled back configuration {config_id} to history index {history_index}")
        
        return restored_config
        
    def validate_config(self, config_id: str) -> List[str]:
        """
        Validate a configuration and return validation errors
        
        Args:
            config_id: Configuration identifier
            
        Returns:
            List of validation error messages
        """
        config = self.get_config(config_id)
        return self._validate_config(config, return_errors=True)
        
    def export_config(self, config_id: str, format: str = 'yaml') -> str:
        """
        Export configuration to string format
        
        Args:
            config_id: Configuration identifier
            format: Export format ('yaml' or 'json')
            
        Returns:
            Configuration string
        """
        config = self.get_config(config_id)
        config_dict = asdict(config)
        config_dict['environment'] = config.environment.value
        
        if format.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            return json.dumps(config_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def get_environment_config(self, environment: Environment) -> Dict[str, Any]:
        """
        Get environment-specific configuration overrides
        
        Args:
            environment: Environment enum
            
        Returns:
            Dictionary of environment-specific settings
        """
        env_configs = {
            Environment.DEVELOPMENT: {
                'monitoring': {
                    'metrics_interval_seconds': 10,
                    'log_level': 'DEBUG'
                },
                'deployment': {
                    'replicas': 1,
                    'autoscaling_enabled': False
                }
            },
            Environment.STAGING: {
                'monitoring': {
                    'metrics_interval_seconds': 30,
                    'log_level': 'INFO'
                },
                'deployment': {
                    'replicas': 2,
                    'autoscaling_enabled': True
                }
            },
            Environment.PRODUCTION: {
                'monitoring': {
                    'metrics_interval_seconds': 60,
                    'log_level': 'WARNING'
                },
                'deployment': {
                    'replicas': 3,
                    'autoscaling_enabled': True,
                    'rolling_update_enabled': True
                },
                'security': {
                    'authentication_enabled': True,
                    'authorization_enabled': True,
                    'encryption_enabled': True
                }
            }
        }
        
        return env_configs.get(environment, {})
        
    def _dict_to_config(self, config_dict: Dict[str, Any], environment: Environment) -> MLOpsConfig:
        """Convert dictionary to MLOpsConfig object"""
        # Handle nested configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        deployment_config = DeploymentConfig(**config_dict.get('deployment', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        infrastructure_config = InfrastructureConfig(**config_dict.get('infrastructure', {}))
        security_config = SecurityConfig(**config_dict.get('security', {}))
        
        return MLOpsConfig(
            environment=environment,
            version=config_dict.get('version', '1.0.0'),
            created_at=config_dict.get('created_at', datetime.now()),
            updated_at=config_dict.get('updated_at', datetime.now()),
            model=model_config,
            training=training_config,
            deployment=deployment_config,
            monitoring=monitoring_config,
            infrastructure=infrastructure_config,
            security=security_config,
            custom_settings=config_dict.get('custom_settings', {})
        )
        
    def _apply_updates(self, config: MLOpsConfig, updates: Dict[str, Any]) -> MLOpsConfig:
        """Apply updates to configuration"""
        config_dict = asdict(config)
        
        def deep_update(base_dict: Dict, update_dict: Dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
                    
        deep_update(config_dict, updates)
        
        return self._dict_to_config(config_dict, config.environment)
        
    def _validate_config(self, config: MLOpsConfig, return_errors: bool = False) -> List[str]:
        """Validate configuration against schemas and rules"""
        errors = []
        
        # Validate against JSON schema
        try:
            config_dict = asdict(config)
            config_dict['environment'] = config.environment.value
            jsonschema.validate(config_dict, self.schemas['mlops_config'])
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            
        # Apply custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_errors = rule_func(config)
                errors.extend(rule_errors)
            except Exception as e:
                errors.append(f"Validation rule {rule_name} failed: {str(e)}")
                
        # Environment-specific validation
        env_errors = self._validate_environment_specific(config)
        errors.extend(env_errors)
        
        if return_errors:
            return errors
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
        return errors
        
    def _validate_environment_specific(self, config: MLOpsConfig) -> List[str]:
        """Validate environment-specific rules"""
        errors = []
        
        if config.environment == Environment.PRODUCTION:
            # Production-specific validations
            if not config.security.authentication_enabled:
                errors.append("Authentication must be enabled in production")
                
            if not config.security.encryption_enabled:
                errors.append("Encryption must be enabled in production")
                
            if config.deployment.replicas < 2:
                errors.append("At least 2 replicas required in production")
                
        elif config.environment == Environment.DEVELOPMENT:
            # Development-specific validations
            if config.deployment.replicas > 1:
                errors.append("Only 1 replica recommended in development")
                
        return errors
        
    def _load_schemas(self) -> Dict[str, Any]:
        """Load JSON schemas for validation"""
        # Define schema for MLOps configuration
        schema = {
            "type": "object",
            "properties": {
                "environment": {"type": "string", "enum": ["development", "staging", "production"]},
                "version": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "model": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "model_type": {"type": "string"},
                        "model_version": {"type": "string"},
                        "parameters": {"type": "object"},
                        "framework": {"type": "string"},
                        "device": {"type": "string"},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "max_sequence_length": {"type": "integer", "minimum": 1},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                        "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                        "top_k": {"type": "integer", "minimum": 0}
                    },
                    "required": ["model_name", "model_type", "model_version"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "experiment_name": {"type": "string"},
                        "dataset_path": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "epochs": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "gradient_accumulation_steps": {"type": "integer", "minimum": 1},
                        "warmup_steps": {"type": "integer", "minimum": 0},
                        "weight_decay": {"type": "number", "minimum": 0},
                        "save_steps": {"type": "integer", "minimum": 1},
                        "eval_steps": {"type": "integer", "minimum": 1},
                        "logging_steps": {"type": "integer", "minimum": 1},
                        "seed": {"type": "integer"},
                        "mixed_precision": {"type": "boolean"},
                        "gradient_checkpointing": {"type": "boolean"}
                    },
                    "required": ["experiment_name", "dataset_path", "output_dir"]
                },
                "deployment": {
                    "type": "object",
                    "properties": {
                        "deployment_name": {"type": "string"},
                        "model_path": {"type": "string"},
                        "api_endpoint": {"type": "string"},
                        "replicas": {"type": "integer", "minimum": 1},
                        "cpu_request": {"type": "string"},
                        "memory_request": {"type": "string"},
                        "gpu_request": {"type": "string"},
                        "autoscaling_enabled": {"type": "boolean"},
                        "min_replicas": {"type": "integer", "minimum": 1},
                        "max_replicas": {"type": "integer", "minimum": 1},
                        "target_cpu_utilization": {"type": "integer", "minimum": 1, "maximum": 100},
                        "rolling_update_enabled": {"type": "boolean"},
                        "max_unavailable": {"type": "integer", "minimum": 0},
                        "max_surge": {"type": "integer", "minimum": 0}
                    },
                    "required": ["deployment_name", "model_path", "api_endpoint"]
                }
            },
            "required": ["environment", "version"]
        }
        
        return {"mlops_config": schema}
        
    def _load_validation_rules(self) -> Dict[str, callable]:
        """Load custom validation rules"""
        def validate_model_config(config: MLOpsConfig) -> List[str]:
            errors = []
            
            if config.model.batch_size <= 0:
                errors.append("Model batch size must be positive")
                
            if config.model.max_sequence_length <= 0:
                errors.append("Max sequence length must be positive")
                
            if not (0 <= config.model.temperature <= 2):
                errors.append("Temperature must be between 0 and 2")
                
            return errors
            
        def validate_training_config(config: MLOpsConfig) -> List[str]:
            errors = []
            
            if config.training.epochs <= 0:
                errors.append("Training epochs must be positive")
                
            if config.training.learning_rate <= 0:
                errors.append("Learning rate must be positive")
                
            if config.training.batch_size <= 0:
                errors.append("Training batch size must be positive")
                
            return errors
            
        def validate_deployment_config(config: MLOpsConfig) -> List[str]:
            errors = []
            
            if config.deployment.replicas < 1:
                errors.append("Deployment replicas must be at least 1")
                
            if config.deployment.min_replicas > config.deployment.max_replicas:
                errors.append("Min replicas cannot be greater than max replicas")
                
            if not (1 <= config.deployment.target_cpu_utilization <= 100):
                errors.append("Target CPU utilization must be between 1 and 100")
                
            return errors
            
        return {
            "model_config": validate_model_config,
            "training_config": validate_training_config,
            "deployment_config": validate_deployment_config
        }

# Example usage and testing
def main():
    """Example usage of the MLOps configuration manager"""
    config_manager = MLOpsConfigManager()
    
    try:
        # Create a development configuration
        dev_config = config_manager.create_config(
            config_id="dev_config",
            environment=Environment.DEVELOPMENT
        )
        
        # Update configuration with specific settings
        updates = {
            "model": {
                "model_name": "bharat-gpt-7b",
                "model_type": "decoder",
                "model_version": "v1.0",
                "batch_size": 16,
                "max_sequence_length": 1024
            },
            "training": {
                "experiment_name": "bharat-gpt-finetune",
                "dataset_path": "/data/indic-dataset",
                "output_dir": "/models/output",
                "epochs": 5,
                "learning_rate": 2e-5,
                "batch_size": 16
            },
            "deployment": {
                "deployment_name": "bharat-gpt-api",
                "model_path": "/models/bharat-gpt-7b",
                "api_endpoint": "/api/v1/chat",
                "replicas": 1
            }
        }
        
        updated_config = config_manager.update_config("dev_config", updates)
        
        # Save configuration
        config_manager.save_config("dev_config", "development_config.yaml")
        
        # List configurations
        configs = config_manager.list_configs()
        print(f"Available configurations: {configs}")
        
        # Get configuration
        active_config = config_manager.get_config()
        print(f"Active config environment: {active_config.environment.value}")
        print(f"Model name: {active_config.model.model_name}")
        print(f"Training epochs: {active_config.training.epochs}")
        
        # Export configuration
        yaml_config = config_manager.export_config("dev_config", "yaml")
        print(f"Configuration exported as YAML (first 200 chars): {yaml_config[:200]}...")
        
        # Validate configuration
        validation_errors = config_manager.validate_config("dev_config")
        if validation_errors:
            print(f"Validation errors: {validation_errors}")
        else:
            print("Configuration is valid")
            
    except Exception as e:
        logger.error(f"Error in configuration management example: {e}")

if __name__ == "__main__":
    main()