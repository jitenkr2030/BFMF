"""
Configuration module for Bharat-FM
Centralized configuration management for all components
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default inference optimization configuration
INFERENCE_CONFIG = {
    "optimization_enabled": True,
    "caching_enabled": True,
    "batching_enabled": True,
    "cost_monitoring_enabled": True,
    "max_cache_entries": 10000,
    "max_batch_size": 8,
    "max_wait_time": 0.1,
    "cache_dir": "./cache",
    "similarity_threshold": 0.95,
    "cache_ttl": 3600,  # 1 hour in seconds
    "batching_strategy": "adaptive",
    "cost_budget": 100.0,  # $100 daily budget
    "cost_alert_threshold": 0.8,  # Alert at 80% of budget
    "performance_alert_threshold": 0.9,  # Alert at 90% of max latency
    "model_selection_weights": {
        "latency": 0.4,
        "cost": 0.3,
        "accuracy": 0.2,
        "throughput": 0.1
    }
}

# Default conversation memory configuration
MEMORY_CONFIG = {
    "memory_dir": "./conversation_memory",
    "max_history_length": 1000,
    "max_sessions_per_user": 10,
    "context_retention_days": 30,
    "auto_cleanup_enabled": True,
    "cleanup_interval_hours": 24,
    "compression_enabled": True,
    "semantic_search_enabled": True,
    "personalization_enabled": True,
    "emotional_intelligence_enabled": True,
    "topic_extraction_enabled": True,
    "embedding_model": "hash_based",  # Placeholder for real embeddings
    "sentiment_model": "simple",  # Placeholder for real sentiment analysis
    "max_memory_size_mb": 1000,  # 1GB max memory usage
    "backup_enabled": True,
    "backup_interval_hours": 6
}

# Default model configuration
MODEL_CONFIG = {
    "default_model": "bharat-gpt-small",
    "available_models": {
        "bharat-gpt-small": {
            "name": "Bharat GPT Small",
            "description": "Lightweight model for fast inference",
            "latency_ms": 100,
            "cost_per_1k_tokens": 0.001,
            "accuracy": 0.85,
            "max_tokens": 2048,
            "supports_streaming": True
        },
        "bharat-gpt-medium": {
            "name": "Bharat GPT Medium",
            "description": "Balanced model for general use",
            "latency_ms": 300,
            "cost_per_1k_tokens": 0.003,
            "accuracy": 0.92,
            "max_tokens": 4096,
            "supports_streaming": True
        },
        "bharat-gpt-large": {
            "name": "Bharat GPT Large",
            "description": "High-performance model for complex tasks",
            "latency_ms": 800,
            "cost_per_1k_tokens": 0.008,
            "accuracy": 0.96,
            "max_tokens": 8192,
            "supports_streaming": True
        }
    }
}

# Default logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "./logs/bharat_fm.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        },
        "bharat_fm": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Default API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "cors_enabled": True,
    "cors_origins": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    },
    "authentication": {
        "enabled": False,
        "type": "api_key",
        "api_keys": []
    }
}

# Complete default configuration
DEFAULT_CONFIG = {
    "inference": INFERENCE_CONFIG,
    "memory": MEMORY_CONFIG,
    "models": MODEL_CONFIG,
    "logging": LOGGING_CONFIG,
    "api": API_CONFIG,
    "environment": os.getenv("BHARAT_FM_ENV", "development"),
    "version": "1.0.0",
    "debug": os.getenv("BHARAT_FM_DEBUG", "false").lower() == "true"
}

class ConfigManager:
    """Configuration manager for Bharat-FM"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Override with environment variables
        self.load_from_environment()
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            import json
            
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Deep merge configuration
            self._deep_merge(self.config, file_config)
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Inference configuration
        if os.getenv("BHARAT_FM_OPTIMIZATION_ENABLED"):
            self.config["inference"]["optimization_enabled"] = (
                os.getenv("BHARAT_FM_OPTIMIZATION_ENABLED").lower() == "true"
            )
        
        if os.getenv("BHARAT_FM_CACHE_ENABLED"):
            self.config["inference"]["caching_enabled"] = (
                os.getenv("BHARAT_FM_CACHE_ENABLED").lower() == "true"
            )
        
        if os.getenv("BHARAT_FM_BATCHING_ENABLED"):
            self.config["inference"]["batching_enabled"] = (
                os.getenv("BHARAT_FM_BATCHING_ENABLED").lower() == "true"
            )
        
        if os.getenv("BHARAT_FM_MAX_CACHE_ENTRIES"):
            self.config["inference"]["max_cache_entries"] = int(
                os.getenv("BHARAT_FM_MAX_CACHE_ENTRIES")
            )
        
        if os.getenv("BHARAT_FM_MAX_BATCH_SIZE"):
            self.config["inference"]["max_batch_size"] = int(
                os.getenv("BHARAT_FM_MAX_BATCH_SIZE")
            )
        
        if os.getenv("BHARAT_FM_CACHE_DIR"):
            self.config["inference"]["cache_dir"] = os.getenv("BHARAT_FM_CACHE_DIR")
        
        # Memory configuration
        if os.getenv("BHARAT_FM_MEMORY_DIR"):
            self.config["memory"]["memory_dir"] = os.getenv("BHARAT_FM_MEMORY_DIR")
        
        if os.getenv("BHARAT_FM_MAX_HISTORY_LENGTH"):
            self.config["memory"]["max_history_length"] = int(
                os.getenv("BHARAT_FM_MAX_HISTORY_LENGTH")
            )
        
        if os.getenv("BHARAT_FM_CONTEXT_RETENTION_DAYS"):
            self.config["memory"]["context_retention_days"] = int(
                os.getenv("BHARAT_FM_CONTEXT_RETENTION_DAYS")
            )
        
        # API configuration
        if os.getenv("BHARAT_FM_API_HOST"):
            self.config["api"]["host"] = os.getenv("BHARAT_FM_API_HOST")
        
        if os.getenv("BHARAT_FM_API_PORT"):
            self.config["api"]["port"] = int(os.getenv("BHARAT_FM_API_PORT"))
        
        if os.getenv("BHARAT_FM_API_DEBUG"):
            self.config["api"]["debug"] = (
                os.getenv("BHARAT_FM_API_DEBUG").lower() == "true"
            )
        
        logger.info("Environment configuration loaded")
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config.copy()
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.config["inference"].copy()
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration"""
        return self.config["memory"].copy()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config["models"].copy()
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.config["api"].copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._deep_merge(self.config, updates)
        logger.info(f"Configuration updated: {list(updates.keys())}")
    
    def save_config(self, config_path: str):
        """Save current configuration to file"""
        try:
            import json
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Validate inference config
            inference = self.config["inference"]
            assert isinstance(inference["max_cache_entries"], int) and inference["max_cache_entries"] > 0
            assert isinstance(inference["max_batch_size"], int) and inference["max_batch_size"] > 0
            assert isinstance(inference["max_wait_time"], (int, float)) and inference["max_wait_time"] > 0
            
            # Validate memory config
            memory = self.config["memory"]
            assert isinstance(memory["max_history_length"], int) and memory["max_history_length"] > 0
            assert isinstance(memory["max_sessions_per_user"], int) and memory["max_sessions_per_user"] > 0
            assert isinstance(memory["context_retention_days"], int) and memory["context_retention_days"] > 0
            
            # Validate API config
            api = self.config["api"]
            assert isinstance(api["port"], int) and 1 <= api["port"] <= 65535
            assert isinstance(api["host"], str)
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_inference_config() -> Dict[str, Any]:
    """Get inference configuration"""
    return get_config().get_inference_config()

def get_memory_config() -> Dict[str, Any]:
    """Get memory configuration"""
    return get_config().get_memory_config()

def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return get_config().get_model_config()

def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return get_config().get_api_config()

def initialize_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager