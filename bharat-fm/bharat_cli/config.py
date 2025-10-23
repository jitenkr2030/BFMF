"""
Configuration management for BharatFM CLI
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import typer


class CLIConfig:
    """Configuration management for BharatFM CLI"""
    
    def __init__(self):
        self.config_file = Path.home() / ".bharatfm" / "config.json"
        self.config_dir = Path.home() / ".bharatfm"
        
        # Default configuration
        self.defaults = {
            "default_model": "bharat-base",
            "default_dataset": "indic_mix",
            "default_output_dir": "./outputs",
            "default_host": "0.0.0.0",
            "default_port": 8000,
            "default_engine": "vllm",
            "log_level": "INFO",
            "parallel_jobs": 4,
            "cache_dir": str(Path.home() / ".bharatfm" / "cache"),
            "hub_organization": "bharat-ai",
            "tracking_uri": "file:///tmp/mlruns",
            "gpu_memory_utilization": 0.9,
            "max_batch_size": 32,
            "max_context_length": 2048
        }
        
        # Load configuration
        self.config = self.load()
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return self.defaults.copy()
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # Merge with defaults
            merged_config = self.defaults.copy()
            merged_config.update(config)
            
            return merged_config
            
        except Exception as e:
            typer.echo(f"Warning: Could not load config file: {e}")
            return self.defaults.copy()
            
    def save(self):
        """Save configuration to file"""
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            typer.echo(f"Warning: Could not save config file: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default or self.defaults.get(key))
        
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        self.save()
        
    def delete(self, key: str):
        """Delete configuration key"""
        if key in self.config:
            del self.config[key]
            self.save()
            
    def show(self):
        """Show current configuration"""
        typer.echo("BharatFM Configuration:")
        typer.echo("=" * 40)
        
        for key, value in sorted(self.config.items()):
            typer.echo(f"{key:<25}: {value}")
            
    def reset(self):
        """Reset configuration to defaults"""
        self.config = self.defaults.copy()
        self.save()
        typer.echo("Configuration reset to defaults")
        
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        model_configs = {
            "bharat-base": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "intermediate_size": 16384,
                "max_position_embeddings": 2048
            },
            "bharat-lite": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 8192,
                "max_position_embeddings": 2048
            },
            "bharat-moe": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "intermediate_size": 16384,
                "max_position_embeddings": 2048,
                "num_experts": 8,
                "num_experts_per_token": 2
            }
        }
        
        return model_configs.get(model_name, {})
        
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset-specific configuration"""
        dataset_configs = {
            "indic_mix": {
                "languages": ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"],
                "max_length": 512,
                "preprocessing": ["clean", "normalize", "tokenize"]
            },
            "govt_data": {
                "languages": ["hi", "en"],
                "max_length": 1024,
                "domain": "government",
                "preprocessing": ["clean", "normalize", "tokenize", "filter"]
            },
            "edu_data": {
                "languages": ["hi", "en", "bn"],
                "max_length": 512,
                "domain": "education",
                "preprocessing": ["clean", "normalize", "tokenize"]
            }
        }
        
        return dataset_configs.get(dataset_name, {})
        
    def get_benchmark_config(self, benchmark_name: str) -> Dict[str, Any]:
        """Get benchmark-specific configuration"""
        benchmark_configs = {
            "perplexity": {
                "batch_size": 8,
                "max_length": 512,
                "iterations": 100
            },
            "generation_quality": {
                "batch_size": 4,
                "max_tokens": 100,
                "metrics": ["bleu", "rouge", "bertscore"]
            },
            "multilingual_accuracy": {
                "batch_size": 16,
                "languages": ["hi", "en", "bn", "ta", "te"],
                "test_size": 1000
            }
        }
        
        return benchmark_configs.get(benchmark_name, {})
        
    def get_deployment_config(self, deployment_type: str) -> Dict[str, Any]:
        """Get deployment-specific configuration"""
        deployment_configs = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "timeout": 30,
                "max_concurrent_requests": 100
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8001,
                "engine": "vllm",
                "max_batch_size": 32,
                "gpu_memory_utilization": 0.9
            },
            "streaming": {
                "host": "0.0.0.0",
                "port": 8002,
                "chunk_size": 10,
                "max_streaming_connections": 50
            }
        }
        
        return deployment_configs.get(deployment_type, {})
        
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = [
            "default_model",
            "default_dataset",
            "default_output_dir",
            "default_host",
            "default_port"
        ]
        
        for key in required_keys:
            if key not in self.config:
                typer.echo(f"Error: Missing required configuration key: {key}")
                return False
                
        # Validate paths
        if not isinstance(self.config["default_output_dir"], str):
            typer.echo("Error: default_output_dir must be a string")
            return False
            
        # Validate port
        if not isinstance(self.config["default_port"], int) or not (1 <= self.config["default_port"] <= 65535):
            typer.echo("Error: default_port must be an integer between 1 and 65535")
            return False
            
        return True
        
    def export_config(self, output_path: str):
        """Export configuration to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            typer.echo(f"Configuration exported to: {output_path}")
        except Exception as e:
            typer.echo(f"Error exporting configuration: {e}")
            
    def import_config(self, input_path: str):
        """Import configuration from file"""
        try:
            with open(input_path, 'r') as f:
                imported_config = json.load(f)
                
            # Validate imported config
            self.config.update(imported_config)
            
            if self.validate_config():
                self.save()
                typer.echo(f"Configuration imported from: {input_path}")
            else:
                typer.echo("Error: Invalid configuration in imported file")
                
        except Exception as e:
            typer.echo(f"Error importing configuration: {e}")
            
    def get_cache_dir(self) -> Path:
        """Get cache directory"""
        cache_dir = Path(self.get("cache_dir"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
        
    def clear_cache(self):
        """Clear cache directory"""
        cache_dir = self.get_cache_dir()
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            typer.echo("Cache cleared")
        else:
            typer.echo("No cache to clear")
            
    def get_env_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables"""
        env_config = {}
        
        # Map environment variables to config keys
        env_mapping = {
            "BHARATFM_MODEL": "default_model",
            "BHARATFM_DATASET": "default_dataset",
            "BHARATFM_OUTPUT_DIR": "default_output_dir",
            "BHARATFM_HOST": "default_host",
            "BHARATFM_PORT": "default_port",
            "BHARATFM_LOG_LEVEL": "log_level",
            "BHARATFM_HUB_TOKEN": "hub_token",
            "BHARATFM_TRACKING_URI": "tracking_uri"
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_key == "default_port":
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_key == "log_level":
                    value = value.upper()
                    
                env_config[config_key] = value
                
        return env_config
        
    def apply_env_config(self):
        """Apply environment configuration"""
        env_config = self.get_env_config()
        
        for key, value in env_config.items():
            self.set(key, value)
            
        if env_config:
            typer.echo(f"Applied {len(env_config)} environment variable(s)")