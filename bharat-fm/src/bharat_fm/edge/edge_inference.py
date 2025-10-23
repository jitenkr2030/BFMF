"""
Edge AI Module for Bharat-FM
Implements on-device inference capabilities for edge computing
"""

import numpy as np
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EdgeDeviceConfig:
    """Configuration for edge device capabilities"""
    device_type: str = "mobile"  # mobile, embedded, iot, edge_server
    compute_capability: float = 1.0  # Relative compute power
    memory_mb: int = 4096
    storage_mb: int = 16384
    battery_constrained: bool = True
    network_latency_ms: int = 50
    supports_gpu: bool = False
    supports_npu: bool = False

@dataclass
class ModelOptimizationConfig:
    """Configuration for model optimization"""
    quantization_bits: int = 8  # 8-bit quantization
    pruning_ratio: float = 0.5  # 50% pruning
    knowledge_distillation: bool = True
    model_compression: bool = True
    dynamic_batching: bool = True

class EdgeModel(ABC):
    """Abstract base class for edge-optimized models"""
    
    def __init__(self, model_name: str, config: ModelOptimizationConfig):
        self.model_name = model_name
        self.config = config
        self.model_size_mb = 0
        self.inference_time_ms = 0
        self.accuracy = 0.0
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model from file"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make prediction with loaded model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information"""
        pass
    
    def optimize_for_edge(self, original_model: Any) -> Any:
        """Optimize model for edge deployment"""
        optimized_model = original_model
        
        if self.config.quantization_bits < 32:
            optimized_model = self._quantize_model(optimized_model)
        
        if self.config.pruning_ratio > 0:
            optimized_model = self._prune_model(optimized_model)
        
        if self.config.model_compression:
            optimized_model = self._compress_model(optimized_model)
        
        return optimized_model
    
    def _quantize_model(self, model: Any) -> Any:
        """Quantize model to reduce precision"""
        # Simulate quantization
        logger.info(f"Quantizing model to {self.config.quantization_bits}-bit")
        return model
    
    def _prune_model(self, model: Any) -> Any:
        """Prune model to remove redundant parameters"""
        # Simulate pruning
        logger.info(f"Pruning {self.config.pruning_ratio*100}% of model parameters")
        return model
    
    def _compress_model(self, model: Any) -> Any:
        """Compress model for smaller footprint"""
        # Simulate compression
        logger.info("Compressing model for edge deployment")
        return model

class MobileNetEdgeModel(EdgeModel):
    """MobileNet-optimized model for mobile devices"""
    
    def __init__(self, config: ModelOptimizationConfig):
        super().__init__("MobileNet", config)
        self.model_architecture = "depthwise_separable"
    
    def load_model(self, model_path: str) -> bool:
        """Load MobileNet model"""
        try:
            # Simulate loading model
            self.model_size_mb = 14.2 * (self.config.quantization_bits / 32.0)
            self.inference_time_ms = 25 / self.device_config.compute_capability
            self.accuracy = 0.92 - (0.05 * (1 - self.config.quantization_bits / 32.0))
            self.is_loaded = True
            
            logger.info(f"Loaded MobileNet model: {self.model_size_mb:.1f}MB, "
                       f"{self.inference_time_ms:.1f}ms, {self.accuracy:.3f} accuracy")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MobileNet model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict:
        """Make prediction with MobileNet"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Simulate inference
        start_time = time.time()
        
        # Simulate prediction
        num_classes = 1000
        logits = np.random.randn(num_classes)
        probabilities = 1 / (1 + np.exp(-logits))
        probabilities = probabilities / np.sum(probabilities)
        
        # Get top predictions
        top_k = 5
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                "class_id": int(idx),
                "confidence": float(probabilities[idx]),
                "class_name": f"class_{idx}"
            })
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": predictions,
            "inference_time_ms": inference_time,
            "model_name": self.model_name,
            "device_type": self.device_config.device_type
        }
    
    def get_model_info(self) -> Dict:
        """Get MobileNet model information"""
        return {
            "model_name": self.model_name,
            "architecture": self.model_architecture,
            "size_mb": self.model_size_mb,
            "inference_time_ms": self.inference_time_ms,
            "accuracy": self.accuracy,
            "quantization_bits": self.config.quantization_bits,
            "pruning_ratio": self.config.pruning_ratio,
            "is_loaded": self.is_loaded
        }

class TinyMLEdgeModel(EdgeModel):
    """TinyML-optimized model for microcontrollers"""
    
    def __init__(self, config: ModelOptimizationConfig):
        super().__init__("TinyML", config)
        self.model_architecture = "micro_neural_network"
    
    def load_model(self, model_path: str) -> bool:
        """Load TinyML model"""
        try:
            # Simulate loading ultra-lightweight model
            self.model_size_mb = 0.5 * (self.config.quantization_bits / 32.0)
            self.inference_time_ms = 5 / self.device_config.compute_capability
            self.accuracy = 0.85 - (0.08 * (1 - self.config.quantization_bits / 32.0))
            self.is_loaded = True
            
            logger.info(f"Loaded TinyML model: {self.model_size_mb:.1f}MB, "
                       f"{self.inference_time_ms:.1f}ms, {self.accuracy:.3f} accuracy")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TinyML model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict:
        """Make prediction with TinyML"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Simulate ultra-fast inference
        start_time = time.time()
        
        # Simple neural network simulation
        flattened_input = input_data.flatten()
        hidden_layer = np.tanh(np.dot(flattened_input, np.random.randn(len(flattened_input), 16)))
        output = np.dot(hidden_layer, np.random.randn(16, 10))
        probabilities = 1 / (1 + np.exp(-output))
        probabilities = probabilities / np.sum(probabilities)
        
        prediction = np.argmax(probabilities)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "prediction": int(prediction),
            "confidence": float(probabilities[prediction]),
            "inference_time_ms": inference_time,
            "model_name": self.model_name,
            "device_type": self.device_config.device_type
        }
    
    def get_model_info(self) -> Dict:
        """Get TinyML model information"""
        return {
            "model_name": self.model_name,
            "architecture": self.model_architecture,
            "size_mb": self.model_size_mb,
            "inference_time_ms": self.inference_time_ms,
            "accuracy": self.accuracy,
            "quantization_bits": self.config.quantization_bits,
            "pruning_ratio": self.config.pruning_ratio,
            "is_loaded": self.is_loaded
        }

class EdgeInferenceEngine:
    """Main inference engine for edge devices"""
    
    def __init__(self, device_config: EdgeDeviceConfig):
        self.device_config = device_config
        self.loaded_models = {}
        self.model_registry = {}
        self.performance_metrics = {
            "total_inferences": 0,
            "average_latency_ms": 0,
            "battery_usage_mah": 0,
            "memory_usage_mb": 0
        }
    
    def register_model(self, model_name: str, model_class: type, 
                      optimization_config: ModelOptimizationConfig = None):
        """Register a model class for edge deployment"""
        if optimization_config is None:
            optimization_config = ModelOptimizationConfig()
        
        self.model_registry[model_name] = {
            "class": model_class,
            "config": optimization_config
        }
        
        logger.info(f"Registered model: {model_name}")
    
    def load_model(self, model_name: str, model_path: str = None) -> bool:
        """Load a model for inference"""
        if model_name not in self.model_registry:
            logger.error(f"Model {model_name} not registered")
            return False
        
        try:
            model_info = self.model_registry[model_name]
            model_class = model_info["class"]
            config = model_info["config"]
            
            # Create model instance
            model = model_class(config)
            model.device_config = self.device_config
            
            # Load model
            if model_path and os.path.exists(model_path):
                success = model.load_model(model_path)
            else:
                success = model.load_model("simulated")
            
            if success:
                self.loaded_models[model_name] = model
                self.performance_metrics["memory_usage_mb"] += model.model_size_mb
                logger.info(f"Successfully loaded model: {model_name}")
                return True
            else:
                logger.error(f"Failed to load model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, model_name: str, input_data: Any) -> Dict:
        """Make prediction with loaded model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.loaded_models[model_name]
        
        # Perform inference
        result = model.predict(input_data)
        
        # Update performance metrics
        self.performance_metrics["total_inferences"] += 1
        self.performance_metrics["average_latency_ms"] = (
            (self.performance_metrics["average_latency_ms"] * 
             (self.performance_metrics["total_inferences"] - 1) + 
             result["inference_time_ms"]) / self.performance_metrics["total_inferences"]
        )
        
        # Estimate battery usage (simplified)
        if self.device_config.battery_constrained:
            battery_usage = result["inference_time_ms"] * 0.001  # mAh per ms
            self.performance_metrics["battery_usage_mah"] += battery_usage
        
        return result
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            self.performance_metrics["memory_usage_mb"] -= model.model_size_mb
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_loaded_models_info(self) -> List[Dict]:
        """Get information about all loaded models"""
        return [model.get_model_info() for model in self.loaded_models.values()]

class EdgeModelManager:
    """Manages model lifecycle and deployment on edge devices"""
    
    def __init__(self, inference_engine: EdgeInferenceEngine):
        self.inference_engine = inference_engine
        self.model_versions = {}
        self.deployment_history = []
    
    def deploy_model(self, model_name: str, version: str, 
                    model_path: str, optimization_config: ModelOptimizationConfig = None) -> bool:
        """Deploy a model version to edge device"""
        try:
            # Load the model
            success = self.inference_engine.load_model(model_name, model_path)
            
            if success:
                # Record deployment
                deployment_info = {
                    "model_name": model_name,
                    "version": version,
                    "timestamp": time.time(),
                    "optimization_config": asdict(optimization_config) if optimization_config else None,
                    "status": "deployed"
                }
                
                self.model_versions[model_name] = version
                self.deployment_history.append(deployment_info)
                
                logger.info(f"Deployed model {model_name} version {version}")
                return True
            else:
                logger.error(f"Failed to deploy model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying model {model_name}: {e}")
            return False
    
    def rollback_model(self, model_name: str, target_version: str = None) -> bool:
        """Rollback to previous model version"""
        if target_version is None:
            # Find previous version
            for deployment in reversed(self.deployment_history):
                if deployment["model_name"] == model_name and deployment["version"] != self.model_versions.get(model_name):
                    target_version = deployment["version"]
                    break
        
        if target_version:
            # Simulate rollback
            logger.info(f"Rolling back {model_name} to version {target_version}")
            self.model_versions[model_name] = target_version
            return True
        
        return False
    
    def get_deployment_history(self, model_name: str = None) -> List[Dict]:
        """Get deployment history"""
        if model_name:
            return [d for d in self.deployment_history if d["model_name"] == model_name]
        return self.deployment_history.copy()

class EdgeOptimizer:
    """Optimizes model performance for edge devices"""
    
    def __init__(self, device_config: EdgeDeviceConfig):
        self.device_config = device_config
    
    def optimize_model_pipeline(self, model_path: str, 
                               optimization_config: ModelOptimizationConfig) -> Dict:
        """Run complete model optimization pipeline"""
        optimization_steps = []
        
        # Step 1: Quantization
        if optimization_config.quantization_bits < 32:
            quantization_result = self._quantize_model(model_path, optimization_config.quantization_bits)
            optimization_steps.append(quantization_result)
        
        # Step 2: Pruning
        if optimization_config.pruning_ratio > 0:
            pruning_result = self._prune_model(model_path, optimization_config.pruning_ratio)
            optimization_steps.append(pruning_result)
        
        # Step 3: Knowledge Distillation
        if optimization_config.knowledge_distillation:
            distillation_result = self._knowledge_distillation(model_path)
            optimization_steps.append(distillation_result)
        
        # Step 4: Compression
        if optimization_config.model_compression:
            compression_result = self._compress_model(model_path)
            optimization_steps.append(compression_result)
        
        return {
            "optimization_steps": optimization_steps,
            "total_size_reduction": self._calculate_size_reduction(optimization_steps),
            "estimated_speedup": self._calculate_speedup(optimization_steps)
        }
    
    def _quantize_model(self, model_path: str, bits: int) -> Dict:
        """Quantize model to specified bit width"""
        return {
            "step": "quantization",
            "bits": bits,
            "size_reduction": 0.75,  # 75% reduction for 8-bit
            "accuracy_impact": -0.02
        }
    
    def _prune_model(self, model_path: str, ratio: float) -> Dict:
        """Prune model parameters"""
        return {
            "step": "pruning",
            "ratio": ratio,
            "size_reduction": ratio * 0.8,
            "accuracy_impact": -ratio * 0.1
        }
    
    def _knowledge_distillation(self, model_path: str) -> Dict:
        """Apply knowledge distillation"""
        return {
            "step": "knowledge_distillation",
            "size_reduction": 0.6,
            "accuracy_impact": -0.03
        }
    
    def _compress_model(self, model_path: str) -> Dict:
        """Compress model file"""
        return {
            "step": "compression",
            "size_reduction": 0.3,
            "accuracy_impact": 0.0
        }
    
    def _calculate_size_reduction(self, steps: List[Dict]) -> float:
        """Calculate total size reduction"""
        total_reduction = 1.0
        for step in steps:
            total_reduction *= (1 - step["size_reduction"])
        return 1 - total_reduction
    
    def _calculate_speedup(self, steps: List[Dict]) -> float:
        """Calculate estimated speedup"""
        # Simplified speedup calculation
        speedup = 1.0
        for step in steps:
            if step["step"] == "quantization":
                speedup *= 2.0
            elif step["step"] == "pruning":
                speedup *= 1.5
        return speedup

# Factory functions
def create_edge_inference_engine(device_type: str = "mobile", 
                                compute_capability: float = 1.0) -> EdgeInferenceEngine:
    """Create edge inference engine with default configuration"""
    device_config = EdgeDeviceConfig(
        device_type=device_type,
        compute_capability=compute_capability
    )
    
    engine = EdgeInferenceEngine(device_config)
    
    # Register default models
    engine.register_model("mobilenet", MobileNetEdgeModel)
    engine.register_model("tinyml", TinyMLEdgeModel)
    
    return engine

def create_optimization_config(quantization_bits: int = 8, 
                              pruning_ratio: float = 0.5) -> ModelOptimizationConfig:
    """Create model optimization configuration"""
    return ModelOptimizationConfig(
        quantization_bits=quantization_bits,
        pruning_ratio=pruning_ratio
    )

# Example usage and testing
def test_edge_inference():
    """Test edge inference functionality"""
    print("Testing Edge Inference...")
    
    # Create edge inference engine
    engine = create_edge_inference_engine(device_type="mobile", compute_capability=1.0)
    
    # Create model manager
    model_manager = EdgeModelManager(engine)
    
    # Test model deployment
    print("Testing model deployment...")
    optimization_config = create_optimization_config(quantization_bits=8, pruning_ratio=0.5)
    
    success = model_manager.deploy_model("mobilenet", "v1.0", "simulated_path", optimization_config)
    print(f"MobileNet deployment: {'Success' if success else 'Failed'}")
    
    success = model_manager.deploy_model("tinyml", "v1.0", "simulated_path", optimization_config)
    print(f"TinyML deployment: {'Success' if success else 'Failed'}")
    
    # Test inference
    print("Testing inference...")
    
    # Test input data
    test_input = np.random.randn(224, 224, 3)  # Image-like input
    
    # MobileNet inference
    try:
        result = engine.predict("mobilenet", test_input)
        print(f"MobileNet inference: {result['inference_time_ms']:.1f}ms, "
              f"top class: {result['predictions'][0]['class_name']}")
    except Exception as e:
        print(f"MobileNet inference failed: {e}")
    
    # TinyML inference
    try:
        result = engine.predict("tinyml", test_input)
        print(f"TinyML inference: {result['inference_time_ms']:.1f}ms, "
              f"prediction: {result['prediction']}")
    except Exception as e:
        print(f"TinyML inference failed: {e}")
    
    # Test performance metrics
    print("Testing performance metrics...")
    metrics = engine.get_performance_metrics()
    print(f"Total inferences: {metrics['total_inferences']}")
    print(f"Average latency: {metrics['average_latency_ms']:.1f}ms")
    print(f"Memory usage: {metrics['memory_usage_mb']:.1f}MB")
    print(f"Battery usage: {metrics['battery_usage_mah']:.2f}mAh")
    
    # Test model optimization
    print("Testing model optimization...")
    optimizer = EdgeOptimizer(engine.device_config)
    
    optimization_result = optimizer.optimize_model_pipeline(
        "simulated_model", 
        optimization_config
    )
    
    print(f"Size reduction: {optimization_result['total_size_reduction']*100:.1f}%")
    print(f"Estimated speedup: {optimization_result['estimated_speedup']:.1f}x")
    
    # Test model unloading
    print("Testing model unloading...")
    engine.unload_model("mobilenet")
    metrics = engine.get_performance_metrics()
    print(f"Memory usage after unloading: {metrics['memory_usage_mb']:.1f}MB")
    
    print("Edge inference tests completed!")

if __name__ == "__main__":
    test_edge_inference()