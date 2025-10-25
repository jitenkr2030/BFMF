"""
Edge AI Module for Bharat-FM
Provides on-device inference capabilities for edge computing
"""

from .edge_inference import (
    EdgeDeviceConfig,
    ModelOptimizationConfig,
    EdgeModel,
    MobileNetEdgeModel,
    TinyMLEdgeModel,
    EdgeInferenceEngine,
    EdgeModelManager,
    EdgeOptimizer,
    create_edge_inference_engine,
    create_optimization_config
)

__all__ = [
    'EdgeDeviceConfig',
    'ModelOptimizationConfig',
    'EdgeModel',
    'MobileNetEdgeModel',
    'TinyMLEdgeModel',
    'EdgeInferenceEngine',
    'EdgeModelManager',
    'EdgeOptimizer',
    'create_edge_inference_engine',
    'create_optimization_config'
]