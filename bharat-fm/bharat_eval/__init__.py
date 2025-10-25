"""
Bharat Evaluation Module
=======================

Module for evaluation and benchmarking suite with support for
HELM, lm-eval-harness, and OpenCompass.
"""

from .evaluator import BharatEvaluator, EvaluationConfig
from .benchmarks import get_benchmark, BenchmarkRegistry

__version__ = "0.1.0"
__all__ = [
    "BharatEvaluator",
    "EvaluationConfig",
    "get_benchmark",
    "BenchmarkRegistry"
]