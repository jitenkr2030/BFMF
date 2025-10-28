"""
Bharat Foundation Model Framework (Bharat-FM)
A comprehensive AI framework with enhanced inference optimization and conversation memory
"""

__version__ = "1.0.0"
__author__ = "Bharat AI Team"
__email__ = "team@bharatai.org"
__license__ = "Apache 2.0 / Bharat Open AI License (BOAL)"
__description__ = "Bharat Foundation Model Framework - Advanced AI with optimization and memory"

# Import core components
from .core.inference_engine import InferenceEngine, create_inference_engine
from .core.chat_engine import ChatEngine

# Import memory components
from .memory.conversation_memory import ConversationMemoryManager

# Import optimization components
from .optimization.inference_optimizer import InferenceOptimizer
from .optimization.semantic_cache import SemanticCache
from .optimization.dynamic_batcher import DynamicBatcher
from .optimization.cost_monitor import CostMonitor
from .optimization.model_selector import ModelSelector
from .optimization.performance_tracker import PerformanceTracker

# Import advanced AI engines
from .advanced_ai_engines import (
    AdvancedReasoningEngine,
    EmotionalIntelligenceEngine,
    CausalReasoningEngine,
    MultimodalEngine,
    CreativeThinkingEngine,
    AdvancedNLUEngine,
    MetacognitionEngine,
    SelfLearningSystem
)

# Import MLOps components
from .mlops import (
    MLOpsConfig,
    ErrorRateMonitoring,
    AutomatedAlertingSystems,
    ResourceUtilizationTracking,
    RealtimeMonitoring,
    PerformanceMonitoringDashboards,
    ModelDriftDetection,
    DeploymentMonitoringHealthChecks,
    MLOpsDeploymentInfrastructure,
    CanaryDeployments,
    ABTestingInfrastructure,
    PipelineOrchestration
)

# Import configuration
from .config import get_config, get_inference_config, get_memory_config, get_model_config

# Export main classes and functions
__all__ = [
    # Core engines
    "InferenceEngine",
    "ChatEngine",
    "create_inference_engine",
    
    # Memory management
    "ConversationMemoryManager",
    
    # Optimization components
    "InferenceOptimizer",
    "SemanticCache",
    "DynamicBatcher",
    "CostMonitor",
    "ModelSelector",
    "PerformanceTracker",
    
    # Advanced AI engines
    "AdvancedReasoningEngine",
    "EmotionalIntelligenceEngine",
    "CausalReasoningEngine",
    "MultimodalEngine",
    "CreativeThinkingEngine",
    "AdvancedNLUEngine",
    "MetacognitionEngine",
    "SelfLearningSystem",
    
    # MLOps components
    "MLOpsConfig",
    "ErrorRateMonitoring",
    "AutomatedAlertingSystems",
    "ResourceUtilizationTracking",
    "RealtimeMonitoring",
    "PerformanceMonitoringDashboards",
    "ModelDriftDetection",
    "DeploymentMonitoringHealthChecks",
    "MLOpsDeploymentInfrastructure",
    "CanaryDeployments",
    "ABTestingInfrastructure",
    "PipelineOrchestration",
    
    # Configuration
    "get_config",
    "get_inference_config",
    "get_memory_config",
    "get_model_config",
]

# Package-level information
PACKAGE_INFO = {
    "name": "bharat-fm",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "license": __license__,
    "homepage": "https://github.com/bharatai/bharat-fm",
    "documentation": "https://bharat-fm.readthedocs.io",
    "repository": "https://github.com/bharatai/bharat-fm",
    "bugs": "https://github.com/bharatai/bharat-fm/issues"
}

# Phase 1 capabilities
PHASE1_CAPABILITIES = {
    "real_time_inference_optimization": {
        "semantic_caching": "Intelligent caching based on semantic similarity",
        "dynamic_batching": "Adaptive request batching for optimal throughput",
        "cost_monitoring": "Real-time cost tracking and optimization",
        "model_selection": "Intelligent model selection based on requirements",
        "performance_tracking": "Comprehensive performance monitoring and analytics"
    },
    "conversation_memory_management": {
        "multi_session_context": "Context management across multiple sessions",
        "personalization_profiles": "User preference and communication style learning",
        "emotional_intelligence": "Sentiment analysis and emotion tracking",
        "semantic_search": "Vector-based conversation history search",
        "topic_extraction": "Automatic topic identification and tracking",
        "memory_management": "Efficient storage with automatic cleanup"
    }
}

def get_package_info() -> dict:
    """Get package information"""
    return PACKAGE_INFO.copy()

def get_capabilities() -> dict:
    """Get current phase capabilities"""
    return PHASE1_CAPABILITIES.copy()

def get_version() -> str:
    """Get package version"""
    return __version__

def get_supported_features() -> list:
    """Get list of supported features"""
    features = []
    for category, capabilities in PHASE1_CAPABILITIES.items():
        for feature, description in capabilities.items():
            features.append({
                "category": category,
                "feature": feature,
                "description": description
            })
    return features

def print_welcome_message():
    """Print welcome message with package information"""
    print(f"""
🇮🇳  Welcome to Bharat Foundation Model Framework (Bharat-FM) {__version__}
{'=' * 70}
{__description__}

📦 Package: {PACKAGE_INFO['name']}
🏠 Homepage: {PACKAGE_INFO['homepage']}
📚 Documentation: {PACKAGE_INFO['documentation']}
🔧 License: {PACKAGE_INFO['license']}

🚀 Phase 1 Capabilities:
""")
    
    for category, capabilities in PHASE1_CAPABILITIES.items():
        print(f"\n  {category.replace('_', ' ').title()}:")
        for feature, description in capabilities.items():
            print(f"    • {feature.replace('_', ' ').title()}: {description}")
    
    print(f"\n{'=' * 70}")
    print("🇮🇳 Made with ❤️ for Bharat's AI Independence")
    print("=" * 70)

# Auto-print welcome message when imported directly
if __name__ == "__main__":
    print_welcome_message()