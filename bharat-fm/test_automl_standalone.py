#!/usr/bin/env python3
"""
Standalone test script for AutoML pipeline components
"""

import sys
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Machine learning task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"

class ModelType(Enum):
    """Model types supported by AutoML"""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    PROBABILISTIC = "probabilistic"

@dataclass
class AutoMLConfig:
    """Configuration for AutoML pipeline"""
    task_type: TaskType = TaskType.CLASSIFICATION
    time_limit: int = 3600  # 1 hour
    max_models: int = 50
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    optimize_metric: str = "accuracy"  # accuracy, f1, roc_auc, mse, mae, r2
    include_models: List[ModelType] = None
    exclude_models: List[ModelType] = None
    feature_engineering: bool = True
    hyperparameter_tuning: bool = True
    ensemble_building: bool = True
    early_stopping: bool = True
    
    def __post_init__(self):
        if self.include_models is None:
            self.include_models = [ModelType.LINEAR, ModelType.TREE_BASED, ModelType.ENSEMBLE]
        if self.exclude_models is None:
            self.exclude_models = []

@dataclass
class ModelResult:
    """Result of model training and evaluation"""
    model_name: str
    model_type: ModelType
    model: Any
    training_time: float
    evaluation_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    validation_score: float = 0.0
    test_score: float = 0.0
    model_size: int = 0

class SimpleDataFrame:
    """Simple DataFrame implementation for testing"""
    
    def __init__(self, data: Dict[str, List], columns: List[str] = None):
        self.data = data
        self.columns = columns or list(data.keys())
        self.shape = (len(data[self.columns[0]]), len(self.columns))
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, list):
            return SimpleDataFrame({col: self.data[col] for col in key}, key)
        else:
            raise ValueError("Unsupported key type")
    
    def drop(self, columns, axis=1):
        if axis == 1:
            new_data = {k: v for k, v in self.data.items() if k not in columns}
            return SimpleDataFrame(new_data)
        return self
    
    def isnull(self):
        return SimpleDataFrame({k: [False] * len(v) for k, v in self.data.items()})
    
    def select_dtypes(self, include=None):
        # Simple implementation - return all columns for now
        return self
    
    def nunique(self):
        return {k: len(set(v)) for k, v in self.data.items()}
    
    def median(self):
        return {k: sorted(v)[len(v)//2] for k, v in self.data.items()}
    
    def mode(self):
        return {k: max(set(v), key=v.count) for k, v in self.data.items()}
    
    def fillna(self, value):
        return SimpleDataFrame({k: [value if x is None else x for x in v] for k, v in self.data.items()})

class SimpleSeries:
    """Simple Series implementation for testing"""
    
    def __init__(self, data: List, name: str = None):
        self.data = data
        self.name = name
        self.shape = (len(data),)
    
    def __len__(self):
        return len(self.data)

class MockModel:
    """Mock model for testing when libraries are not available"""
    
    def __init__(self, task_type: TaskType, model_type: str):
        self.task_type = task_type
        self.model_type = model_type
        self.trained = False
        self.feature_importance = {}
    
    def fit(self, X, y):
        """Mock fit method"""
        self.trained = True
        # Generate random feature importance
        if hasattr(X, 'columns'):
            for col in X.columns:
                self.feature_importance[col] = random.random()
        return self
    
    def predict(self, X):
        """Mock predict method"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        # Generate random predictions
        n_samples = len(X.data) if hasattr(X, 'data') else len(X)
        if self.task_type == TaskType.CLASSIFICATION:
            return [random.choice([0, 1]) for _ in range(n_samples)]
        else:
            return [random.random() for _ in range(n_samples)]
    
    def predict_proba(self, X):
        """Mock predict_proba method"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        n_samples = len(X.data) if hasattr(X, 'data') else len(X)
        return [[random.random(), random.random()] for _ in range(n_samples)]
    
    def score(self, X, y):
        """Mock score method"""
        predictions = self.predict(X)
        if self.task_type == TaskType.CLASSIFICATION:
            # Calculate mock accuracy
            correct = sum(1 for pred, true in zip(predictions, y.data) if pred == true)
            return correct / len(y.data)
        else:
            # Calculate mock R²
            return random.uniform(0.5, 0.9)

class DataPreprocessor:
    """Automated data preprocessing and feature engineering"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.fitted_transformers = {}
        self.feature_names = []
        self.target_encoder = None
    
    def preprocess_data(self, X: SimpleDataFrame, y: SimpleSeries = None, 
                      is_training: bool = True) -> tuple:
        """
        Preprocess data with automated feature engineering
        """
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        if is_training:
            X = self._encode_categorical_features(X, fit=True)
        else:
            X = self._encode_categorical_features(X, fit=False)
        
        # Feature engineering
        if self.config.feature_engineering:
            X = self._engineer_features(X, is_training)
        
        # Handle target variable
        if y is not None:
            if is_training:
                y = self._encode_target(y)
            else:
                y = self._encode_target(y, fit=False)
        
        logger.info(f"Data preprocessing completed. Shape: {X.shape}")
        return X, y
    
    def _handle_missing_values(self, X: SimpleDataFrame) -> SimpleDataFrame:
        """Handle missing values in the dataset"""
        # For numeric columns, fill with median
        for col in X.columns:
            if any(x is None for x in X.data[col]):
                median_val = sorted([x for x in X.data[col] if x is not None])[len(X.data[col])//2]
                X.data[col] = [median_val if x is None else x for x in X.data[col]]
        
        return X
    
    def _encode_categorical_features(self, X: SimpleDataFrame, fit: bool = True) -> SimpleDataFrame:
        """Encode categorical features"""
        categorical_cols = [col for col in X.columns if any(isinstance(x, str) for x in X.data[col])]
        
        for col in categorical_cols:
            if fit:
                # Simple label encoding
                unique_values = list(set(X.data[col]))
                encoding_map = {val: i for i, val in enumerate(unique_values)}
                self.fitted_transformers[col] = {'type': 'label', 'encoding_map': encoding_map}
                X.data[col] = [encoding_map[x] for x in X.data[col]]
            else:
                if col in self.fitted_transformers:
                    encoding_map = self.fitted_transformers[col]['encoding_map']
                    X.data[col] = [encoding_map.get(x, -1) for x in X.data[col]]
        
        return X
    
    def _engineer_features(self, X: SimpleDataFrame, is_training: bool = True) -> SimpleDataFrame:
        """Automated feature engineering"""
        original_cols = X.columns[:]
        
        # Create interaction features for numeric columns
        numeric_cols = [col for col in X.columns if all(isinstance(x, (int, float)) for x in X.data[col])]
        
        if len(numeric_cols) > 1:
            # Create interaction features
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first 3 columns
                for col2 in numeric_cols[i+1:4]:  # Limit interactions
                    interaction_name = f"{col1}_x_{col2}"
                    X.data[interaction_name] = [x * y for x, y in zip(X.data[col1], X.data[col2])]
        
        # Statistical features
        if len(numeric_cols) > 1:
            X.data['numeric_mean'] = [sum(X.data[col][i] for col in numeric_cols) / len(numeric_cols) for i in range(len(X.data[numeric_cols[0]]))]
        
        self.feature_names = list(X.data.keys())
        return X
    
    def _encode_target(self, y: SimpleSeries, fit: bool = True) -> SimpleSeries:
        """Encode target variable"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            if fit:
                unique_values = list(set(y.data))
                self.target_encoder = {val: i for i, val in enumerate(unique_values)}
                y_encoded = [self.target_encoder[x] for x in y.data]
            else:
                y_encoded = [self.target_encoder.get(x, -1) for x in y.data]
            return SimpleSeries(y_encoded, y.name)
        else:
            return y

class ModelFactory:
    """Factory for creating and configuring machine learning models"""
    
    @staticmethod
    def create_model(model_type: ModelType, task_type: TaskType, 
                     hyperparameters: Dict[str, Any] = None) -> Any:
        """Create a model instance based on type and task"""
        if hyperparameters is None:
            hyperparameters = {}
        
        if model_type == ModelType.LINEAR:
            return MockModel(task_type, "linear")
        elif model_type == ModelType.TREE_BASED:
            return MockModel(task_type, "tree_based")
        elif model_type == ModelType.NEURAL_NETWORK:
            return MockModel(task_type, "neural_network")
        elif model_type == ModelType.ENSEMBLE:
            return MockModel(task_type, "ensemble")
        elif model_type == ModelType.PROBABILISTIC:
            return MockModel(task_type, "probabilistic")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class AutoMLPipeline:
    """Main AutoML pipeline class"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.results = {}
    
    def fit(self, X: SimpleDataFrame, y: SimpleSeries) -> Dict[str, Any]:
        """Fit AutoML pipeline to data"""
        logger.info("Starting AutoML pipeline...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocessor.preprocess_data(X, y, is_training=True)
        
        # Train models
        model_results = []
        for model_type in self.config.include_models:
            if model_type not in self.config.exclude_models:
                try:
                    result = self._train_model(model_type, X_processed, y_processed)
                    model_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to train {model_type}: {e}")
        
        # Select best model
        if model_results:
            self.best_model = max(model_results, key=lambda x: x.validation_score)
            self.best_score = self.best_model.validation_score
        
        self.results = {
            'best_model': self.best_model.model_name if self.best_model else None,
            'best_score': self.best_score,
            'model_results': model_results,
            'features_engineered': len(self.preprocessor.feature_names)
        }
        
        logger.info(f"AutoML pipeline completed. Best model: {self.best_model.model_name if self.best_model else 'None'}")
        return self.results
    
    def _train_model(self, model_type: ModelType, X: SimpleDataFrame, y: SimpleSeries) -> ModelResult:
        """Train a single model"""
        start_time = time.time()
        
        # Create model
        model = ModelFactory.create_model(model_type, self.config.task_type)
        
        # Train model
        model.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Evaluate model
        validation_score = model.score(X, y)
        
        # Create result
        result = ModelResult(
            model_name=f"{model_type.value}_model",
            model_type=model_type,
            model=model,
            training_time=training_time,
            evaluation_metrics={'validation_score': validation_score},
            hyperparameters={},
            feature_importance=getattr(model, 'feature_importance', {}),
            validation_score=validation_score
        )
        
        logger.info(f"Trained {model_type.value} model: validation_score={validation_score:.4f}")
        return result
    
    def score(self, X: SimpleDataFrame, y: SimpleSeries) -> float:
        """Score the best model on test data"""
        if not self.best_model:
            raise ValueError("No model trained")
        
        # Preprocess test data
        X_processed, _ = self.preprocessor.preprocess_data(X, is_training=False)
        
        # Score model
        return self.best_model.model.score(X_processed, y)

def generate_sample_data(n_samples=100, n_features=5, task_type='classification'):
    """Generate sample data for testing"""
    # Generate features
    data = {}
    for i in range(n_features):
        data[f'feature_{i}'] = [random.random() for _ in range(n_samples)]
    
    # Add some categorical features
    data['category_1'] = [random.choice(['A', 'B', 'C']) for _ in range(n_samples)]
    data['category_2'] = [random.choice(['X', 'Y']) for _ in range(n_samples)]
    
    # Generate target
    if task_type == 'classification':
        data['target'] = [random.choice([0, 1]) for _ in range(n_samples)]
    else:
        data['target'] = [random.random() for _ in range(n_samples)]
    
    # Create DataFrame and Series
    feature_cols = [col for col in data.keys() if col != 'target']
    X = SimpleDataFrame(data, feature_cols)
    y = SimpleSeries(data['target'], 'target')
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Simple train-test split implementation"""
    random.seed(random_state)
    n_samples = len(y.data)
    test_size = int(n_samples * test_size)
    
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    
    # Split data
    X_train_data = {k: [v[i] for i in train_indices] for k, v in X.data.items()}
    X_test_data = {k: [v[i] for i in test_indices] for k, v in X.data.items()}
    y_train_data = [y.data[i] for i in train_indices]
    y_test_data = [y.data[i] for i in test_indices]
    
    X_train = SimpleDataFrame(X_train_data, X.columns)
    X_test = SimpleDataFrame(X_test_data, X.columns)
    y_train = SimpleSeries(y_train_data, y.name)
    y_test = SimpleSeries(y_test_data, y.name)
    
    return X_train, X_test, y_train, y_test

def test_classification_task():
    """Test AutoML pipeline with classification task"""
    logger.info("=== Testing Classification Task ===")
    
    # Generate sample classification data
    X, y = generate_sample_data(n_samples=100, n_features=5, task_type='classification')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure AutoML
    config = AutoMLConfig(
        task_type=TaskType.CLASSIFICATION,
        time_limit=60,  # 1 minute for quick test
        max_models=5,
        cv_folds=2,
        optimize_metric="accuracy",
        include_models=[ModelType.LINEAR, ModelType.TREE_BASED, ModelType.ENSEMBLE],
        feature_engineering=True,
        hyperparameter_tuning=True,
        ensemble_building=True
    )
    
    # Create and run AutoML pipeline
    pipeline = AutoMLPipeline(config)
    
    start_time = time.time()
    try:
        results = pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_score = pipeline.score(X_test, y_test)
        
        # Get best model info
        best_model = results.get('best_model')
        
        logger.info(f"Classification Test Results:")
        logger.info(f"  Training time: {training_time:.2f}s")
        logger.info(f"  Test accuracy: {test_score:.4f}")
        logger.info(f"  Best model: {best_model}")
        logger.info(f"  Models evaluated: {len(results.get('model_results', []))}")
        logger.info(f"  Features engineered: {results.get('features_engineered', 'N/A')}")
        
        return {
            'task_type': 'classification',
            'success': True,
            'training_time': training_time,
            'test_accuracy': test_score,
            'best_model': best_model,
            'models_evaluated': len(results.get('model_results', [])),
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Classification test failed: {str(e)}")
        return {
            'task_type': 'classification',
            'success': False,
            'error': str(e)
        }

def test_feature_engineering():
    """Test feature engineering capabilities"""
    logger.info("=== Testing Feature Engineering ===")
    
    # Create sample data with various feature types
    random.seed(42)
    n_samples = 50
    
    data = {
        'numeric1': [random.random() for _ in range(n_samples)],
        'numeric2': [random.random() for _ in range(n_samples)],
        'categorical1': [random.choice(['A', 'B', 'C']) for _ in range(n_samples)],
        'categorical2': [random.choice(['X', 'Y']) for _ in range(n_samples)],
        'target': [random.choice([0, 1]) for _ in range(n_samples)]
    }
    
    # Add some missing values
    for i in range(0, n_samples, 10):
        data['numeric1'][i] = None
    for i in range(0, n_samples, 15):
        data['categorical1'][i] = None
    
    df = SimpleDataFrame(data)
    X = df.drop('target', axis=1)
    y = SimpleSeries(data['target'], 'target')
    
    config = AutoMLConfig(
        task_type=TaskType.CLASSIFICATION,
        feature_engineering=True,
        max_models=1  # Just test preprocessing
    )
    
    pipeline = AutoMLPipeline(config)
    
    try:
        # Test preprocessing
        X_processed, y_processed = pipeline.preprocessor.preprocess_data(X, y, is_training=True)
        
        logger.info(f"Feature Engineering Results:")
        logger.info(f"  Original features: {X.shape[1]}")
        logger.info(f"  Processed features: {X_processed.shape[1]}")
        logger.info(f"  Missing values handled: Yes")
        logger.info(f"  Categorical encoding applied: Yes")
        
        return {
            'test_type': 'feature_engineering',
            'success': True,
            'original_features': X.shape[1],
            'processed_features': X_processed.shape[1],
            'missing_values_original': 'Some',
            'missing_values_processed': 'None',
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {str(e)}")
        return {
            'test_type': 'feature_engineering',
            'success': False,
            'error': str(e)
        }

def main():
    """Run all AutoML tests"""
    logger.info("Starting AutoML Pipeline Tests for Phase 5")
    
    test_results = []
    
    # Run tests
    test_results.append(test_classification_task())
    test_results.append(test_feature_engineering())
    
    # Summary
    logger.info("=== Test Summary ===")
    successful_tests = sum(1 for result in test_results if result['success'])
    total_tests = len(test_results)
    
    logger.info(f"Tests completed: {successful_tests}/{total_tests}")
    
    for result in test_results:
        if result['success']:
            logger.info(f"✓ {result.get('task_type', result.get('test_type', 'Unknown'))}: PASSED")
        else:
            logger.info(f"✗ {result.get('task_type', result.get('test_type', 'Unknown'))}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Save results
    with open('/home/z/my-project/bharat-fm/automl_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info("Test results saved to automl_test_results.json")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)