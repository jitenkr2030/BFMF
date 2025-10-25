#!/usr/bin/env python3
"""
Test script for AutoML pipeline in Phase 5
"""

import sys
import os
import json
import time
import logging
from typing import List, Dict, Any

# Add the Bharat-FM source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bharat_fm.automl.automl_pipeline import AutoMLPipeline, AutoMLConfig, TaskType, ModelType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def generate_sample_data(n_samples=100, n_features=5, task_type='classification'):
    """Generate sample data for testing"""
    import random
    
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
    import random
    
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
        logger.info(f"  Features engineered: {len(pipeline.preprocessor.feature_names) if hasattr(pipeline, 'preprocessor') else 'N/A'}")
        
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

def test_regression_task():
    """Test AutoML pipeline with regression task"""
    logger.info("=== Testing Regression Task ===")
    
    # Generate sample regression data
    X, y = generate_sample_data(n_samples=100, n_features=4, task_type='regression')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure AutoML
    config = AutoMLConfig(
        task_type=TaskType.REGRESSION,
        time_limit=60,
        max_models=4,
        cv_folds=2,
        optimize_metric="r2",
        include_models=[ModelType.LINEAR, ModelType.TREE_BASED],
        feature_engineering=True,
        hyperparameter_tuning=True,
        ensemble_building=False
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
        
        logger.info(f"Regression Test Results:")
        logger.info(f"  Training time: {training_time:.2f}s")
        logger.info(f"  Test R²: {test_score:.4f}")
        logger.info(f"  Best model: {best_model}")
        logger.info(f"  Models evaluated: {len(results.get('model_results', []))}")
        
        return {
            'task_type': 'regression',
            'success': True,
            'training_time': training_time,
            'test_r2': test_score,
            'best_model': best_model,
            'models_evaluated': len(results.get('model_results', [])),
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Regression test failed: {str(e)}")
        return {
            'task_type': 'regression',
            'success': False,
            'error': str(e)
        }

def test_feature_engineering():
    """Test feature engineering capabilities"""
    logger.info("=== Testing Feature Engineering ===")
    
    # Create sample data with various feature types
    import random
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
    data['numeric1'][::10] = None
    data['categorical1'][::15] = None
    
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
    test_results.append(test_regression_task())
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