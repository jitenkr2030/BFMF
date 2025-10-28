"""
AutoML Pipeline for Bharat-FM
Implements automated machine learning pipeline for model selection, training, and optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import time
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

class DataPreprocessor:
    """Automated data preprocessing and feature engineering"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.fitted_transformers = {}
        self.feature_names = []
        self.target_encoder = None
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, 
                      is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
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
        
        # Scale features
        if is_training:
            X = self._scale_features(X, fit=True)
        else:
            X = self._scale_features(X, fit=False)
        
        # Handle target variable
        if y is not None:
            if is_training:
                y = self._encode_target(y)
            else:
                y = self._encode_target(y, fit=False)
        
        logger.info(f"Data preprocessing completed. Shape: {X.shape}")
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numeric columns, fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                # Use label encoding for high cardinality, one-hot for low cardinality
                unique_count = X[col].nunique()
                if unique_count <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                    self.fitted_transformers[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                else:
                    # Label encoding
                    from sklearn.preprocessing import LabelEncoder
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col])
                    self.fitted_transformers[col] = {'type': 'label', 'encoder': encoder}
            else:
                if col in self.fitted_transformers:
                    transformer_info = self.fitted_transformers[col]
                    if transformer_info['type'] == 'onehot':
                        # Handle unseen categories
                        X[col] = X[col].fillna('unknown')
                        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                        # Ensure same columns as training
                        for col_name in transformer_info['columns']:
                            if col_name not in dummies.columns:
                                dummies[col_name] = 0
                        dummies = dummies[transformer_info['columns']]
                        X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                    else:
                        # Label encoding
                        encoder = transformer_info['encoder']
                        # Handle unseen labels
                        X[col] = X[col].fillna('unknown')
                        X[col] = X[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        
        return X
    
    def _engineer_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Automated feature engineering"""
        original_cols = X.columns.tolist()
        
        # Polynomial features for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Create interaction features
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5 columns
                for col2 in numeric_cols[i+1:6]:  # Limit interactions
                    interaction_name = f"{col1}_x_{col2}"
                    X[interaction_name] = X[col1] * X[col2]
        
        # Statistical features
        if len(numeric_cols) > 1:
            X['numeric_mean'] = X[numeric_cols].mean(axis=1)
            X['numeric_std'] = X[numeric_cols].std(axis=1)
            X['numeric_max'] = X[numeric_cols].max(axis=1)
            X['numeric_min'] = X[numeric_cols].min(axis=1)
        
        # Date features (if any date columns exist)
        date_cols = X.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_weekday'] = X[col].dt.weekday
        
        self.feature_names = X.columns.tolist()
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features"""
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        else:
            scaled_data = self.scaler.transform(X)
            X_scaled = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        
        return X_scaled
    
    def _encode_target(self, y: pd.Series, fit: bool = True) -> pd.Series:
        """Encode target variable"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            from sklearn.preprocessing import LabelEncoder
            if fit:
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y)
            else:
                y_encoded = self.target_encoder.transform(y)
            return pd.Series(y_encoded, index=y.index)
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
            return ModelFactory._create_linear_model(task_type, hyperparameters)
        elif model_type == ModelType.TREE_BASED:
            return ModelFactory._create_tree_model(task_type, hyperparameters)
        elif model_type == ModelType.NEURAL_NETWORK:
            return ModelFactory._create_neural_network_model(task_type, hyperparameters)
        elif model_type == ModelType.ENSEMBLE:
            return ModelFactory._create_ensemble_model(task_type, hyperparameters)
        elif model_type == ModelType.PROBABILISTIC:
            return ModelFactory._create_probabilistic_model(task_type, hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def _create_linear_model(task_type: TaskType, hyperparameters: Dict[str, Any]) -> Any:
        """Create linear model"""
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
            from sklearn.svm import SVC, SVR
            
            if task_type == TaskType.CLASSIFICATION:
                default_params = {
                    'C': 1.0,
                    'random_state': 42,
                    'max_iter': 1000
                }
                default_params.update(hyperparameters)
                return LogisticRegression(**default_params)
            elif task_type == TaskType.REGRESSION:
                default_params = {
                    'random_state': 42
                }
                default_params.update(hyperparameters)
                return LinearRegression(**default_params)
        except ImportError:
            logger.warning("Scikit-learn not available, using mock model")
            return MockModel(task_type, "linear")
    
    @staticmethod
    def _create_tree_model(task_type: TaskType, hyperparameters: Dict[str, Any]) -> Any:
        """Create tree-based model"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            from xgboost import XGBClassifier, XGBRegressor
            
            if task_type == TaskType.CLASSIFICATION:
                default_params = {
                    'n_estimators': 100,
                    'random_state': 42,
                    'n_jobs': -1
                }
                default_params.update(hyperparameters)
                return RandomForestClassifier(**default_params)
            elif task_type == TaskType.REGRESSION:
                default_params = {
                    'n_estimators': 100,
                    'random_state': 42,
                    'n_jobs': -1
                }
                default_params.update(hyperparameters)
                return RandomForestRegressor(**default_params)
        except ImportError:
            logger.warning("Tree-based libraries not available, using mock model")
            return MockModel(task_type, "tree_based")
    
    @staticmethod
    def _create_neural_network_model(task_type: TaskType, hyperparameters: Dict[str, Any]) -> Any:
        """Create neural network model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            
            model = Sequential([
                Dense(64, activation='relu', input_shape=(hyperparameters.get('input_shape', 100),)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid' if task_type == TaskType.CLASSIFICATION else 'linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy' if task_type == TaskType.CLASSIFICATION else 'mse',
                metrics=['accuracy'] if task_type == TaskType.CLASSIFICATION else ['mae']
            )
            
            return model
        except ImportError:
            logger.warning("TensorFlow not available, using mock model")
            return MockModel(task_type, "neural_network")
    
    @staticmethod
    def _create_ensemble_model(task_type: TaskType, hyperparameters: Dict[str, Any]) -> Any:
        """Create ensemble model"""
        try:
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            # Create base models
            base_models = []
            
            # Add linear model
            linear_model = ModelFactory._create_linear_model(task_type, {})
            base_models.append(('linear', linear_model))
            
            # Add tree model
            tree_model = ModelFactory._create_tree_model(task_type, {})
            base_models.append(('tree', tree_model))
            
            if task_type == TaskType.CLASSIFICATION:
                return VotingClassifier(estimators=base_models, voting='soft')
            elif task_type == TaskType.REGRESSION:
                return VotingRegressor(estimators=base_models)
        except ImportError:
            logger.warning("Ensemble libraries not available, using mock model")
            return MockModel(task_type, "ensemble")
    
    @staticmethod
    def _create_probabilistic_model(task_type: TaskType, hyperparameters: Dict[str, Any]) -> Any:
        """Create probabilistic model"""
        try:
            from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
            
            if task_type == TaskType.CLASSIFICATION:
                return GaussianProcessClassifier(random_state=42)
            elif task_type == TaskType.REGRESSION:
                return GaussianProcessRegressor(random_state=42)
        except ImportError:
            logger.warning("Probabilistic libraries not available, using mock model")
            return MockModel(task_type, "probabilistic")

class MockModel:
    """Mock model for testing when libraries are not available"""
    
    def __init__(self, task_type: TaskType, model_type: str):
        self.task_type = task_type
        self.model_type = model_type
        self.trained = False
    
    def fit(self, X, y):
        """Mock fit method"""
        self.trained = True
        return self
    
    def predict(self, X):
        """Mock predict method"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        n_samples = len(X) if hasattr(X, '__len__') else 1
        if self.task_type == TaskType.CLASSIFICATION:
            return np.random.randint(0, 2, n_samples)
        else:
            return np.random.randn(n_samples)
    
    def predict_proba(self, X):
        """Mock predict_proba method"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        n_samples = len(X) if hasattr(X, '__len__') else 1
        if self.task_type == TaskType.CLASSIFICATION:
            probs = np.random.dirichlet([1, 1], n_samples)
            return probs
        else:
            return np.random.randn(n_samples)

class HyperparameterOptimizer:
    """Hyperparameter optimization using various strategies"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.optimization_history = []
    
    def optimize_hyperparameters(self, model_factory: Callable, X: pd.DataFrame, 
                               y: pd.Series, model_type: ModelType) -> Dict[str, Any]:
        """Optimize hyperparameters for a given model"""
        logger.info(f"Optimizing hyperparameters for {model_type.value}")
        
        # Define search space based on model type
        search_space = self._get_search_space(model_type)
        
        # Use grid search for small spaces, random search for larger spaces
        total_combinations = 1
        for param_values in search_space.values():
            total_combinations *= len(param_values)
        
        if total_combinations <= 50:
            best_params = self._grid_search(model_factory, X, y, search_space, model_type)
        else:
            best_params = self._random_search(model_factory, X, y, search_space, model_type)
        
        logger.info(f"Best parameters for {model_type.value}: {best_params}")
        return best_params
    
    def _get_search_space(self, model_type: ModelType) -> Dict[str, List[Any]]:
        """Get hyperparameter search space for model type"""
        search_spaces = {
            ModelType.LINEAR: {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2']
            },
            ModelType.TREE_BASED: {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            ModelType.NEURAL_NETWORK: {
                'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01, 0.1]
            },
            ModelType.ENSEMBLE: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            },
            ModelType.PROBABILISTIC: {
                'alpha': [1e-10, 1e-5, 1e-2, 1.0]
            }
        }
        
        return search_spaces.get(model_type, {})
    
    def _grid_search(self, model_factory: Callable, X: pd.DataFrame, 
                    y: pd.Series, search_space: Dict[str, List[Any]], 
                    model_type: ModelType) -> Dict[str, Any]:
        """Perform grid search for hyperparameter optimization"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        
        # Create base model
        base_model = model_factory(model_type, self.config.task_type)
        
        # Define scoring metric
        scoring = self._get_scoring_metric()
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, search_space, 
            cv=self.config.cv_folds, 
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_
    
    def _random_search(self, model_factory: Callable, X: pd.DataFrame, 
                     y: pd.Series, search_space: Dict[str, List[Any]], 
                     model_type: ModelType) -> Dict[str, Any]:
        """Perform random search for hyperparameter optimization"""
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import make_scorer
        import scipy.stats as stats
        
        # Create parameter distributions
        param_distributions = {}
        for param, values in search_space.items():
            if isinstance(values, list) and len(values) > 0:
                param_distributions[param] = values
        
        # Create base model
        base_model = model_factory(model_type, self.config.task_type)
        
        # Define scoring metric
        scoring = self._get_scoring_metric()
        
        # Perform random search
        random_search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=20, cv=self.config.cv_folds,
            scoring=scoring, n_jobs=-1,
            verbose=0, random_state=self.config.random_state
        )
        
        random_search.fit(X, y)
        
        return random_search.best_params_
    
    def _get_scoring_metric(self):
        """Get scoring metric based on task type and optimization metric"""
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metric_mapping = {
            'accuracy': accuracy_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2_score
        }
        
        metric_func = metric_mapping.get(self.config.optimize_metric, accuracy_score)
        return make_scorer(metric_func)

class ModelEvaluator:
    """Model evaluation and comparison"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.evaluation_results = []
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series, model_name: str, 
                      model_type: ModelType) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        if self.config.task_type == TaskType.CLASSIFICATION:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics = self._evaluate_classification(y_test, y_pred, y_pred_proba)
        elif self.config.task_type == TaskType.REGRESSION:
            y_pred = model.predict(X_test)
            metrics = self._evaluate_regression(y_test, y_pred)
        else:
            metrics = {}
        
        # Calculate model size (approximate)
        model_size = self._calculate_model_size(model)
        
        logger.info(f"Model {model_name} evaluation completed")
        return {**metrics, 'model_size': model_size}
    
    def _evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate classification model"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                pass
        
        return metrics
    
    def _evaluate_regression(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def _calculate_model_size(self, model: Any) -> int:
        """Calculate approximate model size in bytes"""
        try:
            return len(pickle.dumps(model))
        except:
            return 0

class AutoMLPipeline:
    """Main AutoML pipeline class"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.preprocessor = DataPreprocessor(self.config)
        self.model_factory = ModelFactory
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        
        self.trained_models = []
        self.best_model = None
        self.best_model_result = None
        self.pipeline_results = {}
        
        self.start_time = None
        self.end_time = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoMLPipeline':
        """Run the complete AutoML pipeline"""
        logger.info("Starting AutoML pipeline...")
        self.start_time = time.time()
        
        # Preprocess data
        X_processed, y_processed = self.preprocessor.preprocess_data(X, y, is_training=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y_processed if self.config.task_type == TaskType.CLASSIFICATION else None
        )
        
        # Train and evaluate models
        self._train_and_evaluate_models(X_train, y_train, X_test, y_test)
        
        # Build ensemble if enabled
        if self.config.ensemble_building and len(self.trained_models) > 1:
            self._build_ensemble(X_train, y_train, X_test, y_test)
        
        # Select best model
        self._select_best_model()
        
        self.end_time = time.time()
        self.pipeline_results['total_time'] = self.end_time - self.start_time
        
        logger.info(f"AutoML pipeline completed in {self.pipeline_results['total_time']:.2f} seconds")
        return self
    
    def _train_and_evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series):
        """Train and evaluate multiple models"""
        logger.info("Training and evaluating models...")
        
        models_trained = 0
        start_time = time.time()
        
        for model_type in self.config.include_models:
            if model_type in self.config.exclude_models:
                continue
            
            # Check time limit
            if self.config.time_limit > 0:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.time_limit * 0.8:  # Leave 20% for ensemble
                    logger.info(f"Time limit reached, stopping model training")
                    break
            
            # Check model limit
            if models_trained >= self.config.max_models:
                logger.info(f"Model limit reached, stopping model training")
                break
            
            # Create and train model
            try:
                model_result = self._train_single_model(
                    model_type, X_train, y_train, X_test, y_test
                )
                
                if model_result:
                    self.trained_models.append(model_result)
                    models_trained += 1
                    
                    logger.info(f"Trained {model_type.value} model: {model_result.model_name} "
                               f"({self.config.optimize_metric}: {model_result.test_score:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type.value} model: {e}")
                continue
        
        self.pipeline_results['models_trained'] = models_trained
        self.pipeline_results['training_time'] = time.time() - start_time
    
    def _train_single_model(self, model_type: ModelType, X_train: pd.DataFrame, 
                           y_train: pd.Series, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Optional[ModelResult]:
        """Train and evaluate a single model"""
        start_time = time.time()
        
        # Optimize hyperparameters if enabled
        if self.config.hyperparameter_tuning:
            hyperparameters = self.hyperparameter_optimizer.optimize_hyperparameters(
                self.model_factory.create_model, X_train, y_train, model_type
            )
        else:
            hyperparameters = {}
        
        # Create model with optimized hyperparameters
        try:
            model = self.model_factory.create_model(
                model_type, self.config.task_type, hyperparameters
            )
            
            # Update input shape for neural networks
            if model_type == ModelType.NEURAL_NETWORK:
                hyperparameters['input_shape'] = X_train.shape[1]
                model = self.model_factory.create_model(
                    model_type, self.config.task_type, hyperparameters
                )
            
            # Train model
            if model_type == ModelType.NEURAL_NETWORK:
                # Special handling for neural networks
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            patience=5, restore_best_weights=True
                        )
                    ] if self.config.early_stopping else []
                )
            else:
                model.fit(X_train, y_train)
            
            # Evaluate model
            evaluation_metrics = self.model_evaluator.evaluate_model(
                model, X_test, y_test, f"{model_type.value}_optimized", model_type
            )
            
            # Calculate feature importance if available
            feature_importance = self._calculate_feature_importance(model, X_train.columns)
            
            training_time = time.time() - start_time
            
            # Create model result
            model_result = ModelResult(
                model_name=f"{model_type.value}_optimized",
                model_type=model_type,
                model=model,
                training_time=training_time,
                evaluation_metrics=evaluation_metrics,
                hyperparameters=hyperparameters,
                feature_importance=feature_importance,
                validation_score=evaluation_metrics.get(self.config.optimize_metric, 0.0),
                test_score=evaluation_metrics.get(self.config.optimize_metric, 0.0),
                model_size=evaluation_metrics.get('model_size', 0)
            )
            
            return model_result
            
        except Exception as e:
            logger.error(f"Error training {model_type.value} model: {e}")
            return None
    
    def _calculate_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Calculate feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]
                importances = np.abs(coef)
                return dict(zip(feature_names, importances))
        except:
            pass
        
        return None
    
    def _build_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series):
        """Build ensemble from best performing models"""
        logger.info("Building ensemble model...")
        
        # Select top performing models
        sorted_models = sorted(self.trained_models, key=lambda x: x.test_score, reverse=True)
        top_models = sorted_models[:min(3, len(sorted_models))]
        
        if len(top_models) < 2:
            logger.info("Not enough models for ensemble building")
            return
        
        # Create ensemble
        try:
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            estimators = [(f"model_{i}", model.model) for i, model in enumerate(top_models)]
            
            if self.config.task_type == TaskType.CLASSIFICATION:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=estimators)
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            evaluation_metrics = self.model_evaluator.evaluate_model(
                ensemble, X_test, y_test, "ensemble", ModelType.ENSEMBLE
            )
            
            # Create ensemble result
            ensemble_result = ModelResult(
                model_name="ensemble",
                model_type=ModelType.ENSEMBLE,
                model=ensemble,
                training_time=0.0,  # Will be calculated
                evaluation_metrics=evaluation_metrics,
                hyperparameters={"estimators": len(top_models)},
                validation_score=evaluation_metrics.get(self.config.optimize_metric, 0.0),
                test_score=evaluation_metrics.get(self.config.optimize_metric, 0.0),
                model_size=evaluation_metrics.get('model_size', 0)
            )
            
            self.trained_models.append(ensemble_result)
            logger.info(f"Ensemble model created with {len(top_models)} base models")
            
        except Exception as e:
            logger.error(f"Failed to build ensemble: {e}")
    
    def _select_best_model(self):
        """Select the best performing model"""
        if not self.trained_models:
            logger.warning("No models trained")
            return
        
        # Sort models by test score
        sorted_models = sorted(self.trained_models, key=lambda x: x.test_score, reverse=True)
        
        self.best_model = sorted_models[0].model
        self.best_model_result = sorted_models[0]
        
        logger.info(f"Best model: {self.best_model_result.model_name} "
                   f"({self.config.optimize_metric}: {self.best_model_result.test_score:.4f})")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the best model"""
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")
        
        # Preprocess input data
        X_processed, _ = self.preprocessor.preprocess_data(X, is_training=False)
        
        # Make predictions
        return self.best_model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions with the best model"""
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")
        
        # Preprocess input data
        X_processed, _ = self.preprocessor.preprocess_data(X, is_training=False)
        
        # Make probability predictions
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X_processed)
        else:
            raise ValueError("Best model does not support probability predictions")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get comprehensive results summary"""
        return {
            'config': asdict(self.config),
            'pipeline_results': self.pipeline_results,
            'trained_models': [
                {
                    'model_name': result.model_name,
                    'model_type': result.model_type.value,
                    'test_score': result.test_score,
                    'training_time': result.training_time,
                    'model_size': result.model_size,
                    'evaluation_metrics': result.evaluation_metrics
                }
                for result in self.trained_models
            ],
            'best_model': {
                'model_name': self.best_model_result.model_name,
                'model_type': self.best_model_result.model_type.value,
                'test_score': self.best_model_result.test_score,
                'hyperparameters': self.best_model_result.hyperparameters
            } if self.best_model_result else None,
            'feature_importance': self.best_model_result.feature_importance if self.best_model_result else None
        }
    
    def save_model(self, filepath: str):
        """Save the best model and pipeline"""
        model_data = {
            'best_model': self.best_model,
            'preprocessor': self.preprocessor,
            'config': self.config,
            'results': self.get_results_summary()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and pipeline"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.preprocessor = model_data['preprocessor']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {filepath}")

# Factory functions
def create_automl_pipeline(task_type: str = "classification", 
                          time_limit: int = 3600, **kwargs) -> AutoMLPipeline:
    """Create AutoML pipeline with specified configuration"""
    config = AutoMLConfig(
        task_type=TaskType(task_type),
        time_limit=time_limit,
        **kwargs
    )
    return AutoMLPipeline(config)

# Example usage and testing
def test_automl_pipeline():
    """Test the AutoML pipeline functionality"""
    print("Testing AutoML Pipeline...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    print(f"Generated dataset: {X_df.shape}")
    print(f"Class distribution: {y_series.value_counts().to_dict()}")
    
    # Create AutoML pipeline
    config = AutoMLConfig(
        task_type=TaskType.CLASSIFICATION,
        time_limit=300,  # 5 minutes
        max_models=10,
        optimize_metric="accuracy",
        feature_engineering=True,
        hyperparameter_tuning=True,
        ensemble_building=True
    )
    
    pipeline = AutoMLPipeline(config)
    
    # Run pipeline
    print("Running AutoML pipeline...")
    pipeline.fit(X_df, y_series)
    
    # Get results
    results = pipeline.get_results_summary()
    
    print(f"\nPipeline Results:")
    print(f"Total time: {results['pipeline_results']['total_time']:.2f} seconds")
    print(f"Models trained: {results['pipeline_results']['models_trained']}")
    print(f"Best model: {results['best_model']['model_name']}")
    print(f"Best score: {results['best_model']['test_score']:.4f}")
    
    # Make predictions
    test_data = X_df.head(5)
    predictions = pipeline.predict(test_data)
    print(f"\nPredictions for test data: {predictions}")
    
    print("AutoML pipeline tests completed!")

if __name__ == "__main__":
    test_automl_pipeline()