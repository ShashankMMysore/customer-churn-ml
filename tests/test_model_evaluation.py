"""
Unit tests for model evaluation module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model_evaluation import ChurnModelEvaluator, evaluate_models


@pytest.fixture
def sample_models_and_data():
    """Create sample models and test data."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Split data
    X_test = X[80:]
    y_test = y[80:]
    X_train = X[:80]
    y_train = y[:80]
    
    # Train models
    models = {
        'lr': LogisticRegression(random_state=42),
        'rf': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    return models, X_test, y_test


class TestChurnModelEvaluator:
    """Test suite for ChurnModelEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ChurnModelEvaluator()
        assert evaluator.results == {}
        assert evaluator.predictions == {}
    
    def test_evaluate_returns_results(self, sample_models_and_data):
        """Test that evaluate returns results for all models."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        results = evaluator.evaluate(models, X_test, y_test)
        
        assert len(results) == len(models)
        assert all(model_name in results for model_name in models.keys())
    
    def test_evaluation_metrics_present(self, sample_models_and_data):
        """Test that all required metrics are present."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        results = evaluator.evaluate(models, X_test, y_test)
        
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1',
            'roc_auc', 'confusion_matrix', 'classification_report'
        ]
        
        for metrics in results.values():
            for metric in required_metrics:
                assert metric in metrics
    
    def test_metrics_in_valid_range(self, sample_models_and_data):
        """Test that metrics are in valid ranges."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        results = evaluator.evaluate(models, X_test, y_test)
        
        for metrics in results.values():
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1'] <= 1
            assert 0 <= metrics['roc_auc'] <= 1
    
    def test_confusion_matrix_shape(self, sample_models_and_data):
        """Test that confusion matrix has correct shape."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        results = evaluator.evaluate(models, X_test, y_test)
        
        for metrics in results.values():
            cm = metrics['confusion_matrix']
            assert cm.shape == (2, 2)
    
    def test_get_metrics_dataframe(self, sample_models_and_data):
        """Test get_metrics_dataframe returns correct format."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        evaluator.evaluate(models, X_test, y_test)
        
        metrics_df = evaluator.get_metrics_dataframe()
        
        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) == len(models)
        assert 'Accuracy' in metrics_df.columns
        assert 'ROC-AUC' in metrics_df.columns
    
    def test_get_top_models(self, sample_models_and_data):
        """Test get_top_models returns correct ranking."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        evaluator.evaluate(models, X_test, y_test)
        
        top_models = evaluator.get_top_models('accuracy', top_n=2)
        
        assert len(top_models) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top_models)
        # Should be sorted in descending order
        assert top_models[0][1] >= top_models[-1][1]
    
    def test_get_confusion_matrix(self, sample_models_and_data):
        """Test get_confusion_matrix for specific model."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        evaluator.evaluate(models, X_test, y_test)
        
        cm = evaluator.get_confusion_matrix('lr')
        assert cm.shape == (2, 2)
    
    def test_get_roc_curve_data(self, sample_models_and_data):
        """Test get_roc_curve_data returns correct data."""
        models, X_test, y_test = sample_models_and_data
        
        evaluator = ChurnModelEvaluator()
        evaluator.evaluate(models, X_test, y_test)
        
        fpr, tpr, roc_auc = evaluator.get_roc_curve_data('lr', y_test)
        
        assert len(fpr) > 0
        assert len(tpr) > 0
        assert 0 <= roc_auc <= 1
    
    def test_error_on_missing_results(self):
        """Test that error is raised when results are missing."""
        evaluator = ChurnModelEvaluator()
        
        with pytest.raises(ValueError):
            evaluator.get_top_models()


class TestEvaluateModelsFunction:
    """Test suite for evaluate_models function."""
    
    def test_evaluate_models_returns_dict(self, sample_models_and_data):
        """Test that evaluate_models returns dictionary."""
        models, X_test, y_test = sample_models_and_data
        
        results = evaluate_models(models, X_test, y_test)
        
        assert isinstance(results, dict)
        assert len(results) == len(models)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
