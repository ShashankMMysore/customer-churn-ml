"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessor import ChurnPreprocessor, preprocess_data
from src.data_loader import generate_synthetic_churn_data


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return generate_synthetic_churn_data(n_samples=100, random_state=42)


class TestChurnPreprocessor:
    """Test suite for ChurnPreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = ChurnPreprocessor(test_size=0.2, random_state=42)
        assert preprocessor.test_size == 0.2
        assert preprocessor.random_state == 42
    
    def test_fit_transform_returns_correct_shapes(self, sample_data):
        """Test that fit_transform returns correct shapes."""
        preprocessor = ChurnPreprocessor(test_size=0.2)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
        
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_fit_transform_returns_correct_test_size(self, sample_data):
        """Test that test_size is respected."""
        preprocessor = ChurnPreprocessor(test_size=0.2)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
        
        test_ratio = len(X_test) / len(sample_data)
        assert 0.15 < test_ratio < 0.25  # Allow small deviation
    
    def test_stratification(self, sample_data):
        """Test that stratification preserves class distribution."""
        original_ratio = sample_data['churn'].mean()
        
        preprocessor = ChurnPreprocessor(test_size=0.2)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
        
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        # Class ratio should be similar across splits
        assert abs(train_ratio - original_ratio) < 0.1
        assert abs(test_ratio - original_ratio) < 0.1
    
    def test_feature_scaling(self, sample_data):
        """Test that features are properly scaled."""
        preprocessor = ChurnPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
        
        # Scaled features should have mean close to 0 and std close to 1
        train_means = X_train.mean(axis=0)
        train_stds = X_train.std(axis=0)
        
        assert np.allclose(train_means, 0, atol=1e-10)
        assert np.allclose(train_stds, 1, atol=0.1)
    
    def test_no_nan_values_after_preprocessing(self, sample_data):
        """Test that there are no NaN values after preprocessing."""
        preprocessor = ChurnPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
        
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()
    
    def test_transform_consistency(self, sample_data):
        """Test that transform produces consistent results."""
        preprocessor = ChurnPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_data)
        
        # Apply transform to same data
        X_transformed = preprocessor.transform(sample_data)
        
        assert X_transformed.shape[1] == X_train.shape[1]


class TestPreprocessFunction:
    """Test suite for preprocess_data function."""
    
    def test_preprocess_data_returns_all_outputs(self, sample_data):
        """Test that preprocess_data returns all expected outputs."""
        result = preprocess_data(sample_data)
        
        assert len(result) == 6
        X_train, X_test, y_train, y_test, feature_names, preprocessor = result
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(feature_names, list)
    
    def test_feature_names_match_data_shape(self, sample_data):
        """Test that feature names match data shape."""
        X_train, X_test, y_train, y_test, feature_names, _ = preprocess_data(sample_data)
        
        assert len(feature_names) == X_train.shape[1]
        assert set(feature_names) == set(X_train.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
