"""
Model training utilities for customer churn prediction.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from pathlib import Path


class ChurnModelTrainer:
    """
    Trainer for multiple churn prediction models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize model trainer.
        
        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        
    def _get_logistic_regression(self):
        """Create logistic regression model."""
        return LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
    
    def _get_random_forest(self):
        """Create random forest model."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _get_xgboost(self):
        """Create XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    def _get_svm(self):
        """Create SVM model."""
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
    
    def train_models(self, X_train, y_train, models_to_train=['lr', 'rf', 'xgb', 'svm']):
        """
        Train multiple models.
        
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target
        models_to_train : list of str
            Model names to train: 'lr', 'rf', 'xgb', 'svm'
        
        Returns
        -------
        dict
            Dictionary of trained models
        """
        model_getters = {
            'lr': self._get_logistic_regression,
            'rf': self._get_random_forest,
            'xgb': self._get_xgboost,
            'svm': self._get_svm
        }
        
        for model_name in models_to_train:
            if model_name not in model_getters:
                print(f"Unknown model: {model_name}")
                continue
            
            print(f"\nTraining {model_name.upper()}...")
            model = model_getters[model_name]()
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            print(f"✓ {model_name.upper()} trained")
        
        return self.trained_models
    
    def evaluate_with_cv(self, X_train, y_train, cv=5):
        """
        Evaluate models using cross-validation.
        
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target
        cv : int, default=5
            Number of cross-validation folds
        
        Returns
        -------
        dict
            Cross-validation scores for each model
        """
        cv_results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"Cross-validating {model_name.upper()}...")
            
            # Accuracy
            acc_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='accuracy'
            )
            
            # ROC-AUC
            auc_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='roc_auc'
            )
            
            # F1-Score
            f1_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='f1'
            )
            
            cv_results[model_name] = {
                'accuracy': acc_scores,
                'accuracy_mean': acc_scores.mean(),
                'accuracy_std': acc_scores.std(),
                'auc': auc_scores,
                'auc_mean': auc_scores.mean(),
                'auc_std': auc_scores.std(),
                'f1': f1_scores,
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
            }
        
        return cv_results
    
    def select_best_model(self, metric='auc_mean'):
        """
        Select best performing model based on metric.
        
        Parameters
        ----------
        metric : str, default='auc_mean'
            Metric to use for selection
        
        Returns
        -------
        tuple of (best_model_name, best_model)
        """
        if not hasattr(self, '_cv_results'):
            raise ValueError("No CV results available. Run evaluate_with_cv first.")
        
        best_score = -np.inf
        best_name = None
        best_model = None
        
        for model_name, model in self.trained_models.items():
            if model_name in self._cv_results:
                score = self._cv_results[model_name].get(metric, -np.inf)
                if score > best_score:
                    best_score = score
                    best_name = model_name
                    best_model = model
        
        self.best_model = best_model
        return best_name, best_model
    
    def save_model(self, model, path):
        """
        Save trained model to disk.
        
        Parameters
        ----------
        model : sklearn model
            Trained model to save
        path : str
            Output file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load trained model from disk.
        
        Parameters
        ----------
        path : str
            Path to model file
        
        Returns
        -------
        Trained model
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {path}")
        return model
    
    def get_feature_importance(self, model_name, feature_names, top_n=10):
        """
        Get feature importance from tree-based models.
        
        Parameters
        ----------
        model_name : str
            Name of the model ('rf' or 'xgb')
        feature_names : list
            Names of features
        top_n : int, default=10
            Number of top features to return
        
        Returns
        -------
        pd.DataFrame
            Feature importance ranking
        """
        model = self.trained_models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not trained")
            return None
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} doesn't support feature importance")
            return None
        
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


def train_models(X_train, y_train, models=['lr', 'rf', 'xgb', 'svm']):
    """
    Convenience function to train all models.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    models : list of str
        Models to train
    
    Returns
    -------
    dict
        Dictionary of trained models
    """
    trainer = ChurnModelTrainer()
    return trainer.train_models(X_train, y_train, models)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessor import preprocess_data
    
    # Example usage
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess_data(df)
    
    print("\nTraining models...")
    trainer = ChurnModelTrainer()
    trainer.train_models(X_train, y_train)
    
    print("\nCross-validating...")
    cv_results = trainer.evaluate_with_cv(X_train, y_train)
    
    for model_name, scores in cv_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {scores['accuracy_mean']:.4f} (+/- {scores['accuracy_std']:.4f})")
        print(f"  ROC-AUC:  {scores['auc_mean']:.4f} (+/- {scores['auc_std']:.4f})")
        print(f"  F1-Score: {scores['f1_mean']:.4f} (+/- {scores['f1_std']:.4f})")
