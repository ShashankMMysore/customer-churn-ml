"""
Model evaluation utilities for customer churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay
)


class ChurnModelEvaluator:
    """
    Evaluator for churn prediction models.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        self.predictions = {}
        
    def evaluate(self, models, X_test, y_test):
        """
        Evaluate multiple models on test set.
        
        Parameters
        ----------
        models : dict
            Dictionary of trained models
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test target
        
        Returns
        -------
        dict
            Evaluation results for each model
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name.upper()}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            
            # Store predictions
            self.predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            results[model_name] = metrics
            self.results[model_name] = metrics
        
        return results
    
    def get_metrics_dataframe(self):
        """
        Get evaluation metrics as DataFrame for easy comparison.
        
        Returns
        -------
        pd.DataFrame
            Model comparison metrics
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        metrics_list = []
        
        for model_name, metrics in self.results.items():
            metrics_list.append({
                'Model': model_name.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        return pd.DataFrame(metrics_list).set_index('Model')
    
    def print_evaluation_report(self):
        """Print detailed evaluation report."""
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        metrics_df = self.get_metrics_dataframe()
        print("\nModel Performance Comparison:")
        print(metrics_df.round(4))
        
        print("\n" + "-"*60)
        print("Detailed Classification Reports:")
        print("-"*60)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(metrics['classification_report'])
        
        print("\n" + "-"*60)
        print("Confusion Matrices:")
        print("-"*60)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name.upper()}:")
            cm = metrics['confusion_matrix']
            print(f"True Negatives:  {cm[0,0]}")
            print(f"False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}")
            print(f"True Positives:  {cm[1,1]}")
    
    def get_top_models(self, metric='roc_auc', top_n=3):
        """
        Get top performing models.
        
        Parameters
        ----------
        metric : str, default='roc_auc'
            Metric to rank by
        top_n : int, default=3
            Number of top models to return
        
        Returns
        -------
        list
            List of (model_name, score) tuples
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        model_scores = [
            (model_name, metrics[metric])
            for model_name, metrics in self.results.items()
        ]
        
        return sorted(model_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_confusion_matrix(self, model_name):
        """
        Get confusion matrix for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        
        Returns
        -------
        np.ndarray
            Confusion matrix
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        return self.results[model_name]['confusion_matrix']
    
    def get_roc_curve_data(self, model_name, y_test):
        """
        Get ROC curve data for plotting.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        y_test : pd.Series or np.ndarray
            True test labels
        
        Returns
        -------
        tuple of (fpr, tpr, roc_auc)
        """
        if model_name not in self.predictions:
            raise ValueError(f"Predictions for {model_name} not found")
        
        y_pred_proba = self.predictions[model_name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return fpr, tpr, roc_auc
    
    def get_precision_recall_curve_data(self, model_name, y_test):
        """
        Get precision-recall curve data for plotting.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        y_test : pd.Series or np.ndarray
            True test labels
        
        Returns
        -------
        tuple of (precision, recall, pr_auc)
        """
        if model_name not in self.predictions:
            raise ValueError(f"Predictions for {model_name} not found")
        
        y_pred_proba = self.predictions[model_name]['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        return precision, recall, pr_auc


def evaluate_models(models, X_test, y_test):
    """
    Convenience function to evaluate models.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : pd.Series or np.ndarray
        Test target
    
    Returns
    -------
    dict
        Evaluation results
    """
    evaluator = ChurnModelEvaluator()
    return evaluator.evaluate(models, X_test, y_test)


def print_model_comparison(results_dict):
    """
    Print model comparison in a nice format.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of evaluation results
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison = []
    for model_name, metrics in results_dict.items():
        comparison.append({
            'Model': model_name.upper(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessor import preprocess_data
    from model_training import ChurnModelTrainer
    
    # Example usage
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess_data(df)
    
    print("\nTraining models...")
    trainer = ChurnModelTrainer()
    models = trainer.train_models(X_train, y_train)
    
    print("\nEvaluating models...")
    evaluator = ChurnModelEvaluator()
    results = evaluator.evaluate(models, X_test, y_test)
    
    evaluator.print_evaluation_report()
