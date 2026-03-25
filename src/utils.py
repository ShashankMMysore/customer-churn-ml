"""
Utility functions for customer churn ML project.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_config(config_path='configs/config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(results, output_path='results/'):
    """
    Save evaluation results to JSON file.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    output_path : str
        Output directory path
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = {}
    for model_name, metrics in results.items():
        results_serializable[model_name] = {}
        for metric_name, value in metrics.items():
            if metric_name == 'confusion_matrix':
                results_serializable[model_name][metric_name] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_serializable[model_name][metric_name] = float(value)
            elif metric_name != 'classification_report':
                results_serializable[model_name][metric_name] = value
    
    output_file = os.path.join(output_path, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"Results saved to {output_file}")


def plot_confusion_matrices(results, y_test, output_path='results/plots/'):
    """
    Create confusion matrix plots for all models.
    
    Parameters
    ----------
    results : dict
        Dictionary of evaluation results with confusion matrices
    y_test : pd.Series or np.ndarray
        True test labels
    output_path : str
        Output directory for plots
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = metrics['confusion_matrix']
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            cbar=False
        )
        
        axes[idx].set_title(f'{model_name.upper()} Confusion Matrix')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_file = os.path.join(output_path, 'confusion_matrices.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {plot_file}")
    plt.close()


def plot_model_comparison(results, metric='roc_auc', output_path='results/plots/'):
    """
    Create bar plot comparing models on a specific metric.
    
    Parameters
    ----------
    results : dict
        Dictionary of evaluation results
    metric : str, default='roc_auc'
        Metric to compare
    output_path : str
        Output directory for plots
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    models = [name.upper() for name in results.keys()]
    scores = [metrics[metric] for metrics in results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom'
        )
    
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel('Model')
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    plot_file = os.path.join(output_path, f'{metric}_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_file}")
    plt.close()


def plot_feature_distribution(df, features, output_path='results/plots/'):
    """
    Create distribution plots for numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    features : list
        List of feature names to plot
    output_path : str
        Output directory for plots
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        if feature in df.columns:
            axes[idx].hist(df[feature], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_file = os.path.join(output_path, 'feature_distributions.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Feature distribution plot saved to {plot_file}")
    plt.close()


def plot_roc_curves(evaluator, model_names, y_test, output_path='results/plots/'):
    """
    Create ROC curve plot for all models.
    
    Parameters
    ----------
    evaluator : ChurnModelEvaluator
        Model evaluator with predictions
    model_names : list
        List of model names
    y_test : pd.Series or np.ndarray
        True test labels
    output_path : str
        Output directory for plots
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for model_name in model_names:
        fpr, tpr, roc_auc = evaluator.get_roc_curve_data(model_name, y_test)
        plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {roc_auc:.4f})', linewidth=2)
    
    # Plot random classifier baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_file = os.path.join(output_path, 'roc_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"ROC curve plot saved to {plot_file}")
    plt.close()


def create_summary_report(results, evaluator, y_test, output_path='results/'):
    """
    Create a comprehensive summary report.
    
    Parameters
    ----------
    results : dict
        Evaluation results
    evaluator : ChurnModelEvaluator
        Model evaluator
    y_test : pd.Series or np.ndarray
        True test labels
    output_path : str
        Output directory
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    report_file = os.path.join(output_path, 'summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CUSTOMER CHURN PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("TOP 3 MODELS\n")
        f.write("-"*80 + "\n\n")
        
        top_models = evaluator.get_top_models('roc_auc', top_n=3)
        for rank, (model_name, score) in enumerate(top_models, 1):
            f.write(f"{rank}. {model_name.upper()} (ROC-AUC: {score:.4f})\n")
    
    print(f"Summary report saved to {report_file}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions for customer churn ML project")
    print("\nAvailable functions:")
    print("  - load_config()")
    print("  - save_results()")
    print("  - plot_confusion_matrices()")
    print("  - plot_model_comparison()")
    print("  - plot_feature_distribution()")
    print("  - plot_roc_curves()")
    print("  - create_summary_report()")
