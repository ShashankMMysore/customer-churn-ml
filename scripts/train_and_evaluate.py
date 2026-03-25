"""
Main script for training and evaluating customer churn models.

This script orchestrates the entire ML pipeline:
1. Generate/load data
2. Preprocess and feature engineering
3. Train multiple models
4. Evaluate and compare models
5. Save results and visualizations
"""

print("🚀 Script started...")
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, save_data
from src.preprocessor import preprocess_data
from src.model_training import ChurnModelTrainer
from src.model_evaluation import ChurnModelEvaluator
from src.utils import (
    load_config,
    save_results,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_distribution,
    plot_roc_curves,
    create_summary_report
)


def main(args):
    """
    Main pipeline function.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    
    print("="*80)
    print("CUSTOMER CHURN PREDICTION - ML PIPELINE")
    print("="*80)
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    try:
        config = load_config(args.config)
        print("✓ Configuration loaded")
    except FileNotFoundError:
        print("⚠ Config file not found, using defaults")
        config = {}
    
    # Load data
    print("\n[2/6] Loading data...")
    df = load_data(
        raw_data_path=config.get('data', {}).get('raw_path', 'data/raw/customer_churn.csv'),
        generate_if_missing=True
    )
    print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"  Churn rate: {df['churn'].mean():.2%}")
    
    # Save raw data
    Path(config.get('data', {}).get('raw_path', 'data/raw')).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.get('data', {}).get('raw_path', 'data/raw/customer_churn.csv'), index=False)
    
    # Preprocess data
    print("\n[3/6] Preprocessing and feature engineering...")
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data(
        df,
        test_size=config.get('data', {}).get('test_size', 0.2),
        random_state=config.get('data', {}).get('random_state', 42)
    )
    print(f"✓ Data preprocessed")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    
    # Save processed data
    processed_path = config.get('data', {}).get('processed_path', 'data/processed/customer_churn_processed.csv')
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    X_train_combined = X_train.copy()
    X_train_combined['churn'] = y_train.values
    X_train_combined.to_csv(processed_path, index=False)
    print(f"✓ Processed data saved to {processed_path}")
    
    # Train models
    print("\n[4/6] Training models...")
    models_to_train = ['lr', 'rf', 'xgb', 'svm']
    trainer = ChurnModelTrainer(random_state=config.get('data', {}).get('random_state', 42))
    
    trained_models = trainer.train_models(X_train, y_train, models_to_train=models_to_train)
    print(f"✓ Trained {len(trained_models)} models")
    
    # Evaluate models
    print("\n[5/6] Evaluating models...")
    evaluator = ChurnModelEvaluator()
    results = evaluator.evaluate(trained_models, X_test, y_test)
    
    # Print evaluation report
    evaluator.print_evaluation_report()
    
    # Save results
    print("\n[6/6] Saving results and visualizations...")
    save_results(results, output_dir)
    
    # Generate visualizations
    try:
        plot_confusion_matrices(results, y_test, output_path=plots_dir)
        plot_model_comparison(results, metric='roc_auc', output_path=plots_dir)
        plot_model_comparison(results, metric='f1', output_path=plots_dir)
        plot_feature_distribution(df, feature_names[:6], output_path=plots_dir)
        plot_roc_curves(evaluator, list(trained_models.keys()), y_test, output_path=plots_dir)
        create_summary_report(results, evaluator, y_test, output_path=output_dir)
        print("✓ Visualizations generated")
    except Exception as e:
        print(f"⚠ Warning: Could not generate some visualizations: {e}")
    
    # Save best model
    print("\nSaving models...")
    best_score = -float('inf')
    best_model_name = None
    best_model = None
    
    for model_name, metrics in results.items():
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model_name = model_name
            best_model = trained_models[model_name]
    
    model_path = config.get('output', {}).get('model_path', 'models/churn_model.pkl')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_model, model_path)
    print(f"✓ Best model saved: {best_model_name.upper()} (ROC-AUC: {best_score:.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Best model: {best_model_name.upper()} with ROC-AUC score of {best_score:.4f}")
    print("\nTop 3 Models:")
    for rank, (name, score) in enumerate(evaluator.get_top_models('roc_auc', top_n=3), 1):
        print(f"  {rank}. {name.upper()}: {score:.4f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate customer churn prediction models'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/',
        help='Output directory for results'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/raw/customer_churn.csv',
        help='Path to data file'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
