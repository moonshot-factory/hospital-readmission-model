#!/usr/bin/env python3
"""
Hospital Readmission Risk Model - Training Script

Standalone script for training the readmission prediction model.
Can be used for batch training, automated retraining, or CI/CD pipelines.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
Usage: python train_model.py [--data path/to/data.csv] [--output output/dir]
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import joblib

from src.data_processing import EHRDataProcessor
from src.feature_engineering import ComprehensiveFeatureEngineer
from src.models import ReadmissionPredictor, ModelOptimizer
from src.visualization import ReadmissionVisualizer

import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Hospital Readmission Risk Model')
    parser.add_argument('--data', default='data/sample_data.csv', 
                      help='Path to input data CSV file')
    parser.add_argument('--output', default='models/', 
                      help='Output directory for trained model')
    parser.add_argument('--model-type', default='auto', choices=['auto', 'logistic', 'decision_tree'],
                      help='Type of model to train')
    parser.add_argument('--cv-folds', type=int, default=5,
                      help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Fraction of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random state for reproducibility')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save performance plots')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    
    return parser.parse_args()


def load_and_process_data(data_path, verbose=False):
    """
    Load and process the EHR data.
    
    Args:
        data_path (str): Path to the data file
        verbose (bool): Print verbose output
        
    Returns:
        tuple: (X, y, feature_names, processing_report)
    """
    if verbose:
        print(f"Loading data from {data_path}...")
    
    # Load raw data
    raw_data = pd.read_csv(data_path)
    if verbose:
        print(f"Loaded {len(raw_data)} records with {len(raw_data.columns)} columns")
    
    # Initialize processors
    data_processor = EHRDataProcessor()
    feature_engineer = ComprehensiveFeatureEngineer()
    
    # Process data
    if verbose:
        print("Processing raw data...")
    processed_data, processing_report = data_processor.process_pipeline(raw_data)
    
    # Engineer features
    if verbose:
        print("Engineering features...")
    featured_data, feature_report = feature_engineer.engineer_all_features(processed_data)
    
    # Prepare final dataset
    X, y, feature_names = feature_engineer.select_final_features(featured_data)
    
    if verbose:
        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_names, processing_report


def train_model(X, y, model_type='auto', cv_folds=5, random_state=42, verbose=False):
    """
    Train the readmission prediction model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model_type (str): Type of model to train
        cv_folds (int): Number of CV folds
        random_state (int): Random state
        verbose (bool): Print verbose output
        
    Returns:
        tuple: (trained_model, training_results, optimization_results)
    """
    if verbose:
        print(f"Training model with {cv_folds}-fold cross-validation...")
    
    if model_type == 'auto':
        # Use optimizer to select best model
        optimizer = ModelOptimizer(random_state=random_state)
        comparison_results = optimizer.compare_models(X, y, cv=cv_folds, verbose=verbose)
        
        # Get the best model
        best_model = optimizer.get_best_model(X, y, cv=cv_folds)
        
        if verbose:
            print(f"Best model selected: {best_model.model_type}")
        
        optimization_results = optimizer.optimization_results
        
    else:
        # Train specific model type
        best_model = ReadmissionPredictor(model_type, random_state=random_state)
        optimization_results = None
        
        if verbose:
            print(f"Training {model_type} model...")
    
    # Train the final model
    training_results = best_model.fit(X, y, verbose=verbose)
    
    return best_model, training_results, optimization_results


def evaluate_model(model, X, y, verbose=False):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        verbose (bool): Print verbose output
        
    Returns:
        dict: Evaluation results
    """
    from src.models import ClinicalEvaluator
    
    if verbose:
        print("Evaluating model performance...")
    
    # Generate predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Clinical evaluation
    evaluator = ClinicalEvaluator()
    clinical_metrics = evaluator.evaluate_clinical_performance(y, y_pred, y_proba, model.model_type)
    
    # Calibration assessment
    calibration_results = evaluator.assess_calibration(y, y_proba)
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    
    if verbose:
        print(f"Model Performance:")
        print(f"  ROC-AUC: {clinical_metrics['ROC_AUC']:.3f}")
        print(f"  Sensitivity: {clinical_metrics['Sensitivity']:.3f}")
        print(f"  Specificity: {clinical_metrics['Specificity']:.3f}")
        print(f"  PPV: {clinical_metrics['PPV']:.3f}")
        print(f"  Calibration: {calibration_results['calibration_quality']}")
    
    return {
        'clinical_metrics': clinical_metrics,
        'calibration_results': calibration_results,
        'feature_importance': feature_importance,
        'predictions': {'y_pred': y_pred, 'y_proba': y_proba}
    }


def save_model_and_results(model, evaluation_results, training_results, 
                          output_dir, save_plots=False, verbose=False):
    """
    Save the trained model and results.
    
    Args:
        model: Trained model
        evaluation_results (dict): Evaluation results
        training_results (dict): Training results
        output_dir (str): Output directory
        save_plots (bool): Whether to save plots
        verbose (bool): Print verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f'readmission_model_{model.model_type}_{timestamp}.joblib'
    model_path = os.path.join(output_dir, model_filename)
    model.save_model(model_path)
    
    if verbose:
        print(f"Model saved to {model_path}")
    
    # Save results summary
    results_summary = {
        'timestamp': timestamp,
        'model_type': model.model_type,
        'training_results': training_results,
        'evaluation_results': {
            'clinical_metrics': evaluation_results['clinical_metrics'],
            'calibration_results': evaluation_results['calibration_results'],
            'feature_importance': evaluation_results['feature_importance'].to_dict('records')
        }
    }
    
    import json
    results_filename = f'training_results_{timestamp}.json'
    results_path = os.path.join(output_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    if verbose:
        print(f"Results saved to {results_path}")
    
    # Save plots if requested
    if save_plots:
        plots_dir = os.path.join(output_dir, f'plots_{timestamp}')
        os.makedirs(plots_dir, exist_ok=True)
        
        visualizer = ReadmissionVisualizer()
        
        # Model performance plot
        visualizer.plot_model_performance(
            evaluation_results['predictions']['y_pred'],
            evaluation_results['predictions']['y_pred'], 
            evaluation_results['predictions']['y_proba'],
            model.model_type.title(),
            save_path=os.path.join(plots_dir, 'model_performance.png')
        )
        
        # Feature importance plot
        visualizer.plot_feature_importance(
            evaluation_results['feature_importance'],
            save_path=os.path.join(plots_dir, 'feature_importance.png')
        )
        
        if verbose:
            print(f"Plots saved to {plots_dir}")
    
    return model_path, results_path


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("HOSPITAL READMISSION RISK MODEL - TRAINING")
    print("=" * 60)
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Data: {args.data}")
        print(f"  Output: {args.output}")
        print(f"  Model Type: {args.model_type}")
        print(f"  CV Folds: {args.cv_folds}")
        print(f"  Test Size: {args.test_size}")
        print(f"  Random State: {args.random_state}")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load and process data
        print("\n📊 Loading and processing data...")
        X, y, feature_names, processing_report = load_and_process_data(
            args.data, verbose=args.verbose
        )
        
        # Step 2: Train model
        print("\n🤖 Training model...")
        model, training_results, optimization_results = train_model(
            X, y, 
            model_type=args.model_type,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            verbose=args.verbose
        )
        
        # Step 3: Evaluate model
        print("\n📈 Evaluating model...")
        evaluation_results = evaluate_model(model, X, y, verbose=args.verbose)
        
        # Step 4: Save results
        print("\n💾 Saving model and results...")
        model_path, results_path = save_model_and_results(
            model, evaluation_results, training_results,
            args.output, save_plots=args.save_plots, verbose=args.verbose
        )
        
        # Summary
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Model: {model.model_type}")
        print(f"Performance: AUC = {evaluation_results['clinical_metrics']['ROC_AUC']:.3f}")
        print(f"Calibration: {evaluation_results['calibration_results']['calibration_quality']}")
        print(f"Files saved:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()