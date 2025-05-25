#!/usr/bin/env python3
"""
Hospital Readmission Risk Model - Prediction Script

Script for making predictions on new patient data using a trained model.
Suitable for batch predictions or integration into clinical workflows.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
Usage: python predict.py --model path/to/model.joblib --data path/to/new_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import warnings

from src.data_processing import EHRDataProcessor
from src.feature_engineering import ComprehensiveFeatureEngineer
from src.models import ReadmissionPredictor
from src.visualization import ReadmissionVisualizer

warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with Hospital Readmission Risk Model')
    parser.add_argument('--model', required=True,
                      help='Path to trained model file (.joblib)')
    parser.add_argument('--data', required=True,
                      help='Path to input data CSV file')
    parser.add_argument('--output', default='predictions.csv',
                      help='Output file for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Risk threshold for binary classification')
    parser.add_argument('--include-probabilities', action='store_true',
                      help='Include probability scores in output')
    parser.add_argument('--create-dashboard', action='store_true',
                      help='Create individual patient dashboards')
    parser.add_argument('--dashboard-dir', default='dashboards/',
                      help='Directory for patient dashboards')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    
    return parser.parse_args()


def load_model(model_path, verbose=False):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the model file
        verbose (bool): Print verbose output
        
    Returns:
        ReadmissionPredictor: Loaded model
    """
    if verbose:
        print(f"Loading model from {model_path}...")
    
    try:
        model = ReadmissionPredictor.load_model(model_path)
        if verbose:
            print(f"Model loaded successfully: {model.model_type}")
            print(f"Features expected: {len(model.feature_names)}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)


def preprocess_new_data(data_path, model_feature_names, verbose=False):
    """
    Preprocess new data to match the training data format.
    
    Args:
        data_path (str): Path to new data
        model_feature_names (list): Expected feature names from model
        verbose (bool): Print verbose output
        
    Returns:
        pd.DataFrame: Preprocessed features ready for prediction
    """
    if verbose:
        print(f"Loading new data from {data_path}...")
    
    # Load raw data
    raw_data = pd.read_csv(data_path)
    if verbose:
        print(f"Loaded {len(raw_data)} records")
    
    # Initialize processors
    data_processor = EHRDataProcessor()
    feature_engineer = ComprehensiveFeatureEngineer()
    
    # Process data
    if verbose:
        print("Processing data...")
    processed_data, _ = data_processor.process_pipeline(raw_data)
    
    # Engineer features
    if verbose:
        print("Engineering features...")
    featured_data, _ = feature_engineer.engineer_all_features(processed_data)
    
    # Select features that match the model
    available_features = []
    missing_features = []
    
    for feature in model_feature_names:
        if feature in featured_data.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from new data:")
        for feature in missing_features[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(missing_features) > 5:
            print(f"  ... and {len(missing_features) - 5} more")
        
        # Create missing features with default values
        for feature in missing_features:
            if feature.startswith('has_') or feature.endswith('_admission') or 'risk' in feature:
                featured_data[feature] = 0  # Binary features default to 0
            elif 'age' in feature.lower():
                featured_data[feature] = featured_data.get('age', 65)  # Default age
            else:
                featured_data[feature] = 0  # Default to 0 for other features
    
    # Select final features in the correct order
    X = featured_data[model_feature_names].copy()
    
    if verbose:
        print(f"Final prediction dataset: {X.shape}")
    
    return X, featured_data


def make_predictions(model, X, threshold=0.5, verbose=False):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Feature matrix
        threshold (float): Classification threshold
        verbose (bool): Print verbose output
        
    Returns:
        dict: Prediction results
    """
    if verbose:
        print("Making predictions...")
    
    # Generate predictions
    y_pred_binary = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Apply custom threshold if different from 0.5
    if threshold != 0.5:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Risk categories
    risk_categories = []
    for prob in y_pred_proba:
        if prob < 0.2:
            risk_categories.append('Low')
        elif prob < 0.5:
            risk_categories.append('Moderate')
        elif prob < 0.8:
            risk_categories.append('High')
        else:
            risk_categories.append('Very High')
    
    if verbose:
        print(f"Predictions generated for {len(X)} patients")
        risk_distribution = pd.Series(risk_categories).value_counts()
        print(f"Risk distribution:")
        for risk, count in risk_distribution.items():
            print(f"  {risk}: {count} ({count/len(X)*100:.1f}%)")
    
    return {
        'binary_predictions': y_pred_binary,
        'probabilities': y_pred_proba,
        'risk_categories': risk_categories
    }


def create_output_file(predictions, original_data, output_path, include_probabilities=True, verbose=False):
    """
    Create output file with predictions.
    
    Args:
        predictions (dict): Prediction results
        original_data (pd.DataFrame): Original patient data
        output_path (str): Output file path
        include_probabilities (bool): Include probability scores
        verbose (bool): Print verbose output
    """
    if verbose:
        print(f"Creating output file: {output_path}")
    
    # Create output DataFrame
    output_data = original_data[['patient_id']].copy() if 'patient_id' in original_data.columns else pd.DataFrame()
    
    # Add basic patient info if available
    info_columns = ['age', 'gender', 'length_of_stay', 'emergency_admission']
    for col in info_columns:
        if col in original_data.columns:
            output_data[col] = original_data[col]
    
    # Add predictions
    output_data['predicted_readmission'] = predictions['binary_predictions']
    output_data['risk_category'] = predictions['risk_categories']
    
    if include_probabilities:
        output_data['readmission_probability'] = predictions['probabilities']
    
    # Add clinical recommendations
    recommendations = []
    for i, (risk_cat, prob) in enumerate(zip(predictions['risk_categories'], predictions['probabilities'])):
        if risk_cat == 'Very High':
            rec = "Intensive discharge planning, early follow-up, case management"
        elif risk_cat == 'High':
            rec = "Enhanced discharge planning, follow-up within 7 days"
        elif risk_cat == 'Moderate':
            rec = "Standard discharge protocols, follow-up within 14 days"
        else:
            rec = "Standard care, routine follow-up"
        recommendations.append(rec)
    
    output_data['clinical_recommendations'] = recommendations
    
    # Save to file
    output_data.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Output saved with {len(output_data)} patient predictions")


def create_patient_dashboards(predictions, featured_data, model, dashboard_dir, verbose=False):
    """
    Create individual patient risk dashboards.
    
    Args:
        predictions (dict): Prediction results
        featured_data (pd.DataFrame): Full patient data
        model: Trained model
        dashboard_dir (str): Output directory for dashboards
        verbose (bool): Print verbose output
    """
    if verbose:
        print(f"Creating patient dashboards in {dashboard_dir}")
    
    os.makedirs(dashboard_dir, exist_ok=True)
    
    visualizer = ReadmissionVisualizer()
    
    # Create dashboards for high-risk patients (limit to first 10 for demo)
    high_risk_indices = [i for i, cat in enumerate(predictions['risk_categories']) if cat in ['High', 'Very High']]
    
    dashboard_count = 0
    for i in high_risk_indices[:10]:  # Limit to 10 dashboards
        try:
            # Prepare patient data
            patient_data = {
                'patient_id': featured_data.iloc[i].get('patient_id', f'Patient_{i+1}'),
                'age': featured_data.iloc[i].get('age', 'Unknown'),
                'gender': featured_data.iloc[i].get('gender_std_Female', 0),
                'length_of_stay': featured_data.iloc[i].get('length_of_stay', 'Unknown'),
                'emergency_admission': featured_data.iloc[i].get('emergency_admission', 0),
                'has_diabetes': featured_data.iloc[i].get('has_diabetes', 0),
                'has_hypertension': featured_data.iloc[i].get('has_hypertension', 0),
                'has_heart_disease': featured_data.iloc[i].get('has_heart_disease', 0),
                'has_kidney_disease': featured_data.iloc[i].get('has_kidney_disease', 0),
                'previous_admissions': featured_data.iloc[i].get('previous_admissions', 0),
                'risk_probability': predictions['probabilities'][i]
            }
            
            # Create dashboard
            dashboard_path = os.path.join(dashboard_dir, f"patient_{i+1}_dashboard.png")
            visualizer.create_risk_dashboard(patient_data, model, save_path=dashboard_path)
            dashboard_count += 1
            
        except Exception as e:
            if verbose:
                print(f"Error creating dashboard for patient {i+1}: {str(e)}")
            continue
    
    if verbose:
        print(f"Created {dashboard_count} patient dashboards")


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 60)
    print("HOSPITAL READMISSION RISK MODEL - PREDICTIONS")
    print("=" * 60)
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Model: {args.model}")
        print(f"  Data: {args.data}")
        print(f"  Output: {args.output}")
        print(f"  Threshold: {args.threshold}")
        print(f"  Include Probabilities: {args.include_probabilities}")
        print(f"  Create Dashboards: {args.create_dashboard}")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load model
        print("\n🤖 Loading trained model...")
        model = load_model(args.model, verbose=args.verbose)
        
        # Step 2: Preprocess new data
        print("\n📊 Preprocessing new data...")
        X, featured_data = preprocess_new_data(args.data, model.feature_names, verbose=args.verbose)
        
        # Step 3: Make predictions
        print("\n🔮 Making predictions...")
        predictions = make_predictions(model, X, threshold=args.threshold, verbose=args.verbose)
        
        # Step 4: Create output file
        print("\n💾 Saving predictions...")
        original_data = pd.read_csv(args.data)
        create_output_file(predictions, original_data, args.output, 
                         include_probabilities=args.include_probabilities, verbose=args.verbose)
        
        # Step 5: Create dashboards if requested
        if args.create_dashboard:
            print("\n📈 Creating patient dashboards...")
            create_patient_dashboards(predictions, featured_data, model, 
                                    args.dashboard_dir, verbose=args.verbose)
        
        # Summary
        total_time = (datetime.now() - start_time).total_seconds()
        high_risk_count = sum(1 for cat in predictions['risk_categories'] if cat in ['High', 'Very High'])
        
        print(f"\n✅ PREDICTIONS COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Patients processed: {len(X)}")
        print(f"High-risk patients identified: {high_risk_count} ({high_risk_count/len(X)*100:.1f}%)")
        print(f"Average risk score: {np.mean(predictions['probabilities']):.3f}")
        print(f"Results saved to: {args.output}")
        
        if args.create_dashboard:
            print(f"Patient dashboards saved to: {args.dashboard_dir}")
        
    except Exception as e:
        print(f"\n❌ PREDICTION FAILED: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()