#!/usr/bin/env python3
"""
Hospital Readmission Risk Model - Main Pipeline Execution

This script demonstrates the complete end-to-end pipeline using all src/ modules.
Replicates the workflow from the 2015 internship project.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
Usage: python main.py
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# FIXED: Add src directory to Python path BEFORE imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import our custom modules
try:
    from data_processing import EHRDataProcessor
    from feature_engineering import ComprehensiveFeatureEngineer
    from models import ReadmissionPredictor, ModelOptimizer, ClinicalEvaluator
    from visualization import ReadmissionVisualizer
    logger.info("Successfully imported all custom modules")
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    sys.exit(1)

# Configuration
class Config:
    """Configuration class for the pipeline."""
    def __init__(self):
        self.DATA_PATH = 'data/sample_data.csv'
        self.OUTPUT_DIR = 'output'
        self.PLOTS_DIR = os.path.join(self.OUTPUT_DIR, 'plots')
        self.MODELS_DIR = os.path.join(self.OUTPUT_DIR, 'models')
        self.LOG_FILE = 'pipeline.log'
        
    def validate_paths(self):
        """Validate that required paths exist."""
        # Check if data file exists
        if not os.path.exists(self.DATA_PATH):
            logger.warning(f"Data file {self.DATA_PATH} not found. Will generate synthetic data.")
            return False
        return True
    
    def create_directories(self):
        """Create output directories if they don't exist."""
        directories = [self.OUTPUT_DIR, self.PLOTS_DIR, self.MODELS_DIR]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")

def validate_environment():
    """Validate that the environment is properly set up."""
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True

def generate_sample_data():
    """Generate synthetic data if the data file doesn't exist."""
    logger.info("Generating synthetic data...")
    np.random.seed(42)
    n_patients = 1000
    
    data = pd.DataFrame({
        'patient_id': [f"PID_{i:06d}" for i in range(n_patients)],
        'age': np.random.normal(65, 15, n_patients),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'admission_date': pd.date_range('2014-01-01', periods=n_patients, freq='H'),
        'length_of_stay': np.random.exponential(5, n_patients),
        'diagnosis_codes': [
            np.random.choice(['250.00;401.9', 'E11.9;I10', 'DM2;HTN', '428.0;585.9'])
            for _ in range(n_patients)
        ],
        'previous_admissions': np.random.poisson(1.5, n_patients),
        'emergency_admission': np.random.binomial(1, 0.6, n_patients),
        'insurance_type': np.random.choice(['Medicare', 'Private', 'Medicaid'], n_patients),
        'readmission_30_day': np.random.binomial(1, 0.15, n_patients)
    })
    
    return data

def safe_step(step_name, step_function, *args, **kwargs):
    """Execute a pipeline step with error handling."""
    try:
        logger.info(f"Starting {step_name}...")
        start_time = datetime.now()
        result = step_function(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"{step_name} completed in {duration:.2f} seconds")
        return result, True
    except Exception as e:
        logger.error(f"{step_name} failed: {str(e)}")
        logger.exception("Full traceback:")
        return None, False

def cleanup_resources():
    """Clean up temporary resources."""
    try:
        # Clear matplotlib figures to free memory
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("✓ Resources cleaned up")
    except Exception as e:
        logger.warning(f"Warning during cleanup: {e}")

def main():
    """
    Execute the complete hospital readmission prediction pipeline.
    """
    pipeline_start_time = datetime.now()
    
    print("=" * 60)
    print("HOSPITAL READMISSION RISK MODEL - MAIN PIPELINE")
    print("Replicating 2015 Southeast Texas Internship Project")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    try:
        # Step 0: Environment validation
        result, success = safe_step(
            "Environment Validation",
            validate_environment
        )
        if not success:
            return False
        
        # Step 1: Setup directories and paths
        result, success = safe_step(
            "Directory Setup",
            config.create_directories
        )
        if not success:
            return False
        
        # Step 2: Data loading
        data_exists = config.validate_paths()
        if data_exists:
            raw_data, success = safe_step(
                "Data Loading",
                pd.read_csv,
                config.DATA_PATH
            )
        else:
            raw_data, success = safe_step(
                "Synthetic Data Generation",
                generate_sample_data
            )
            # Save synthetic data for future use
            if success and raw_data is not None:
                os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
                raw_data.to_csv(config.DATA_PATH, index=False)
                logger.info(f"Synthetic data saved to {config.DATA_PATH}")
        
        if not success or raw_data is None:
            logger.error("Failed to load or generate data")
            return False
        
        logger.info(f"Dataset loaded: {raw_data.shape[0]} patients, {raw_data.shape[1]} columns")
        
        # Step 3: Data processing
        processor = EHRDataProcessor()
        processed_data, success = safe_step(
            "Data Processing",
            processor.process_pipeline,
            raw_data
        )
        if not success:
            return False
        
        processed_data, processing_report = processed_data
        logger.info(f"Data processing: {processing_report['original_shape']} -> {processing_report['final_shape']}")
        
        # Step 4: Feature engineering
        feature_engineer = ComprehensiveFeatureEngineer()
        featured_data, success = safe_step(
            "Feature Engineering",
            feature_engineer.engineer_all_features,
            processed_data
        )
        if not success:
            return False
        
        featured_data, feature_report = featured_data
        logger.info(f"Feature engineering: {len(feature_report['original_features'])} -> {feature_report['final_feature_count']} features")
        
        # Step 5: Prepare modeling data
        modeling_data, success = safe_step(
            "Modeling Data Preparation",
            feature_engineer.select_final_features,
            featured_data
        )
        if not success:
            return False
        
        X, y, feature_list = modeling_data
        logger.info(f"Modeling dataset prepared: {X.shape}")
        
        # Step 6: Model training and optimization
        optimizer = ModelOptimizer()
        model_comparison, success = safe_step(
            "Model Comparison",
            optimizer.compare_models,
            X, y, 5, True
        )
        if not success:
            return False
        
        best_model, success = safe_step(
            "Best Model Selection",
            optimizer.get_best_model,
            X, y, 5
        )
        if not success:
            return False
        
        training_results, success = safe_step(
            "Model Training",
            best_model.fit,
            X, y, 0.2, True
        )
        if not success:
            return False
        
        # Step 7: Model evaluation
        evaluator = ClinicalEvaluator()
        y_pred = best_model.predict(X)
        y_proba = best_model.predict_proba(X)
        
        clinical_metrics, success = safe_step(
            "Clinical Evaluation",
            evaluator.evaluate_clinical_performance,
            y, y_pred, y_proba, best_model.model_type.title()
        )
        if not success:
            return False
        
        logger.info(f"Model performance - ROC-AUC: {clinical_metrics['ROC_AUC']:.3f}")
        
        # Step 8: Visualization
        visualizer = ReadmissionVisualizer()
        
        viz_steps = [
            ("Clinical Overview", visualizer.plot_clinical_overview, 
             featured_data, os.path.join(config.PLOTS_DIR, 'clinical_overview.png')),
            ("Model Performance", visualizer.plot_model_performance,
             y, y_pred, y_proba, best_model.model_type.title(), 
             os.path.join(config.PLOTS_DIR, 'model_performance.png')),
            ("Feature Importance", visualizer.plot_feature_importance,
             best_model.get_feature_importance(),
             os.path.join(config.PLOTS_DIR, 'feature_importance.png'))
        ]
        
        for viz_name, viz_func, *viz_args in viz_steps:
            result, success = safe_step(f"Visualization: {viz_name}", viz_func, *viz_args)
            if not success:
                logger.warning(f"Visualization {viz_name} failed, continuing...")
        
        # Step 9: Save model
        model_path = os.path.join(config.MODELS_DIR, 'readmission_model.joblib')
        result, success = safe_step(
            "Model Saving",
            best_model.save_model,
            model_path
        )
        
        # Step 10: Generate final report
        total_time = (datetime.now() - pipeline_start_time).total_seconds()
        
        report_content = f"""
# Hospital Readmission Risk Model - Pipeline Execution Report

**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Execution Time:** {total_time:.1f} seconds

## Pipeline Results
- **Dataset:** {raw_data.shape[0]} patients processed
- **Final Features:** {len(feature_list)}
- **Model Type:** {best_model.model_type.title()}
- **Performance (ROC-AUC):** {clinical_metrics['ROC_AUC']:.3f}
- **Sensitivity:** {clinical_metrics['Sensitivity']:.3f}
- **Specificity:** {clinical_metrics['Specificity']:.3f}

## Files Generated
- Model: {model_path}
- Visualizations: {config.PLOTS_DIR}/
- Log: {config.LOG_FILE}

## Status: ✅ SUCCESS
Pipeline completed successfully with all major steps executed.
"""
        
        report_path = os.path.join(config.OUTPUT_DIR, 'pipeline_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Model AUC: {clinical_metrics['ROC_AUC']:.3f}")
        print(f"Files saved in: {config.OUTPUT_DIR}/")
        print(f"Report: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {e}")
        logger.exception("Full traceback:")
        return False
    
    finally:
        # Always cleanup resources
        cleanup_resources()
        total_time = (datetime.now() - pipeline_start_time).total_seconds()
        logger.info(f"Pipeline finished. Total time: {total_time:.1f} seconds")

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Pipeline execution failed. Check logs for details.")
        sys.exit(1)
    else:
        print("\n✅ All systems operational. Model ready for deployment!")
        sys.exit(0)
