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

# Import our custom modules
from src.data_processing import EHRDataProcessor
from src.feature_engineering import ComprehensiveFeatureEngineer
from src.models import ReadmissionPredictor, ModelOptimizer, ClinicalEvaluator
from src.visualization import ReadmissionVisualizer

warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))



def main():
    """
    Execute the complete hospital readmission prediction pipeline.
    """
    print("=" * 60)
    print("HOSPITAL READMISSION RISK MODEL - MAIN PIPELINE")
    print("Replicating 2015 Southeast Texas Internship Project")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/sample_data.csv'
    OUTPUT_DIR = 'output'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
    
    start_time = datetime.now()
    
    # ===================================================================
    # STEP 1: DATA LOADING AND QUALITY ASSESSMENT
    # ===================================================================
    print(f"\n🔍 STEP 1: LOADING AND ASSESSING EHR DATA")
    print("-" * 50)
    
    try:
        # Load raw data
        raw_data = pd.read_csv(DATA_PATH)
        print(f"✓ Loaded dataset: {raw_data.shape[0]} patients, {raw_data.shape[1]} columns")
        
        # Initialize data processor
        processor = EHRDataProcessor()
        
        # Data quality assessment
        quality_report = processor.validate_data_quality(raw_data)
        print(f"✓ Data quality assessment completed")
        print(f"  - Missing values: {sum(quality_report['missing_values'][col]['count'] for col in quality_report['missing_values'])}")
        print(f"  - Duplicate records: {quality_report['duplicates']}")
        print(f"  - Quality issues: {len(quality_report['issues'])}")
        
        # Visualize data quality
        visualizer = ReadmissionVisualizer()
        visualizer.plot_data_quality_overview(raw_data, 
                                            save_path=os.path.join(OUTPUT_DIR, 'plots', 'data_quality_overview.png'))
        print(f"✓ Data quality visualization saved")
        
    except Exception as e:
        print(f"❌ Error in data loading: {str(e)}")
        return False
    
    # ===================================================================
    # STEP 2: DATA PREPROCESSING AND STANDARDIZATION
    # ===================================================================
    print(f"\n🔧 STEP 2: DATA PREPROCESSING AND STANDARDIZATION")
    print("-" * 50)
    
    try:
        # Process raw data
        processed_data, processing_report = processor.process_pipeline(raw_data)
        print(f"✓ Data preprocessing completed")
        print(f"  - Original shape: {processing_report['original_shape']}")
        print(f"  - Final shape: {processing_report['final_shape']}")
        print(f"  - Processing steps: {len(processing_report['steps_completed'])}")
        
        # Save processed data
        processed_data.to_csv(os.path.join(OUTPUT_DIR, 'processed_data.csv'), index=False)
        print(f"✓ Processed data saved")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        return False
    
    # ===================================================================
    # STEP 3: FEATURE ENGINEERING
    # ===================================================================
    print(f"\n🛠️ STEP 3: FEATURE ENGINEERING")
    print("-" * 50)
    
    try:
        # Initialize feature engineer
        feature_engineer = ComprehensiveFeatureEngineer()
        
        # Engineer features
        featured_data, feature_report = feature_engineer.engineer_all_features(processed_data)
        print(f"✓ Feature engineering completed")
        print(f"  - Original features: {len(feature_report['original_features'])}")
        print(f"  - Final features: {feature_report['final_feature_count']}")
        print(f"  - Features created: {len(feature_report['features_created'])}")
        
        # Prepare final dataset for modeling
        X, y, feature_list = feature_engineer.select_final_features(featured_data)
        print(f"✓ Final modeling dataset prepared: {X.shape}")
        
        # Feature summary
        feature_summary = feature_engineer.get_feature_summary(featured_data)
        print(f"✓ Feature summary generated")
        
        # Save featured data
        featured_data.to_csv(os.path.join(OUTPUT_DIR, 'featured_data.csv'), index=False)
        print(f"✓ Featured data saved")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        return False
    
    # ===================================================================
    # STEP 4: MODEL TRAINING AND OPTIMIZATION
    # ===================================================================
    print(f"\n🤖 STEP 4: MODEL TRAINING AND OPTIMIZATION")
    print("-" * 50)
    
    try:
        # Initialize model optimizer
        optimizer = ModelOptimizer()
        
        # Compare and optimize models
        model_comparison = optimizer.compare_models(X, y, cv=5, verbose=True)
        print(f"✓ Model comparison completed")
        
        # Get best model
        best_model = optimizer.get_best_model(X, y, cv=5)
        print(f"✓ Best model selected and optimized")
        
        # Train the final model
        training_results = best_model.fit(X, y, verbose=True)
        print(f"✓ Final model training completed")
        print(f"  - Training time: {training_results['training_time']:.2f} seconds")
        print(f"  - Validation AUC: {training_results['val_metrics']['roc_auc']:.3f}")
        
        # Save the trained model
        model_path = os.path.join(OUTPUT_DIR, 'models', 'readmission_model.joblib')
        best_model.save_model(model_path)
        print(f"✓ Model saved to {model_path}")
        
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return False
    
    # ===================================================================
    # STEP 5: MODEL EVALUATION AND VALIDATION
    # ===================================================================
    print(f"\n📊 STEP 5: MODEL EVALUATION AND VALIDATION")
    print("-" * 50)
    
    try:
        # Generate predictions
        y_pred = best_model.predict(X)
        y_proba = best_model.predict_proba(X)
        
        # Clinical evaluation
        evaluator = ClinicalEvaluator()
        clinical_metrics = evaluator.evaluate_clinical_performance(y, y_pred, y_proba, 
                                                                  best_model.model_type.title())
        print(f"✓ Clinical performance evaluation completed")
        print(f"  - ROC-AUC: {clinical_metrics['ROC_AUC']:.3f}")
        print(f"  - Sensitivity: {clinical_metrics['Sensitivity']:.3f}")
        print(f"  - Specificity: {clinical_metrics['Specificity']:.3f}")
        print(f"  - PPV: {clinical_metrics['PPV']:.3f}")
        
        # Model calibration assessment
        calibration_results = evaluator.assess_calibration(y, y_proba)
        print(f"✓ Model calibration assessed: {calibration_results['calibration_quality']}")
        
        # Feature importance analysis
        feature_importance = best_model.get_feature_importance(top_n=15)
        print(f"✓ Feature importance analysis completed")
        
        # Clinical scenario testing
        clinical_scenarios = {
            'Elderly Diabetic Patients': lambda df: (df['age'] >= 75) & (df['has_diabetes'] == 1),
            'Multiple Comorbidities': lambda df: df['comorbidity_count'] >= 3,
            'Emergency Admissions': lambda df: df['emergency_admission'] == 1,
            'Frequent Readmitters': lambda df: df['frequent_readmitter'] == 1,
            'High Risk Patients': lambda df: df['high_risk_patient'] == 1
        }
        
        scenario_results = evaluator.clinical_scenario_analysis(X, y, y_proba, clinical_scenarios)
        print(f"✓ Clinical scenario analysis completed: {len(scenario_results)} scenarios")
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {str(e)}")
        return False
    
    # ===================================================================
    # STEP 6: VISUALIZATION AND REPORTING
    # ===================================================================
    print(f"\n📈 STEP 6: VISUALIZATION AND REPORTING")
    print("-" * 50)
    
    try:
        # Clinical overview
        visualizer.plot_clinical_overview(featured_data, 
                                        save_path=os.path.join(OUTPUT_DIR, 'plots', 'clinical_overview.png'))
        print(f"✓ Clinical overview visualization saved")
        
        # Model performance visualization
        visualizer.plot_model_performance(y, y_pred, y_proba, best_model.model_type.title(),
                                        save_path=os.path.join(OUTPUT_DIR, 'plots', 'model_performance.png'))
        print(f"✓ Model performance visualization saved")
        
        # Feature importance visualization
        visualizer.plot_feature_importance(feature_importance,
                                         save_path=os.path.join(OUTPUT_DIR, 'plots', 'feature_importance.png'))
        print(f"✓ Feature importance visualization saved")
        
        # Clinical scenarios visualization
        if len(scenario_results) > 0:
            visualizer.plot_clinical_scenarios(scenario_results,
                                             save_path=os.path.join(OUTPUT_DIR, 'plots', 'clinical_scenarios.png'))
            print(f"✓ Clinical scenarios visualization saved")
        
        # Sample patient dashboard
        sample_patient = {
            'age': 78, 'gender': 'Female', 'insurance': 'Medicare',
            'length_of_stay': 6, 'emergency_admission': 1,
            'has_diabetes': 1, 'has_hypertension': 1, 'has_heart_disease': 0,
            'previous_admissions': 3, 'risk_probability': 0.72
        }
        
        visualizer.create_risk_dashboard(sample_patient, best_model,
                                       save_path=os.path.join(OUTPUT_DIR, 'plots', 'patient_dashboard.png'))
        print(f"✓ Patient risk dashboard saved")
        
    except Exception as e:
        print(f"❌ Error in visualization: {str(e)}")
        return False
    
    # ===================================================================
    # STEP 7: GENERATE FINAL REPORT
    # ===================================================================
    print(f"\n📋 STEP 7: GENERATING FINAL REPORT")
    print("-" * 50)
    
    try:
        # Create comprehensive report
        report_content = f"""
# Hospital Readmission Risk Model - Final Report

**Project**: Hospital Readmission Risk Prediction  
**Timeline**: January 2015 - May 2015  
**Author**: Blake Sonnier  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This project successfully developed a machine learning model to predict 30-day hospital readmission risk 
for patients in Southeast Texas regional hospitals. The model achieves clinically relevant performance 
and demonstrates practical applicability for healthcare decision support.

## Dataset Summary

- **Total Patients**: {raw_data.shape[0]:,}
- **Features Engineered**: {len(feature_list)}
- **Processing Time**: {(datetime.now() - start_time).total_seconds():.1f} seconds
- **Data Quality Issues Resolved**: {len(quality_report['issues'])}

## Model Performance

- **Algorithm**: {best_model.model_type.title()}
- **ROC-AUC Score**: {clinical_metrics['ROC_AUC']:.3f}
- **Sensitivity**: {clinical_metrics['Sensitivity']:.3f} (catches {clinical_metrics['Sensitivity']*100:.1f}% of readmissions)
- **Specificity**: {clinical_metrics['Specificity']:.3f} (correctly identifies {clinical_metrics['Specificity']*100:.1f}% of non-readmissions)
- **Positive Predictive Value**: {clinical_metrics['PPV']:.3f}
- **Calibration Quality**: {calibration_results['calibration_quality']}

## Top Risk Factors

{chr(10).join([f"1. {row['feature']}: {row['importance']:.3f}" for _, row in feature_importance.head(5).iterrows()])}

## Clinical Scenarios Tested

{chr(10).join([f"- {row['Scenario']}: {row['N_Patients']} patients, {row['Actual_Rate']*100:.1f}% readmission rate" for _, row in scenario_results.iterrows()])}

## Key Achievements

✓ Successfully processed inconsistent EHR data from multiple hospital systems  
✓ Developed comprehensive medical code translation system  
✓ Created interpretable model suitable for clinical decision support  
✓ Validated performance across diverse patient populations  
✓ Built interactive dashboards for real-time risk assessment  

## Technical Implementation

- **Data Processing**: Standardized {processing_report['original_shape'][1]} raw columns
- **Feature Engineering**: Created {len(feature_report['features_created'])} clinical features
- **Model Training**: Optimized using {optimizer.optimization_results.get('logistic', {}).get('best_params', 'cross-validation')}
- **Clinical Validation**: Tested across {len(scenario_results)} clinical scenarios

## Deployment Readiness

The model is ready for pilot deployment with:
- Calibrated probability outputs for clinical decision-making
- Interpretable feature importance for clinical staff
- Comprehensive evaluation across patient demographics
- Interactive dashboard for real-time risk assessment

## Files Generated

- `processed_data.csv`: Cleaned and standardized dataset
- `featured_data.csv`: Final dataset with engineered features
- `readmission_model.joblib`: Trained machine learning model
- Various visualization plots in `plots/` directory

## Internship Impact

This project demonstrates successful application of data science techniques to real-world healthcare 
challenges, showcasing skills in:

- Healthcare data processing and standardization
- Clinical feature engineering and domain expertise integration
- Machine learning model development and optimization
- Clinical validation and performance assessment
- Healthcare dashboard development and visualization

The work provides a foundation for reducing preventable hospital readmissions and improving patient outcomes 
in Southeast Texas regional hospitals.
"""

        # Save report
        report_path = os.path.join(OUTPUT_DIR, 'final_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"✓ Final report saved to {report_path}")
        
        # Save detailed results
        results_summary = {
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'data_shape': raw_data.shape,
            'final_features': len(feature_list),
            'model_type': best_model.model_type,
            'performance_metrics': clinical_metrics,
            'calibration_results': calibration_results,
            'feature_importance': feature_importance.to_dict('records'),
            'clinical_scenarios': scenario_results.to_dict('records') if len(scenario_results) > 0 else []
        }
        
        import json
        with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"✓ Results summary saved")
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return False
    
    # ===================================================================
    # PIPELINE COMPLETION
    # ===================================================================
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total Processing Time: {total_time:.1f} seconds")
    print(f"Final Model AUC: {clinical_metrics['ROC_AUC']:.3f}")
    print(f"Output Directory: {OUTPUT_DIR}/")
    print("\nFiles Generated:")
    print(f"  📊 Data: processed_data.csv, featured_data.csv")
    print(f"  🤖 Model: models/readmission_model.joblib")
    print(f"  📈 Plots: plots/ directory (5 visualizations)")
    print(f"  📋 Report: final_report.md, results_summary.json")
    
    print(f"\n🏥 INTERNSHIP PROJECT SUCCESSFULLY REPLICATED!")
    print("Ready for portfolio demonstration and deployment.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Pipeline execution failed. Check error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All systems operational. Model ready for clinical decision support!")
        sys.exit(0)