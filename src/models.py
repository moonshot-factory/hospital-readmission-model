"""
Hospital Readmission Risk Model - Machine Learning Models Module

This module contains model classes and training utilities for the
hospital readmission prediction project.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


class ReadmissionPredictor:
    """
    Main class for hospital readmission risk prediction models.
    Focuses on interpretability and clinical applicability.
    """
    
    def __init__(self, model_type='logistic', random_state=42):
        """
        Initialize the readmission predictor.
        
        Args:
            model_type (str): Type of model ('logistic', 'decision_tree', 'random_forest')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        self.training_history = {}
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            )
            self.requires_scaling = True
            
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                random_state=self.random_state,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            )
            self.requires_scaling = False
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                criterion='gini',
                max_depth=None
            )
            self.requires_scaling = False
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y, validation_split=0.2, verbose=True):
        """
        Train the readmission prediction model.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            y (pd.Series or np.array): Target variable
            validation_split (float): Fraction of data to use for validation
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results and metrics
        """
        if verbose:
            print(f"Training {self.model_type} model for readmission prediction...")
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Scale features if required
        if self.requires_scaling:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Train model
        start_time = datetime.now()
        self.model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Generate predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        training_results = {
            'training_time': training_time,
            'train_metrics': self._calculate_metrics(y_train, y_train_pred, y_train_proba),
            'val_metrics': self._calculate_metrics(y_val, y_val_pred, y_val_proba),
            'model_params': self.model.get_params(),
            'feature_count': len(self.feature_names)
        }
        
        # Store training history
        self.training_history = training_results
        self.is_fitted = True
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Validation ROC-AUC: {training_results['val_metrics']['roc_auc']:.3f}")
            print(f"Validation Precision: {training_results['val_metrics']['precision']:.3f}")
            print(f"Validation Recall: {training_results['val_metrics']['recall']:.3f}")
        
        return training_results
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            
        Returns:
            np.array: Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features if required
        if self.requires_scaling and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            
        Returns:
            np.array: Probability of readmission (positive class)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features if required
        if self.requires_scaling and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance scores.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if self.model_type == 'logistic':
            # Use absolute coefficients for logistic regression
            importance_scores = np.abs(self.model.coef_[0])
        elif self.model_type in ['decision_tree', 'random_forest']:
            # Use built-in feature importance
            importance_scores = self.model.feature_importances_
        else:
            raise ValueError(f"Feature importance not implemented for {self.model_type}")
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n) if top_n else importance_df
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical metrics
        metrics.update({
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
        
        return metrics
    
    def cross_validate(self, X, y, cv=5, scoring='roc_auc', verbose=True):
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            y (pd.Series or np.array): Target variable
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for cross-validation
            verbose (bool): Whether to print results
            
        Returns:
            dict: Cross-validation results
        """
        if verbose:
            print(f"Performing {cv}-fold cross-validation...")
        
        # Scale features if required
        if self.requires_scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring=scoring)
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scoring_metric': scoring
        }
        
        if verbose:
            print(f"CV {scoring}: {results['mean_score']:.3f} (+/- {results['std_score'] * 2:.3f})")
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'requires_scaling': self.requires_scaling
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            ReadmissionPredictor: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data['training_history']
        instance.requires_scaling = model_data['requires_scaling']
        instance.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        return instance


class ModelOptimizer:
    """
    Class for hyperparameter optimization and model selection.
    """
    
    def __init__(self, random_state=42):
        """Initialize model optimizer."""
        self.random_state = random_state
        self.optimization_results = {}
    
    def optimize_logistic_regression(self, X, y, cv=5, verbose=True):
        """
        Optimize logistic regression hyperparameters.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            y (pd.Series or np.array): Target variable
            cv (int): Number of cross-validation folds
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Optimization results
        """
        if verbose:
            print("Optimizing Logistic Regression hyperparameters...")
        
        # Parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Grid search
        grid_search = GridSearchCV(
            LogisticRegression(random_state=self.random_state, max_iter=1000),
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_scaled, y)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_
        }
        
        self.optimization_results['logistic'] = results
        
        if verbose:
            print(f"Best parameters: {results['best_params']}")
            print(f"Best CV score: {results['best_score']:.3f}")
        
        return results
    
    def optimize_decision_tree(self, X, y, cv=5, verbose=True):
        """
        Optimize decision tree hyperparameters.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            y (pd.Series or np.array): Target variable
            cv (int): Number of cross-validation folds
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Optimization results
        """
        if verbose:
            print("Optimizing Decision Tree hyperparameters...")
        
        # Parameter grid focused on preventing overfitting
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        
        # Grid search
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X, y)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_
        }
        
        self.optimization_results['decision_tree'] = results
        
        if verbose:
            print(f"Best parameters: {results['best_params']}")
            print(f"Best CV score: {results['best_score']:.3f}")
        
        return results
    
    def compare_models(self, X, y, cv=5, verbose=True):
        """
        Compare multiple model types with optimized hyperparameters.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            y (pd.Series or np.array): Target variable
            cv (int): Number of cross-validation folds
            verbose (bool): Whether to print results
            
        Returns:
            pd.DataFrame: Model comparison results
        """
        if verbose:
            print("Comparing optimized models...")
        
        # Optimize each model type
        lr_results = self.optimize_logistic_regression(X, y, cv, verbose=False)
        dt_results = self.optimize_decision_tree(X, y, cv, verbose=False)
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Logistic Regression
        comparison_data.append({
            'Model': 'Logistic Regression',
            'Best_CV_Score': lr_results['best_score'],
            'Best_Params': str(lr_results['best_params']),
            'Interpretability': 'High',
            'Training_Speed': 'Fast',
            'Clinical_Utility': 'High'
        })
        
        # Decision Tree
        comparison_data.append({
            'Model': 'Decision Tree',
            'Best_CV_Score': dt_results['best_score'],
            'Best_Params': str(dt_results['best_params']),
            'Interpretability': 'Very High',
            'Training_Speed': 'Fast',
            'Clinical_Utility': 'Very High'
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Best_CV_Score', ascending=False)
        
        if verbose:
            print("\nModel Comparison Results:")
            print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_best_model(self, X, y, cv=5):
        """
        Get the best performing model after optimization.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix
            y (pd.Series or np.array): Target variable
            cv (int): Number of cross-validation folds
            
        Returns:
            ReadmissionPredictor: Best performing model
        """
        comparison_df = self.compare_models(X, y, cv, verbose=False)
        best_model_name = comparison_df.iloc[0]['Model']
        
        if 'Logistic' in best_model_name:
            best_params = self.optimization_results['logistic']['best_params']
            model = ReadmissionPredictor('logistic')
            model.model.set_params(**best_params)
        else:  # Decision Tree
            best_params = self.optimization_results['decision_tree']['best_params']
            model = ReadmissionPredictor('decision_tree')
            model.model.set_params(**best_params)
        
        print(f"Best model: {best_model_name}")
        print(f"Best parameters: {best_params}")
        
        return model


class ClinicalEvaluator:
    """
    Specialized evaluator for clinical model assessment.
    """
    
    def __init__(self):
        """Initialize clinical evaluator."""
        pass
    
    def evaluate_clinical_performance(self, y_true, y_pred, y_proba, model_name="Model"):
        """
        Comprehensive clinical performance evaluation.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_proba (np.array): Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            dict: Clinical evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Clinical utility metrics
        nns = 1 / precision if precision > 0 else float('inf')  # Number needed to screen
        likelihood_ratio_pos = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
        likelihood_ratio_neg = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
        
        # Calibration metrics
        brier_score = brier_score_loss(y_true, y_proba)
        
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC_AUC': roc_auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'NNS': nns,
            'LR_positive': likelihood_ratio_pos,
            'LR_negative': likelihood_ratio_neg,
            'Brier_Score': brier_score,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        }
    
    def assess_calibration(self, y_true, y_proba, n_bins=10):
        """
        Assess model calibration.
        
        Args:
            y_true (np.array): True labels
            y_proba (np.array): Predicted probabilities
            n_bins (int): Number of bins for calibration curve
            
        Returns:
            dict: Calibration assessment results
        """
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )
        
        # Brier score
        brier_score = brier_score_loss(y_true, y_proba)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Calibration quality assessment
        if ece < 0.05:
            quality = "Excellent"
        elif ece < 0.10:
            quality = "Good"
        elif ece < 0.15:
            quality = "Fair"
        else:
            quality = "Poor"
        
        return {
            'brier_score': brier_score,
            'ece': ece,
            'calibration_quality': quality,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    
    def clinical_scenario_analysis(self, X, y_true, y_proba, scenarios):
        """
        Analyze model performance on specific clinical scenarios.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y_true (np.array): True labels
            y_proba (np.array): Predicted probabilities
            scenarios (dict): Dictionary of scenario names and condition functions
            
        Returns:
            pd.DataFrame: Scenario analysis results
        """
        results = []
        
        for scenario_name, condition_func in scenarios.items():
            try:
                # Apply scenario condition
                scenario_mask = condition_func(X)
                
                if scenario_mask.sum() == 0:
                    continue
                
                scenario_y_true = y_true[scenario_mask]
                scenario_y_proba = y_proba[scenario_mask]
                
                # Calculate metrics for this scenario
                actual_rate = scenario_y_true.mean()
                predicted_rate = scenario_y_proba.mean()
                n_patients = len(scenario_y_true)
                
                if len(scenario_y_true.unique()) > 1:
                    auc = roc_auc_score(scenario_y_true, scenario_y_proba)
                else:
                    auc = np.nan
                
                results.append({
                    'Scenario': scenario_name,
                    'N_Patients': n_patients,
                    'Actual_Rate': actual_rate,
                    'Predicted_Rate': predicted_rate,
                    'AUC': auc
                })
                
            except Exception as e:
                print(f"Error processing scenario {scenario_name}: {str(e)}")
                continue
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    print("Machine Learning Models Module - Example Usage")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.15, n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Sample data: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Positive class rate: {y.mean()*100:.1f}%")
    
    # Test ReadmissionPredictor
    print("\n1. Testing ReadmissionPredictor...")
    model = ReadmissionPredictor('logistic')
    training_results = model.fit(X_df, y, verbose=True)
    
    # Test predictions
    predictions = model.predict(X_df[:10])
    probabilities = model.predict_proba(X_df[:10])
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities: {probabilities}")
    
    # Test feature importance
    importance = model.get_feature_importance(top_n=5)
    print(f"\nTop 5 features:")
    print(importance)
    
    # Test ModelOptimizer
    print("\n2. Testing ModelOptimizer...")
    optimizer = ModelOptimizer()
    comparison = optimizer.compare_models(X_df, y, cv=3, verbose=True)
    
    # Test ClinicalEvaluator
    print("\n3. Testing ClinicalEvaluator...")
    evaluator = ClinicalEvaluator()
    
    y_pred = model.predict(X_df)
    y_proba = model.predict_proba(X_df)
    
    clinical_metrics = evaluator.evaluate_clinical_performance(y, y_pred, y_proba, "Test Model")
    print(f"\nClinical metrics:")
    for key, value in clinical_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test calibration assessment
    calibration = evaluator.assess_calibration(y, y_proba)
    print(f"\nCalibration assessment:")
    print(f"  Brier Score: {calibration['brier_score']:.4f}")
    print(f"  ECE: {calibration['ece']:.4f}")
    print(f"  Quality: {calibration['calibration_quality']}")
    
    print("\nModule testing complete!")