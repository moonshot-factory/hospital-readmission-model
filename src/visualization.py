"""
Hospital Readmission Risk Model - Visualization Module

This module contains visualization functions and dashboard components
for the hospital readmission prediction project.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# Scikit-learn imports for metrics
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False



class ReadmissionVisualizer:
    """
    Main visualization class for readmission risk analysis.
    """
    
    def __init__(self, style='healthcare'):
        """
        Initialize the visualizer with healthcare-appropriate styling.
        
        Args:
            style (str): Visual style ('healthcare', 'clinical', 'presentation')
        """
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Set up matplotlib and seaborn styling."""
        if self.style == 'healthcare':
            # Professional healthcare color palette
            self.colors = {
                'primary': '#2E86AB',      # Medical blue
                'secondary': '#A23B72',    # Deep pink
                'accent': '#F18F01',       # Orange
                'success': '#C73E1D',      # Red for alerts
                'neutral': '#6B7280',      # Gray
                'background': '#F9FAFB'    # Light gray
            }
            sns.set_palette([self.colors['primary'], self.colors['secondary'], 
                           self.colors['accent'], self.colors['success']])
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def plot_data_quality_overview(self, df, save_path=None):
        """
        Create comprehensive data quality visualization.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('EHR Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # Missing values heatmap
        ax = axes[0, 0]
        missing_data = df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data.iloc[:100], cbar=True, ax=ax, cmap='viridis')
            ax.set_title('Missing Values Pattern (First 100 Records)')
            ax.set_ylabel('Patient Records')
        else:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Missing Values Pattern')
        
        # Missing values by column
        ax = axes[0, 1]
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_cols = missing_pct[missing_pct > 0]
        
        if len(missing_cols) > 0:
            missing_cols.plot(kind='bar', ax=ax, color=self.colors['accent'])
            ax.set_title('Missing Values by Column (%)')
            ax.set_ylabel('Percentage Missing')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Missing Values by Column')
        
        # Data types distribution
        ax = axes[0, 2]
        dtype_counts = df.dtypes.value_counts()
        ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        ax.set_title('Data Types Distribution')
        
        # Numeric columns distribution
        ax = axes[1, 0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sample_col = numeric_cols[0]
            df[sample_col].hist(bins=30, ax=ax, color=self.colors['primary'], alpha=0.7)
            ax.set_title(f'Sample Distribution: {sample_col}')
            ax.set_ylabel('Frequency')
        
        # Categorical columns
        ax = axes[1, 1]
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            sample_cat_col = categorical_cols[0]
            value_counts = df[sample_cat_col].value_counts().head(10)
            value_counts.plot(kind='bar', ax=ax, color=self.colors['secondary'])
            ax.set_title(f'Sample Categories: {sample_cat_col}')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        # Dataset overview
        ax = axes[1, 2]
        overview_text = f"""
        Dataset Overview:
        
        • Total Records: {len(df):,}
        • Total Features: {len(df.columns)}
        • Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}
        • Categorical Features: {len(df.select_dtypes(include=['object']).columns)}
        • Missing Values: {df.isnull().sum().sum():,}
        • Duplicates: {df.duplicated().sum():,}
        • Memory Usage: {df.memory_usage().sum() / 1024**2:.1f} MB
        """
        ax.text(0.1, 0.9, overview_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Dataset Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_clinical_overview(self, df, target_col='readmission_30_day', save_path=None):
        """
        Create clinical data overview visualization.
        
        Args:
            df (pd.DataFrame): Dataset with clinical features
            target_col (str): Target variable column name
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Clinical Data Overview', fontsize=16, fontweight='bold')
        
        # Target variable distribution
        ax = axes[0, 0]
        if target_col in df.columns:
            target_counts = df[target_col].value_counts()
            labels = ['No Readmission', 'Readmission'] if len(target_counts) == 2 else target_counts.index
            colors = [self.colors['primary'], self.colors['accent']]
            ax.pie(target_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.set_title(f'Readmission Rate\n({target_counts[1]/len(df)*100:.1f}% positive)')
        else:
            ax.text(0.5, 0.5, 'Target variable\nnot available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Readmission Rate')
        
        # Age distribution
        ax = axes[0, 1]
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            df[age_col].hist(bins=30, ax=ax, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(df[age_col].mean(), color=self.colors['accent'], linestyle='--', 
                      label=f'Mean: {df[age_col].mean():.1f}')
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Length of stay distribution
        ax = axes[0, 2]
        los_cols = [col for col in df.columns if 'length' in col.lower() and 'stay' in col.lower()]
        if los_cols:
            los_col = los_cols[0]
            df[los_col].hist(bins=30, ax=ax, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            ax.axvline(df[los_col].mean(), color=self.colors['accent'], linestyle='--',
                      label=f'Mean: {df[los_col].mean():.1f}')
            ax.set_title('Length of Stay Distribution')
            ax.set_xlabel('Days')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Medical conditions prevalence
        ax = axes[1, 0]
        condition_cols = [col for col in df.columns if col.startswith('has_')]
        if condition_cols:
            condition_prev = df[condition_cols].mean().sort_values(ascending=True)
            condition_prev.plot(kind='barh', ax=ax, color=self.colors['primary'])
            ax.set_title('Medical Conditions Prevalence')
            ax.set_xlabel('Prevalence')
            # Clean up condition names
            ax.set_yticklabels([label.get_text().replace('has_', '').replace('_', ' ').title() 
                               for label in ax.get_yticklabels()])
        
        # Previous admissions
        ax = axes[1, 1]
        prev_adm_cols = [col for col in df.columns if 'previous' in col.lower() and 'admission' in col.lower()]
        if prev_adm_cols:
            prev_adm_col = prev_adm_cols[0]
            prev_adm_counts = df[prev_adm_col].value_counts().head(10).sort_index()
            prev_adm_counts.plot(kind='bar', ax=ax, color=self.colors['secondary'])
            ax.set_title('Previous Admissions Distribution')
            ax.set_xlabel('Number of Previous Admissions')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=0)
        
        # Emergency vs Scheduled admissions
        ax = axes[1, 2]
        emergency_cols = [col for col in df.columns if 'emergency' in col.lower()]
        if emergency_cols:
            emergency_col = emergency_cols[0]
            emergency_counts = df[emergency_col].value_counts()
            labels = ['Scheduled', 'Emergency'] if len(emergency_counts) == 2 else emergency_counts.index
            ax.pie(emergency_counts.values, labels=labels, autopct='%1.1f%%', 
                  colors=[self.colors['primary'], self.colors['accent']])
            ax.set_title('Admission Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance(self, y_true, y_pred, y_proba, model_name="Model", save_path=None):
        """
        Create comprehensive model performance visualization.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels  
            y_proba (np.array): Predicted probabilities
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{model_name} Performance Analysis', fontsize=16, fontweight='bold')
        
        # ROC Curve
        ax = axes[0, 0]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=self.colors['primary'], linewidth=2, 
               label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors['neutral'], linestyle='--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax = axes[0, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        baseline = y_true.mean()
        ax.plot(recall, precision, color=self.colors['secondary'], linewidth=2, label='PR Curve')
        ax.axhline(y=baseline, color=self.colors['neutral'], linestyle='--', alpha=0.5, 
                  label=f'Baseline ({baseline:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax = axes[0, 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Readmission', 'Readmission'],
                   yticklabels=['No Readmission', 'Readmission'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Prediction Distribution
        ax = axes[1, 0]
        ax.hist(y_proba[y_true == 0], bins=30, alpha=0.7, label='No Readmission', 
               color=self.colors['primary'], density=True)
        ax.hist(y_proba[y_true == 1], bins=30, alpha=0.7, label='Readmission', 
               color=self.colors['accent'], density=True)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calibration Plot
        ax = axes[1, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', color=self.colors['primary'], 
               linewidth=2, markersize=6, label='Model')
        ax.plot([0, 1], [0, 1], '--', color=self.colors['neutral'], alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance Metrics
        ax = axes[1, 2]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_true, y_proba)
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=self.colors['primary'], alpha=0.7)
        ax.set_title('Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, top_n=15, save_path=None):
        """
        Visualize feature importance with clinical context.
        
        Args:
            feature_importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        # Get top features
        top_features = feature_importance_df.head(top_n).copy()
        
        # Clean feature names for better readability
        top_features['feature_clean'] = top_features['feature'].apply(self._clean_feature_name)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors['primary'], alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features['feature_clean'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for Readmission Prediction')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', ha='left')
        
        # Add feature categories with colors
        clinical_categories = {
            'Medical Condition': ['has_', 'diabetes', 'hypertension', 'heart', 'kidney'],
            'Demographics': ['age', 'gender'],
            'Utilization': ['length', 'stay', 'previous', 'admission'],
            'Risk Factors': ['emergency', 'high_risk', 'frequent']
        }
        
        # Color bars by category
        for i, feature in enumerate(top_features['feature']):
            category_color = self.colors['primary']  # default
            for category, keywords in clinical_categories.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    if category == 'Medical Condition':
                        category_color = self.colors['accent']
                    elif category == 'Demographics':
                        category_color = self.colors['secondary']
                    elif category == 'Risk Factors':
                        category_color = self.colors['success']
                    break
            bars[i].set_color(category_color)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _clean_feature_name(self, feature_name):
        """Clean feature names for better readability."""
        # Remove prefixes
        cleaned = feature_name.replace('has_', '').replace('_std_', ' ')
        
        # Replace underscores with spaces
        cleaned = cleaned.replace('_', ' ')
        
        # Capitalize words
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())
        
        # Handle special cases
        replacements = {
            'Copd': 'COPD',
            'Ppv': 'PPV',
            'Npv': 'NPV',
            'Auc': 'AUC',
            'Los': 'Length of Stay'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def plot_clinical_scenarios(self, scenario_results_df, save_path=None):
        """
        Visualize model performance across clinical scenarios.
        
        Args:
            scenario_results_df (pd.DataFrame): Results from clinical scenario analysis
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Clinical Scenario Analysis', fontsize=16, fontweight='bold')
        
        # Scenario sample sizes
        ax = axes[0, 0]
        scenario_results_df.plot(x='Scenario', y='N_Patients', kind='bar', ax=ax, 
                               color=self.colors['primary'], alpha=0.7)
        ax.set_title('Patient Count by Scenario')
        ax.set_ylabel('Number of Patients')
        ax.tick_params(axis='x', rotation=45)
        
        # Actual vs Predicted rates
        ax = axes[0, 1]
        x = np.arange(len(scenario_results_df))
        width = 0.35
        
        ax.bar(x - width/2, scenario_results_df['Actual_Rate'] * 100, width, 
              label='Actual Rate', color=self.colors['primary'], alpha=0.7)
        ax.bar(x + width/2, scenario_results_df['Predicted_Rate'] * 100, width,
              label='Predicted Rate', color=self.colors['accent'], alpha=0.7)
        
        ax.set_xlabel('Clinical Scenarios')
        ax.set_ylabel('Readmission Rate (%)')
        ax.set_title('Actual vs Predicted Readmission Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_results_df['Scenario'], rotation=45, ha='right')
        ax.legend()
        
        # Model performance (AUC) by scenario
        ax = axes[1, 0]
        valid_auc = scenario_results_df.dropna(subset=['AUC'])
        if len(valid_auc) > 0:
            bars = ax.bar(range(len(valid_auc)), valid_auc['AUC'], 
                         color=self.colors['secondary'], alpha=0.7)
            ax.set_title('Model Performance (AUC) by Scenario')
            ax.set_ylabel('AUC Score')
            ax.set_xticks(range(len(valid_auc)))
            ax.set_xticklabels(valid_auc['Scenario'], rotation=45, ha='right')
            ax.axhline(y=0.7, color=self.colors['success'], linestyle='--', alpha=0.7, label='Good Performance')
            ax.axhline(y=0.5, color=self.colors['neutral'], linestyle='--', alpha=0.7, label='Random')
            ax.legend()
            
            # Add value labels
            for bar, value in zip(bars, valid_auc['AUC']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
        
        # Risk distribution across scenarios
        ax = axes[1, 1]
        ax.scatter(scenario_results_df['Actual_Rate'] * 100, 
                  scenario_results_df['Predicted_Rate'] * 100,
                  c=scenario_results_df['N_Patients'], s=100, 
                  cmap='viridis', alpha=0.7)
        
        # Perfect prediction line
        max_rate = max(scenario_results_df['Actual_Rate'].max(), 
                      scenario_results_df['Predicted_Rate'].max()) * 100
        ax.plot([0, max_rate], [0, max_rate], '--', color=self.colors['neutral'], alpha=0.5)
        
        ax.set_xlabel('Actual Readmission Rate (%)')
        ax.set_ylabel('Predicted Readmission Rate (%)')
        ax.set_title('Prediction Accuracy by Scenario')
        
        # Add colorbar
        scatter = ax.collections[0]
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Patients')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_risk_dashboard(self, patient_data, model, save_path=None):
        """
        Create an interactive dashboard for individual patient risk assessment.
        
        Args:
            patient_data (dict): Individual patient data
            model: Trained model for prediction
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Individual Patient Risk Assessment Dashboard', fontsize=16, fontweight='bold')
        
        # Risk gauge (simulated)
        ax = axes[0, 0]
        risk_score = patient_data.get('risk_probability', 0.5)
        
        # Create gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color zones
        ax.fill_between(theta[0:33], 0, r[0:33], color='green', alpha=0.3, label='Low Risk')
        ax.fill_between(theta[33:66], 0, r[33:66], color='yellow', alpha=0.3, label='Moderate Risk')
        ax.fill_between(theta[66:100], 0, r[66:100], color='red', alpha=0.3, label='High Risk')
        
        # Risk needle
        risk_angle = np.pi * (1 - risk_score)
        ax.arrow(0, 0, 0.8 * np.cos(risk_angle), 0.8 * np.sin(risk_angle), 
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'30-Day Readmission Risk\n{risk_score*100:.1f}%')
        ax.axis('off')
        
        # Patient demographics
        ax = axes[0, 1]
        demo_text = f"""
        Patient Demographics:
        
        Age: {patient_data.get('age', 'N/A')} years
        Gender: {patient_data.get('gender', 'N/A')}
        Insurance: {patient_data.get('insurance', 'N/A')}
        
        Current Admission:
        Length of Stay: {patient_data.get('length_of_stay', 'N/A')} days
        Emergency Admission: {'Yes' if patient_data.get('emergency_admission', 0) else 'No'}
        """
        
        ax.text(0.1, 0.9, demo_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Patient Information')
        
        # Medical conditions
        ax = axes[0, 2]
        conditions = {
            'Diabetes': patient_data.get('has_diabetes', 0),
            'Hypertension': patient_data.get('has_hypertension', 0),
            'Heart Disease': patient_data.get('has_heart_disease', 0),
            'Kidney Disease': patient_data.get('has_kidney_disease', 0),
            'COPD': patient_data.get('has_copd', 0)
        }
        
        present_conditions = [condition for condition, present in conditions.items() if present]
        
        if present_conditions:
            y_pos = np.arange(len(present_conditions))
            ax.barh(y_pos, [1] * len(present_conditions), color=self.colors['accent'], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(present_conditions)
            ax.set_xlabel('Present')
            ax.set_title('Medical Conditions')
        else:
            ax.text(0.5, 0.5, 'No Major\nConditions Recorded', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Medical Conditions')
            ax.axis('off')
        
        # Risk factors breakdown
        ax = axes[1, 0]
        risk_factors = {
            'Age > 75': 1 if patient_data.get('age', 0) > 75 else 0,
            'Multiple Conditions': 1 if sum(conditions.values()) >= 3 else 0,
            'Previous Admissions': 1 if patient_data.get('previous_admissions', 0) >= 2 else 0,
            'Long Stay': 1 if patient_data.get('length_of_stay', 0) > 7 else 0,
            'Emergency Admission': patient_data.get('emergency_admission', 0)
        }
        
        present_risks = [risk for risk, present in risk_factors.items() if present]
        risk_count = len(present_risks)
        
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        color = colors[min(risk_count, 4)]
        
        ax.pie([risk_count, 5-risk_count], labels=[f'Present ({risk_count})', f'Absent ({5-risk_count})'],
              colors=[color, 'lightgray'], autopct='%1.0f%%')
        ax.set_title('Risk Factors Present')
        
        # Admission history
        ax = axes[1, 1]
        prev_admissions = patient_data.get('previous_admissions', 0)
        days_since_last = patient_data.get('days_since_last_admission', 999)
        
        history_text = f"""
        Admission History:
        
        Previous Admissions: {prev_admissions}
        Days Since Last: {days_since_last if days_since_last < 999 else 'First admission'}
        
        Risk Category:
        """
        
        if prev_admissions == 0:
            risk_cat = "First-time patient"
        elif days_since_last <= 30:
            risk_cat = "Recent return"
        elif prev_admissions >= 3:
            risk_cat = "Frequent utilizer"
        else:
            risk_cat = "Standard risk"
        
        history_text += risk_cat
        
        ax.text(0.1, 0.9, history_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Admission History')
        
        # Recommendations
        ax = axes[1, 2]
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.append("• High risk - Consider intensive discharge planning")
            recommendations.append("• Schedule early follow-up appointment")
        elif risk_score > 0.3:
            recommendations.append("• Moderate risk - Standard discharge protocols")
            recommendations.append("• Ensure medication reconciliation")
        else:
            recommendations.append("• Low risk - Standard care")
        
        if patient_data.get('has_diabetes', 0):
            recommendations.append("• Diabetes education and monitoring")
        
        if patient_data.get('has_heart_disease', 0):
            recommendations.append("• Cardiology follow-up")
        
        if prev_admissions >= 2:
            recommendations.append("• Case management involvement")
        
        rec_text = "Clinical Recommendations:\n\n" + "\n".join(recommendations)
        
        ax.text(0.1, 0.9, rec_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Clinical Recommendations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Interactive Plotly visualizations (if available)
class InteractiveVisualizer:
    """
    Interactive visualization class using Plotly (if available).
    """
    
    def __init__(self):
        """Initialize interactive visualizer."""
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Install with 'pip install plotly' for interactive visualizations.")
        self.available = PLOTLY_AVAILABLE
    
    def interactive_performance_dashboard(self, y_true, y_pred, y_proba, model_name="Model"):
        """
        Create interactive performance dashboard.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_proba (np.array): Predicted probabilities
            model_name (str): Name of the model
        """
        if not self.available:
            print("Plotly not available for interactive visualizations")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Prediction Distribution', 'Calibration Plot'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = np.trapz(tpr, fpr)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # Prediction Distribution
        fig.add_trace(
            go.Histogram(x=y_proba[y_true == 0], name='No Readmission', 
                        opacity=0.7, nbinsx=30),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=y_proba[y_true == 1], name='Readmission', 
                        opacity=0.7, nbinsx=30),
            row=2, col=1
        )
        
        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
        fig.add_trace(
            go.Scatter(x=mean_predicted_value, y=fraction_of_positives, 
                      mode='markers+lines', name='Model Calibration',
                      marker=dict(size=8)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                      line=dict(color='gray', dash='dash')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'{model_name} Performance Dashboard',
            height=600,
            showlegend=True
        )
        
        fig.show()
    
    def interactive_risk_calculator(self, model, feature_names):
        """
        Create interactive risk calculator interface.
        (This would be implemented with Dash or similar framework)
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
        """
        if not self.available:
            print("Plotly not available for interactive risk calculator")
            return
        
        print("Interactive risk calculator would be implemented here using Dash")
        print("This would allow real-time risk calculation based on patient inputs")


if __name__ == "__main__":
    # Example usage
    print("Visualization Module - Example Usage")
    print("=" * 50)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Sample dataset
    sample_data = pd.DataFrame({
        'age': np.random.normal(65, 15, n_samples),
        'length_of_stay': np.random.exponential(5, n_samples),
        'previous_admissions': np.random.poisson(1.5, n_samples),
        'emergency_admission': np.random.binomial(1, 0.6, n_samples),
        'has_diabetes': np.random.binomial(1, 0.3, n_samples),
        'has_hypertension': np.random.binomial(1, 0.4, n_samples),
        'has_heart_disease': np.random.binomial(1, 0.25, n_samples),
        'readmission_30_day': np.random.binomial(1, 0.15, n_samples)
    })
    
    # Sample predictions
    y_true = sample_data['readmission_30_day'].values
    y_pred = np.random.binomial(1, 0.15, n_samples)
    y_proba = np.random.beta(1, 5, n_samples)  # Realistic probability distribution
    
    print(f"Sample data created: {sample_data.shape[0]} patients")
    
    # Initialize visualizer
    visualizer = ReadmissionVisualizer(style='healthcare')
    
    # Test data quality overview
    print("\n1. Creating data quality overview...")
    visualizer.plot_data_quality_overview(sample_data)
    
    # Test clinical overview
    print("\n2. Creating clinical overview...")
    visualizer.plot_clinical_overview(sample_data)
    
    # Test model performance
    print("\n3. Creating model performance visualization...")
    visualizer.plot_model_performance(y_true, y_pred, y_proba, "Sample Model")
    
    # Test feature importance
    print("\n4. Creating feature importance plot...")
    feature_importance = pd.DataFrame({
        'feature': ['has_heart_disease', 'age', 'previous_admissions', 'has_diabetes', 
                   'length_of_stay', 'emergency_admission', 'has_hypertension'],
        'importance': [0.25, 0.20, 0.18, 0.15, 0.12, 0.06, 0.04]
    })
    visualizer.plot_feature_importance(feature_importance)
    
    # Test clinical scenarios
    print("\n5. Creating clinical scenarios visualization...")
    scenario_data = pd.DataFrame({
        'Scenario': ['Elderly Diabetic', 'Young Patients', 'Multiple Comorbidities', 'Emergency Admissions'],
        'N_Patients': [150, 200, 120, 300],
        'Actual_Rate': [0.25, 0.08, 0.35, 0.18],
        'Predicted_Rate': [0.23, 0.10, 0.32, 0.20],
        'AUC': [0.78, 0.72, 0.81, 0.75]
    })
    visualizer.plot_clinical_scenarios(scenario_data)
    
    # Test patient dashboard
    print("\n6. Creating patient risk dashboard...")
    sample_patient = {
        'age': 78,
        'gender': 'Female',
        'insurance': 'Medicare',
        'length_of_stay': 5,
        'emergency_admission': 1,
        'has_diabetes': 1,
        'has_hypertension': 1,
        'has_heart_disease': 0,
        'has_kidney_disease': 0,
        'has_copd': 0,
        'previous_admissions': 2,
        'days_since_last_admission': 45,
        'risk_probability': 0.65
    }
    visualizer.create_risk_dashboard(sample_patient, None)
    
    # Test interactive visualizer (if Plotly available)
    print("\n7. Testing interactive visualizations...")
    interactive_viz = InteractiveVisualizer()
    if interactive_viz.available:
        interactive_viz.interactive_performance_dashboard(y_true, y_pred, y_proba, "Interactive Model")
    
    print("\nVisualization module testing complete!")
    print("\nAll visualization functions are ready for integration with the main project.")
    print("Key features:")
    print("• Healthcare-themed styling")
    print("• Comprehensive model performance analysis")
    print("• Clinical scenario visualization")
    print("• Individual patient risk dashboards")
    print("• Interactive components (with Plotly)")
    print("• Publication-ready plots with customizable styling")


# Utility functions for dashboard integration
def create_model_summary_report(model_results, save_path=None):
    """
    Create a comprehensive model summary report with multiple visualizations.
    
    Args:
        model_results (dict): Dictionary containing model evaluation results
        save_path (str): Path to save the report
    """
    visualizer = ReadmissionVisualizer()
    
    # Create figure with multiple subplots for summary report
    fig = plt.figure(figsize=(16, 12))
    
    # Add main title
    fig.suptitle('Hospital Readmission Risk Model - Summary Report', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Model performance summary (top section)
    gs1 = fig.add_gridspec(2, 3, top=0.85, bottom=0.55, hspace=0.3, wspace=0.3)
    
    # ROC Curve
    ax1 = fig.add_subplot(gs1[0, 0])
    fpr, tpr, _ = roc_curve(model_results['y_true'], model_results['y_proba'])
    roc_auc = np.trapz(tpr, fpr)
    ax1.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], '--', alpha=0.5)
    ax1.set_title('ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature Importance
    ax2 = fig.add_subplot(gs1[0, 1])
    top_features = model_results['feature_importance'].head(8)
    ax2.barh(range(len(top_features)), top_features['importance'])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels([visualizer._clean_feature_name(f) for f in top_features['feature']])
    ax2.set_title('Top Features')
    ax2.set_xlabel('Importance')
    
    # Performance Metrics
    ax3 = fig.add_subplot(gs1[0, 2])
    metrics = model_results['metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_values = [metrics[m.lower().replace('-', '_')] for m in metric_names]
    bars = ax3.bar(metric_names, metric_values)
    ax3.set_title('Performance Metrics')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    for bar, value in zip(bars, metric_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Clinical Analysis (middle section)
    gs2 = fig.add_gridspec(1, 2, top=0.50, bottom=0.30, hspace=0.2, wspace=0.3)
    
    # Confusion Matrix
    ax4 = fig.add_subplot(gs2[0, 0])
    cm = confusion_matrix(model_results['y_true'], model_results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
               xticklabels=['No Readmission', 'Readmission'],
               yticklabels=['No Readmission', 'Readmission'])
    ax4.set_title('Confusion Matrix')
    
    # Clinical Metrics Table
    ax5 = fig.add_subplot(gs2[0, 1])
    ax5.axis('tight')
    ax5.axis('off')
    
    clinical_data = []
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    clinical_metrics = [
        ['Metric', 'Value', 'Clinical Interpretation'],
        ['Sensitivity', f'{sensitivity:.3f}', f'Catches {sensitivity*100:.1f}% of readmissions'],
        ['Specificity', f'{specificity:.3f}', f'Correctly IDs {specificity*100:.1f}% of non-readmissions'],
        ['PPV', f'{ppv:.3f}', f'{ppv*100:.1f}% of high-risk predictions correct'],
        ['NPV', f'{npv:.3f}', f'{npv*100:.1f}% of low-risk predictions correct'],
        ['NNS', f'{1/metrics["precision"]:.1f}', 'Patients to screen per readmission found']
    ]
    
    table = ax5.table(cellText=clinical_metrics[1:], colLabels=clinical_metrics[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax5.set_title('Clinical Performance Metrics', pad=20)
    
    # Summary Text (bottom section)
    gs3 = fig.add_gridspec(1, 1, top=0.25, bottom=0.05)
    ax6 = fig.add_subplot(gs3[0, 0])
    ax6.axis('off')
    
    summary_text = f"""
    MODEL SUMMARY & CLINICAL RECOMMENDATIONS
    
    Performance: The model achieves an AUC of {roc_auc:.3f}, indicating {['poor', 'fair', 'good', 'excellent'][min(3, int(roc_auc*4))]} discriminatory performance.
    
    Clinical Utility: With {sensitivity*100:.1f}% sensitivity and {ppv*100:.1f}% precision, the model balances detection and false alarms effectively.
    
    Implementation: Suitable for clinical decision support with appropriate human oversight. Recommended threshold: {np.percentile(model_results['y_proba'], 85):.2f}
    
    Next Steps: Pilot implementation with clinical staff feedback and continuous monitoring of performance metrics.
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def export_dashboard_data(data, model_results, export_path):
    """
    Export processed data and results for dashboard deployment.
    
    Args:
        data (pd.DataFrame): Processed dataset
        model_results (dict): Model evaluation results
        export_path (str): Path to export the data
    """
    import json
    
    # Prepare data for JSON export
    dashboard_data = {
        'dataset_summary': {
            'total_patients': len(data),
            'features': list(data.columns),
            'readmission_rate': data['readmission_30_day'].mean() if 'readmission_30_day' in data.columns else None
        },
        'model_performance': {
            'auc': model_results.get('auc', None),
            'accuracy': model_results.get('accuracy', None),
            'precision': model_results.get('precision', None),
            'recall': model_results.get('recall', None)
        },
        'feature_importance': model_results.get('feature_importance', {}).to_dict('records') if 'feature_importance' in model_results else [],
        'clinical_scenarios': model_results.get('clinical_scenarios', []),
        'export_timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON file
    with open(export_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print(f"Dashboard data exported to {export_path}")


# Custom color palettes for different contexts
HEALTHCARE_COLORS = {
    'primary': '#2E86AB',      # Medical blue
    'secondary': '#A23B72',    # Deep pink  
    'accent': '#F18F01',       # Orange
    'danger': '#C73E1D',       # Red
    'success': '#28A745',      # Green
    'warning': '#FFC107',      # Yellow
    'neutral': '#6B7280',      # Gray
    'background': '#F9FAFB'    # Light gray
}

CLINICAL_COLORS = {
    'low_risk': '#28A745',     # Green
    'moderate_risk': '#FFC107', # Yellow
    'high_risk': '#DC3545',    # Red
    'critical_risk': '#6F1E1E' # Dark red
}

def get_risk_color(risk_score, palette='clinical'):
    """
    Get appropriate color based on risk score.
    
    Args:
        risk_score (float): Risk score between 0 and 1
        palette (str): Color palette to use
        
    Returns:
        str: Color code
    """
    if palette == 'clinical':
        if risk_score < 0.2:
            return CLINICAL_COLORS['low_risk']
        elif risk_score < 0.5:
            return CLINICAL_COLORS['moderate_risk']
        elif risk_score < 0.8:
            return CLINICAL_COLORS['high_risk']
        else:
            return CLINICAL_COLORS['critical_risk']
    else:
        return HEALTHCARE_COLORS['primary']