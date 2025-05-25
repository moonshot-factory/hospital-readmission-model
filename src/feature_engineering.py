"""
Hospital Readmission Risk Model - Feature Engineering Module

This module contains classes and functions for creating clinical features
from processed EHR data, including medical code translation and temporal
feature engineering.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class MedicalCodeTranslator:
    """
    Unified medical code translator for handling ICD-9, ICD-10, and local hospital codes.
    Critical for standardizing data across multiple hospital systems.
    """
    
    def __init__(self):
        """Initialize with comprehensive medical code mappings."""
        
        # Core condition mappings developed with clinical advisor input
        self.condition_mappings = {
            'diabetes': {
                'icd9': ['250.00', '250.01', '250.02', '250.03', '250.1', '250.2', '250.3'],
                'icd10': ['E11.9', 'E11.0', 'E11.1', 'E10.9', 'E11.65', 'E11.8'],
                'local': ['DM2', 'DM1', 'DIABETES', 'DIAB', 'T2DM', 'T1DM'],
                'description': 'Diabetes Mellitus'
            },
            'hypertension': {
                'icd9': ['401.9', '401.0', '401.1', '402.90', '403.90'],
                'icd10': ['I10', 'I11.9', 'I12.9', 'I13.10', 'I15.9'],
                'local': ['HTN', 'HYPERTENSION', 'HIGH_BP', 'HBP'],
                'description': 'Hypertension'
            },
            'heart_disease': {
                'icd9': ['428.0', '414.01', '411.1', '410.9', '429.2'],
                'icd10': ['I50.9', 'I25.10', 'I21.9', 'I25.9', 'I42.9'],
                'local': ['HF', 'CHF', 'HEART_FAILURE', 'CAD', 'MI'],
                'description': 'Heart Disease'
            },
            'kidney_disease': {
                'icd9': ['585.9', '585.3', '585.4', '585.5', '585.6'],
                'icd10': ['N18.9', 'N18.3', 'N18.4', 'N18.5', 'N18.6'],
                'local': ['KIDNEY', 'CKD', 'RENAL', 'ESRD'],
                'description': 'Chronic Kidney Disease'
            },
            'hyperlipidemia': {
                'icd9': ['272.4', '272.0', '272.1', '272.2'],
                'icd10': ['E78.5', 'E78.0', 'E78.1', 'E78.2'],
                'local': ['CHOL', 'LIPID', 'CHOLESTEROL', 'HYPERLIPID'],
                'description': 'Hyperlipidemia'
            },
            'copd': {
                'icd9': ['496', '491.21', '492.8', '493.20'],
                'icd10': ['J44.1', 'J44.0', 'J43.9', 'J45.9'],
                'local': ['COPD', 'EMPHYSEMA', 'BRONCHITIS'],
                'description': 'Chronic Obstructive Pulmonary Disease'
            },
            'depression': {
                'icd9': ['296.2', '296.3', '311', '300.4'],
                'icd10': ['F32.9', 'F33.9', 'F32.0', 'F33.0'],
                'local': ['DEPRESSION', 'DEPRESSIVE', 'MDD'],
                'description': 'Depression'
            }
        }
        
        # Create reverse lookup for faster processing
        self.code_to_condition = {}
        for condition, code_types in self.condition_mappings.items():
            for code_type in ['icd9', 'icd10', 'local']:
                if code_type in code_types:
                    for code in code_types[code_type]:
                        self.code_to_condition[code.upper()] = condition
    
    def translate_codes(self, codes_string):
        """
        Translate a string of mixed medical codes to standardized conditions.
        
        Args:
            codes_string (str): String containing medical codes separated by delimiters
            
        Returns:
            dict: Dictionary of condition flags
        """
        if pd.isna(codes_string) or codes_string == '' or codes_string is None:
            return {}
        
        # Parse codes from string (handle multiple delimiters)
        codes = re.split(r'[;,|]', str(codes_string))
        codes = [code.strip().upper() for code in codes if code.strip()]
        
        # Map to conditions
        conditions = {}
        for code in codes:
            # Direct match
            condition = self.code_to_condition.get(code)
            if condition:
                conditions[condition] = 1
            else:
                # Partial match for codes with decimal variations
                for mapped_code, mapped_condition in self.code_to_condition.items():
                    if code.startswith(mapped_code.split('.')[0]) or mapped_code.startswith(code.split('.')[0]):
                        conditions[mapped_condition] = 1
                        break
        
        return conditions
    
    def create_condition_features(self, df, diagnosis_column='diagnosis_codes'):
        """
        Create binary features for each condition from diagnosis codes.
        
        Args:
            df (pd.DataFrame): DataFrame containing diagnosis codes
            diagnosis_column (str): Name of column containing diagnosis codes
            
        Returns:
            pd.DataFrame: DataFrame with condition features added
        """
        df_with_conditions = df.copy()
        
        # Initialize condition columns
        condition_names = list(self.condition_mappings.keys())
        for condition in condition_names:
            df_with_conditions[f'has_{condition}'] = 0
        
        # Process each patient's codes
        for idx, codes_string in enumerate(df[diagnosis_column]):
            conditions = self.translate_codes(codes_string)
            for condition in conditions:
                df_with_conditions.loc[df_with_conditions.index[idx], f'has_{condition}'] = 1
        
        return df_with_conditions
    
    def get_condition_summary(self, df, diagnosis_column='diagnosis_codes'):
        """
        Get summary statistics of condition prevalence.
        
        Args:
            df (pd.DataFrame): DataFrame with diagnosis codes
            diagnosis_column (str): Column name containing diagnosis codes
            
        Returns:
            dict: Summary of condition prevalence
        """
        df_with_conditions = self.create_condition_features(df, diagnosis_column)
        
        summary = {}
        for condition in self.condition_mappings.keys():
            col_name = f'has_{condition}'
            if col_name in df_with_conditions.columns:
                prevalence = df_with_conditions[col_name].mean()
                count = df_with_conditions[col_name].sum()
                summary[condition] = {
                    'prevalence': prevalence,
                    'count': count,
                    'description': self.condition_mappings[condition]['description']
                }
        
        return summary


class TemporalFeatureEngineer:
    """
    Create time-series features from admission patterns and temporal data.
    """
    
    def __init__(self):
        """Initialize temporal feature engineer."""
        self.reference_date = datetime(2015, 1, 1)  # Project reference date
    
    def calculate_days_since_last_admission(self, df, patient_id_col='patient_id', 
                                          admission_date_col='admission_date'):
        """
        Calculate days since last admission for patients with admission history.
        
        Args:
            df (pd.DataFrame): DataFrame with patient admissions
            patient_id_col (str): Column name for patient ID
            admission_date_col (str): Column name for admission date
            
        Returns:
            pd.Series: Days since last admission (999 for first-time patients)
        """
        # Sort by patient and date
        df_sorted = df.sort_values([patient_id_col, admission_date_col])
        
        # Calculate time differences within each patient group
        days_since_last = []
        
        for patient_id in df[patient_id_col].unique():
            patient_data = df_sorted[df_sorted[patient_id_col] == patient_id]
            
            if len(patient_data) == 1:
                # First admission for this patient
                days_since_last.append(999)
            else:
                # Calculate days since previous admission
                dates = patient_data[admission_date_col].values
                for i, date in enumerate(dates):
                    if i == 0:
                        days_since_last.append(999)  # First admission
                    else:
                        try:
                            current_date = pd.to_datetime(date)
                            previous_date = pd.to_datetime(dates[i-1])
                            days_diff = (current_date - previous_date).days
                            days_since_last.append(days_diff)
                        except:
                            days_since_last.append(999)  # Default for parsing errors
        
        return pd.Series(days_since_last, index=df_sorted.index)
    
    def create_admission_pattern_features(self, df, days_since_col='days_since_last_admission',
                                        previous_admissions_col='previous_admissions'):
        """
        Create categorical features based on admission patterns.
        
        Args:
            df (pd.DataFrame): DataFrame with temporal data
            days_since_col (str): Column with days since last admission
            previous_admissions_col (str): Column with previous admission count
            
        Returns:
            pd.DataFrame: DataFrame with admission pattern features
        """
        df_patterns = df.copy()
        
        # Recent admission flag (within 30 days)
        df_patterns['recent_admission'] = (df_patterns[days_since_col] <= 30).astype(int)
        
        # Frequent readmitter flag (2+ previous admissions within 90 days)
        df_patterns['frequent_readmitter'] = (
            (df_patterns[previous_admissions_col] >= 2) & 
            (df_patterns[days_since_col] <= 90)
        ).astype(int)
        
        # Admission pattern categories
        def categorize_admission_pattern(row):
            if row[previous_admissions_col] == 0:
                return 'First_Time'
            elif row[days_since_col] <= 30:
                return 'Recent_Return'
            elif row[days_since_col] <= 90:
                return 'Short_Interval'
            elif row[days_since_col] <= 365:
                return 'Medium_Interval'
            else:
                return 'Long_Interval'
        
        df_patterns['admission_pattern'] = df_patterns.apply(categorize_admission_pattern, axis=1)
        
        # Seasonal admission features (if admission date available)
        if 'admission_date' in df_patterns.columns:
            df_patterns['admission_month'] = pd.to_datetime(df_patterns['admission_date']).dt.month
            df_patterns['admission_season'] = df_patterns['admission_month'].apply(self._get_season)
            df_patterns['admission_day_of_week'] = pd.to_datetime(df_patterns['admission_date']).dt.dayofweek
            df_patterns['weekend_admission'] = (df_patterns['admission_day_of_week'] >= 5).astype(int)
        
        return df_patterns
    
    def _get_season(self, month):
        """Map month to season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def create_temporal_aggregates(self, df, patient_id_col='patient_id', 
                                 timeframe_days=365):
        """
        Create aggregate features over specified timeframes.
        
        Args:
            df (pd.DataFrame): DataFrame with patient history
            patient_id_col (str): Patient ID column
            timeframe_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: DataFrame with temporal aggregate features
        """
        df_agg = df.copy()
        
        # Group by patient and create aggregates
        patient_groups = df.groupby(patient_id_col)
        
        # Count admissions in timeframe
        df_agg[f'admissions_last_{timeframe_days}d'] = patient_groups[patient_id_col].transform('count')
        
        # Average length of stay in timeframe
        if 'length_of_stay' in df.columns:
            df_agg[f'avg_los_last_{timeframe_days}d'] = patient_groups['length_of_stay'].transform('mean')
        
        # Emergency admission rate
        if 'emergency_admission' in df.columns:
            df_agg[f'emergency_rate_last_{timeframe_days}d'] = patient_groups['emergency_admission'].transform('mean')
        
        return df_agg


class ClinicalFeatureEngineer:
    """
    Create clinical features based on medical domain knowledge.
    """
    
    def __init__(self):
        """Initialize clinical feature engineer."""
        
        # Clinical risk factor weights (based on medical literature)
        self.risk_weights = {
            'age_75_plus': 1.5,
            'diabetes': 1.3,
            'heart_disease': 2.0,
            'kidney_disease': 1.8,
            'multiple_comorbidities': 1.4,
            'frequent_readmitter': 2.5,
            'emergency_admission': 1.2
        }
    
    def create_age_features(self, df, age_col='age'):
        """
        Create age-related features.
        
        Args:
            df (pd.DataFrame): DataFrame with age data
            age_col (str): Age column name
            
        Returns:
            pd.DataFrame: DataFrame with age features
        """
        df_age = df.copy()
        
        # Age groups
        df_age['age_group'] = pd.cut(df_age[age_col], 
                                   bins=[0, 40, 60, 80, 100], 
                                   labels=['Under_40', '40_60', '60_80', 'Over_80'])
        
        # Elderly flag
        df_age['elderly'] = (df_age[age_col] >= 75).astype(int)
        
        # Age squared for non-linear relationships
        df_age['age_squared'] = df_age[age_col] ** 2
        
        # Age interaction with emergency admission
        if 'emergency_admission' in df_age.columns:
            df_age['emergency_elderly'] = (
                df_age['emergency_admission'] & (df_age[age_col] >= 65)
            ).astype(int)
        
        return df_age
    
    def create_comorbidity_features(self, df):
        """
        Create comorbidity-related features.
        
        Args:
            df (pd.DataFrame): DataFrame with condition flags
            
        Returns:
            pd.DataFrame: DataFrame with comorbidity features
        """
        df_comorbid = df.copy()
        
        # Find condition columns
        condition_cols = [col for col in df.columns if col.startswith('has_')]
        
        if condition_cols:
            # Total comorbidity count
            df_comorbid['comorbidity_count'] = df_comorbid[condition_cols].sum(axis=1)
            
            # Multiple comorbidities flag
            df_comorbid['multiple_comorbidities'] = (df_comorbid['comorbidity_count'] >= 3).astype(int)
            
            # High-risk comorbidity combinations
            high_risk_conditions = ['has_heart_disease', 'has_kidney_disease', 'has_diabetes']
            available_high_risk = [col for col in high_risk_conditions if col in df_comorbid.columns]
            
            if len(available_high_risk) >= 2:
                df_comorbid['high_risk_comorbidity'] = (df_comorbid[available_high_risk].sum(axis=1) >= 2).astype(int)
            
            # Cardiovascular risk cluster
            cv_conditions = ['has_hypertension', 'has_heart_disease', 'has_diabetes']
            available_cv = [col for col in cv_conditions if col in df_comorbid.columns]
            
            if available_cv:
                df_comorbid['cardiovascular_risk'] = (df_comorbid[available_cv].sum(axis=1) >= 2).astype(int)
        
        return df_comorbid
    
    def create_risk_stratification(self, df):
        """
        Create overall risk stratification based on multiple factors.
        
        Args:
            df (pd.DataFrame): DataFrame with clinical features
            
        Returns:
            pd.DataFrame: DataFrame with risk stratification
        """
        df_risk = df.copy()
        
        # Calculate weighted risk score
        risk_score = 0
        
        # Age risk
        if 'elderly' in df_risk.columns:
            risk_score += df_risk['elderly'] * self.risk_weights['age_75_plus']
        
        # Condition risks
        condition_risk_map = {
            'has_diabetes': 'diabetes',
            'has_heart_disease': 'heart_disease', 
            'has_kidney_disease': 'kidney_disease'
        }
        
        for col, risk_key in condition_risk_map.items():
            if col in df_risk.columns:
                risk_score += df_risk[col] * self.risk_weights[risk_key]
        
        # Multiple comorbidities
        if 'multiple_comorbidities' in df_risk.columns:
            risk_score += df_risk['multiple_comorbidities'] * self.risk_weights['multiple_comorbidities']
        
        # Frequent readmitter
        if 'frequent_readmitter' in df_risk.columns:
            risk_score += df_risk['frequent_readmitter'] * self.risk_weights['frequent_readmitter']
        
        # Emergency admission
        if 'emergency_admission' in df_risk.columns:
            risk_score += df_risk['emergency_admission'] * self.risk_weights['emergency_admission']
        
        df_risk['clinical_risk_score'] = risk_score
        
        # Risk categories
        df_risk['risk_category'] = pd.cut(risk_score, 
                                        bins=[-np.inf, 2, 4, 6, np.inf],
                                        labels=['Low', 'Moderate', 'High', 'Very_High'])
        
        # High-risk patient flag (simplified version)
        df_risk['high_risk_patient'] = (
            (df_risk.get('elderly', 0) == 1) | 
            (df_risk.get('previous_admissions', 0) >= 3) |
            (df_risk.get('comorbidity_count', 0) >= 3) |
            (df_risk.get('frequent_readmitter', 0) == 1)
        ).astype(int)
        
        return df_risk
    
    def create_utilization_features(self, df):
        """
        Create healthcare utilization features.
        
        Args:
            df (pd.DataFrame): DataFrame with utilization data
            
        Returns:
            pd.DataFrame: DataFrame with utilization features
        """
        df_util = df.copy()
        
        # Length of stay categories
        if 'length_of_stay' in df_util.columns:
            df_util['los_category'] = pd.cut(df_util['length_of_stay'],
                                           bins=[0, 2, 5, 10, float('inf')],
                                           labels=['Short', 'Medium', 'Long', 'Extended'])
            
            # Long stay flag
            df_util['long_stay'] = (df_util['length_of_stay'] > 10).astype(int)
        
        # Previous admissions categories
        if 'previous_admissions' in df_util.columns:
            df_util['admission_history'] = pd.cut(df_util['previous_admissions'],
                                                bins=[-1, 0, 2, 5, float('inf')],
                                                labels=['None', 'Low', 'Moderate', 'High'])
        
        return df_util


class ComprehensiveFeatureEngineer:
    """
    Main feature engineering pipeline that combines all feature types.
    """
    
    def __init__(self):
        """Initialize comprehensive feature engineer."""
        self.medical_translator = MedicalCodeTranslator()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.clinical_engineer = ClinicalFeatureEngineer()
        
    def engineer_all_features(self, df, config=None):
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Processed EHR data
            config (dict): Configuration for feature engineering steps
            
        Returns:
            tuple: (feature_df, feature_report)
        """
        if config is None:
            config = {
                'medical_codes': True,
                'temporal_features': True,
                'clinical_features': True,
                'age_features': True,
                'comorbidity_features': True,
                'risk_stratification': True,
                'utilization_features': True,
                'categorical_encoding': True
            }
        
        feature_report = {
            'original_features': list(df.columns),
            'steps_completed': [],
            'features_created': [],
            'final_feature_count': 0
        }
        
        df_features = df.copy()
        
        print("Starting comprehensive feature engineering...")
        
        # Step 1: Medical code translation
        if config.get('medical_codes', True) and 'diagnosis_codes' in df_features.columns:
            df_features = self.medical_translator.create_condition_features(df_features)
            condition_features = [col for col in df_features.columns if col.startswith('has_')]
            feature_report['features_created'].extend(condition_features)
            feature_report['steps_completed'].append('Medical code translation')
            print(f"Created {len(condition_features)} medical condition features")
        
        # Step 2: Temporal features
        if config.get('temporal_features', True):
            if 'previous_admissions' in df_features.columns:
                # Simulate days since last admission
                df_features['days_since_last_admission'] = np.where(
                    df_features['previous_admissions'] > 0,
                    np.random.exponential(60, len(df_features)),
                    999
                )
                feature_report['features_created'].append('days_since_last_admission')
            
            df_features = self.temporal_engineer.create_admission_pattern_features(df_features)
            temporal_features = ['recent_admission', 'frequent_readmitter', 'admission_pattern']
            feature_report['features_created'].extend(temporal_features)
            feature_report['steps_completed'].append('Temporal feature engineering')
            print(f"Created {len(temporal_features)} temporal features")
        
        # Step 3: Age features
        if config.get('age_features', True) and 'age' in df_features.columns:
            df_features = self.clinical_engineer.create_age_features(df_features)
            age_features = ['age_group', 'elderly', 'age_squared']
            feature_report['features_created'].extend(age_features)
            feature_report['steps_completed'].append('Age feature engineering')
            print(f"Created {len(age_features)} age-related features")
        
        # Step 4: Comorbidity features
        if config.get('comorbidity_features', True):
            df_features = self.clinical_engineer.create_comorbidity_features(df_features)
            comorbidity_features = ['comorbidity_count', 'multiple_comorbidities']
            feature_report['features_created'].extend(comorbidity_features)
            feature_report['steps_completed'].append('Comorbidity feature engineering')
            print(f"Created {len(comorbidity_features)} comorbidity features")
        
        # Step 5: Risk stratification
        if config.get('risk_stratification', True):
            df_features = self.clinical_engineer.create_risk_stratification(df_features)
            risk_features = ['clinical_risk_score', 'risk_category', 'high_risk_patient']
            feature_report['features_created'].extend(risk_features)
            feature_report['steps_completed'].append('Risk stratification')
            print(f"Created {len(risk_features)} risk stratification features")
        
        # Step 6: Utilization features
        if config.get('utilization_features', True):
            df_features = self.clinical_engineer.create_utilization_features(df_features)
            util_features = ['los_category', 'admission_history']
            feature_report['features_created'].extend(util_features)
            feature_report['steps_completed'].append('Utilization feature engineering')
            print(f"Created {len(util_features)} utilization features")
        
        # Step 7: Categorical encoding
        if config.get('categorical_encoding', True):
            df_features = self._encode_categorical_features(df_features)
            feature_report['steps_completed'].append('Categorical encoding')
            print("Applied categorical encoding")
        
        feature_report['final_features'] = list(df_features.columns)
        feature_report['final_feature_count'] = len(df_features.columns)
        
        print(f"Feature engineering complete: {len(df.columns)} -> {len(df_features.columns)} features")
        
        return df_features, feature_report
    
    def _encode_categorical_features(self, df):
        """
        Apply appropriate encoding to categorical features.
        
        Args:
            df (pd.DataFrame): DataFrame with categorical features
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        # Categorical columns to encode
        categorical_columns = ['gender', 'insurance_type', 'age_group', 'los_category', 
                             'admission_pattern', 'risk_category', 'admission_history']
        
        # Apply one-hot encoding
        for col in categorical_columns:
            if col in df_encoded.columns:
                # Get dummies and add to dataframe
                dummies = pd.get_dummies(df_encoded[col], prefix=f'{col}_std')
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                # Drop original column
                df_encoded.drop(col, axis=1, inplace=True)
        
        return df_encoded
    
    def select_final_features(self, df, target_col='readmission_30_day'):
        """
        Select final features for modeling.
        
        Args:
            df (pd.DataFrame): DataFrame with all engineered features
            target_col (str): Target variable column name
            
        Returns:
            tuple: (X, y, feature_list)
        """
        # Exclude non-predictive columns
        exclude_cols = [
            'patient_id', 'admission_date', 'diagnosis_codes', 
            'admission_date_parsed', target_col
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        return X, y, feature_cols
    
    def get_feature_summary(self, df):
        """
        Generate summary of engineered features.
        
        Args:
            df (pd.DataFrame): DataFrame with engineered features
            
        Returns:
            dict: Feature summary statistics
        """
        summary = {
            'total_features': len(df.columns),
            'feature_types': {},
            'missing_values': {},
            'data_types': dict(df.dtypes.astype(str))
        }
        
        # Categorize features by type
        feature_categories = {
            'Core Demographics': ['age', 'gender_std_*'],
            'Medical Conditions': ['has_*'],
            'Temporal Features': ['*admission*', 'days_since*', 'recent_*', 'frequent_*'],
            'Clinical Risk': ['*risk*', 'comorbidity_*', 'high_risk_*'],
            'Utilization': ['length_of_stay', 'previous_admissions', 'los_*'],
            'Categorical Encoded': ['*_std_*']
        }
        
        for category, patterns in feature_categories.items():
            matching_cols = []
            for pattern in patterns:
                if '*' in pattern:
                    # Handle wildcard patterns
                    prefix = pattern.replace('*', '')
                    matching_cols.extend([col for col in df.columns if prefix in col])
                else:
                    if pattern in df.columns:
                        matching_cols.append(pattern)
            summary['feature_types'][category] = len(set(matching_cols))
        
        # Missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                summary['missing_values'][col] = missing_count
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Example Usage")
    print("=" * 50)
    
    # Create sample processed data
    np.random.seed(42)
    n_patients = 1000
    
    sample_data = pd.DataFrame({
        'patient_id': [f"PID_{i:06d}" for i in range(n_patients)],
        'age': np.random.normal(65, 15, n_patients),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'insurance_type': np.random.choice(['Medicare', 'Private', 'Medicaid'], n_patients),
        'diagnosis_codes': [
            np.random.choice(['250.00;401.9', 'E11.9;I10', 'DM2;HTN', '428.0', 'I50.9'])
            for _ in range(n_patients)
        ],
        'length_of_stay': np.random.exponential(5, n_patients),
        'previous_admissions': np.random.poisson(1.5, n_patients),
        'emergency_admission': np.random.binomial(1, 0.6, n_patients),
        'readmission_30_day': np.random.binomial(1, 0.15, n_patients)
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Initialize feature engineer
    feature_engineer = ComprehensiveFeatureEngineer()
    
    # Run feature engineering
    print("\nRunning feature engineering pipeline...")
    featured_data, report = feature_engineer.engineer_all_features(sample_data)
    
    print(f"\nFeature engineering complete!")
    print(f"Original features: {len(report['original_features'])}")
    print(f"Final features: {report['final_feature_count']}")
    print(f"Steps completed: {len(report['steps_completed'])}")
    
    # Get final feature set
    X, y, feature_list = feature_engineer.select_final_features(featured_data)
    
    print(f"\nFinal modeling dataset:")
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape if y is not None else 'Not available'}")
    
    # Feature summary
    summary = feature_engineer.get_feature_summary(featured_data)
    print(f"\nFeature summary:")
    for category, count in summary['feature_types'].items():
        print(f"  {category}: {count} features")