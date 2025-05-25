"""
Hospital Readmission Risk Model - Data Processing Module

This module contains functions for processing raw EHR data, handling
inconsistent formats, and preparing data for feature engineering.

Author: Blake Sonnier
Project: Hospital Readmission Risk Prediction (2015)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')


class EHRDataProcessor:
    """
    Main class for processing Electronic Health Record data.
    Handles the major data quality issues typical in healthcare datasets.
    """
    
    def __init__(self):
        """Initialize the EHR data processor with standard configurations."""
        self.date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d', 
            '%m-%d-%Y', '%d/%m/%Y', '%Y/%m/%d'
        ]
        
        # Standard mappings for common inconsistencies
        self.gender_mapping = {
            'M': 'Male', 'MALE': 'Male', 'Male': 'Male', 'm': 'Male',
            'F': 'Female', 'FEMALE': 'Female', 'Female': 'Female', 'f': 'Female', 'female': 'Female',
            '': 'Unknown', 'Unknown': 'Unknown', 'UNK': 'Unknown', 
            'U': 'Unknown', np.nan: 'Unknown', None: 'Unknown'
        }
        
        self.insurance_keywords = {
            'Medicare': ['MEDICARE', 'medicare', 'Medicare', 'MCR'],
            'Medicaid': ['MEDICAID', 'medicaid', 'Medicaid', 'MCD'],
            'Private': ['PRIVATE', 'private', 'Private', 'Commercial', 'COMMERCIAL'],
            'Self-Pay': ['SELF', 'self', 'Self', 'CASH', 'cash', 'Cash', 'Uninsured']
        }
    
    def load_raw_data(self, filepath, file_type='csv'):
        """
        Load raw EHR data from various file formats.
        
        Args:
            filepath (str): Path to the data file
            file_type (str): Type of file ('csv', 'excel', 'json')
            
        Returns:
            pd.DataFrame: Raw data loaded into DataFrame
        """
        try:
            if file_type.lower() == 'csv':
                # Try different encodings common in healthcare data
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        data = pd.read_csv(filepath, encoding=encoding)
                        print(f"Successfully loaded {len(data)} records using {encoding} encoding")
                        return data
                    except UnicodeDecodeError:
                        continue
                        
            elif file_type.lower() in ['excel', 'xlsx', 'xls']:
                data = pd.read_excel(filepath)
                print(f"Successfully loaded {len(data)} records from Excel file")
                return data
                
            elif file_type.lower() == 'json':
                data = pd.read_json(filepath)
                print(f"Successfully loaded {len(data)} records from JSON file")
                return data
                
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def standardize_gender(self, gender_series):
        """
        Standardize gender field variations.
        
        Args:
            gender_series (pd.Series): Series containing gender data
            
        Returns:
            pd.Series: Standardized gender values
        """
        return gender_series.map(self.gender_mapping).fillna('Unknown')
    
    def standardize_insurance(self, insurance_series):
        """
        Standardize insurance type variations.
        
        Args:
            insurance_series (pd.Series): Series containing insurance data
            
        Returns:
            pd.Series: Standardized insurance values
        """
        def clean_insurance(value):
            if pd.isna(value) or value == '' or value is None:
                return 'Unknown'
            
            value_upper = str(value).upper().strip()
            
            # Check against keyword mappings
            for standard_type, keywords in self.insurance_keywords.items():
                if any(keyword.upper() in value_upper for keyword in keywords):
                    return standard_type
            
            # Default to 'Other' if no match found
            return 'Other'
        
        return insurance_series.apply(clean_insurance)
    
    def parse_dates(self, date_series, column_name="date"):
        """
        Parse dates from multiple inconsistent formats.
        
        Args:
            date_series (pd.Series): Series containing date strings
            column_name (str): Name of the column for error reporting
            
        Returns:
            pd.Series: Parsed datetime objects
        """
        parsed_dates = []
        failed_parses = 0
        
        for i, date_str in enumerate(date_series):
            if pd.isna(date_str) or date_str == '' or str(date_str).upper() in ['INVALID_DATE', 'NULL', 'NONE']:
                parsed_dates.append(pd.NaT)
                failed_parses += 1
                continue
            
            parsed = None
            
            # Try each format
            for fmt in self.date_formats:
                try:
                    parsed = pd.to_datetime(str(date_str), format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            
            # If no format worked, try pandas automatic parsing
            if parsed is None:
                try:
                    parsed = pd.to_datetime(str(date_str), errors='coerce')
                except:
                    parsed = pd.NaT
                    failed_parses += 1
            
            parsed_dates.append(parsed)
        
        success_rate = ((len(date_series) - failed_parses) / len(date_series)) * 100
        print(f"{column_name} parsing: {failed_parses} failed out of {len(date_series)} ({success_rate:.1f}% success rate)")
        
        return pd.Series(parsed_dates, index=date_series.index)
    
    def handle_outliers(self, series, method='clip', lower_percentile=1, upper_percentile=99):
        """
        Handle outliers in numeric data using various methods.
        
        Args:
            series (pd.Series): Numeric series to process
            method (str): Method to use ('clip', 'remove', 'cap', 'none')
            lower_percentile (float): Lower percentile for outlier detection
            upper_percentile (float): Upper percentile for outlier detection
            
        Returns:
            pd.Series: Series with outliers handled
        """
        if method == 'clip':
            lower_bound = series.quantile(lower_percentile / 100)
            upper_bound = series.quantile(upper_percentile / 100)
            outliers_count = ((series < lower_bound) | (series > upper_bound)).sum()
            print(f"Clipping {outliers_count} outliers in range [{lower_bound:.2f}, {upper_bound:.2f}]")
            return series.clip(lower_bound, upper_bound)
            
        elif method == 'remove':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (series >= lower_bound) & (series <= upper_bound)
            outliers_removed = (~mask).sum()
            print(f"Removing {outliers_removed} outliers using IQR method")
            return series[mask]
            
        elif method == 'cap':
            lower_bound = series.quantile(0.05)
            upper_bound = series.quantile(0.95)
            series_capped = series.copy()
            series_capped[series_capped < lower_bound] = lower_bound
            series_capped[series_capped > upper_bound] = upper_bound
            return series_capped
            
        else:  # method == 'none'
            return series
    
    def impute_missing_values(self, df, strategy='domain_specific'):
        """
        Impute missing values using domain-appropriate strategies.
        
        Args:
            df (pd.DataFrame): DataFrame with missing values
            strategy (str): Imputation strategy to use
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        df_imputed = df.copy()
        imputation_report = {}
        
        for column in df_imputed.columns:
            missing_count = df_imputed[column].isnull().sum()
            if missing_count == 0:
                continue
                
            original_missing = missing_count
            
            if strategy == 'domain_specific':
                # Healthcare-specific imputation rules
                if 'age' in column.lower():
                    # Use median for age
                    median_age = df_imputed[column].median()
                    df_imputed[column].fillna(median_age, inplace=True)
                    imputation_report[column] = f"Median imputation: {median_age:.1f}"
                    
                elif 'admission' in column.lower() and 'previous' in column.lower():
                    # Conservative approach: assume 0 previous admissions if missing
                    df_imputed[column].fillna(0, inplace=True)
                    imputation_report[column] = "Zero imputation (conservative)"
                    
                elif 'length' in column.lower() and 'stay' in column.lower():
                    # Use median for length of stay
                    median_los = df_imputed[column].median()
                    df_imputed[column].fillna(median_los, inplace=True)
                    imputation_report[column] = f"Median imputation: {median_los:.1f}"
                    
                elif df_imputed[column].dtype == 'object':
                    # For categorical variables, use mode or 'Unknown'
                    if not df_imputed[column].mode().empty:
                        mode_val = df_imputed[column].mode()[0]
                        df_imputed[column].fillna(mode_val, inplace=True)
                        imputation_report[column] = f"Mode imputation: {mode_val}"
                    else:
                        df_imputed[column].fillna('Unknown', inplace=True)
                        imputation_report[column] = "Unknown imputation"
                        
                else:
                    # For other numeric variables, use median
                    if df_imputed[column].dtype in ['int64', 'float64']:
                        median_val = df_imputed[column].median()
                        df_imputed[column].fillna(median_val, inplace=True)
                        imputation_report[column] = f"Median imputation: {median_val:.2f}"
            
            final_missing = df_imputed[column].isnull().sum()
            if original_missing > final_missing:
                print(f"{column}: Imputed {original_missing - final_missing} missing values")
        
        return df_imputed, imputation_report
    
    def validate_data_quality(self, df):
        """
        Perform comprehensive data quality validation.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            dict: Data quality report
        """
        quality_report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'duplicates': df.duplicated().sum(),
            'issues': []
        }
        
        # Missing values analysis
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_report['missing_values'][column] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_pct > 50:
                quality_report['issues'].append(f"{column}: >50% missing values ({missing_pct:.1f}%)")
        
        # Data types
        quality_report['data_types'] = dict(df.dtypes.astype(str))
        
        # Check for potential issues
        for column in df.columns:
            if df[column].dtype == 'object':
                unique_values = df[column].nunique()
                if unique_values > len(df) * 0.9:
                    quality_report['issues'].append(f"{column}: High cardinality ({unique_values} unique values)")
        
        # Age validation (if age column exists)
        age_columns = [col for col in df.columns if 'age' in col.lower()]
        for age_col in age_columns:
            if df[age_col].dtype in ['int64', 'float64']:
                invalid_ages = ((df[age_col] < 0) | (df[age_col] > 120)).sum()
                if invalid_ages > 0:
                    quality_report['issues'].append(f"{age_col}: {invalid_ages} invalid age values")
        
        return quality_report
    
    def clean_text_fields(self, df, text_columns=None):
        """
        Clean and standardize text fields.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            text_columns (list): List of text columns to clean
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text fields
        """
        if text_columns is None:
            text_columns = df.select_dtypes(include=['object']).columns
        
        df_cleaned = df.copy()
        
        for column in text_columns:
            if column in df_cleaned.columns:
                # Remove extra whitespace
                df_cleaned[column] = df_cleaned[column].astype(str).str.strip()
                
                # Standardize case for certain fields
                if any(keyword in column.lower() for keyword in ['gender', 'sex']):
                    df_cleaned[column] = df_cleaned[column].str.title()
                elif 'insurance' in column.lower():
                    df_cleaned[column] = df_cleaned[column].str.title()
                
                # Remove special characters from certain fields
                if any(keyword in column.lower() for keyword in ['id', 'number']):
                    df_cleaned[column] = df_cleaned[column].str.replace(r'[^\w\s]', '', regex=True)
        
        return df_cleaned
    
    def prepare_for_modeling(self, df, target_col='readmission_30_day'):
        """
        Prepare data specifically for machine learning models.
        This removes non-numeric columns that can't be used in ML models.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            target_col (str): Target variable column name
            
        Returns:
            tuple: (X, y, excluded_columns)
        """
        df_model = df.copy()
        excluded_columns = []
        
        # Remove datetime columns (convert to numeric features if needed)
        datetime_cols = df_model.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            if 'date' in col.lower():
                # Convert to days since reference date for modeling
                reference_date = pd.Timestamp('2014-01-01')
                df_model[f'{col}_days'] = (df_model[col] - reference_date).dt.days
                excluded_columns.append(col)
                df_model.drop(col, axis=1, inplace=True)
        
        # Remove non-numeric object columns that aren't encoded
        object_cols = df_model.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col != target_col and not col.endswith('_std_Female') and not col.endswith('_std_Male'):
                excluded_columns.append(col)
                df_model.drop(col, axis=1, inplace=True)
        
        # Remove patient ID and other identifier columns
        id_cols = [col for col in df_model.columns if 'id' in col.lower()]
        for col in id_cols:
            excluded_columns.append(col)
            df_model.drop(col, axis=1, inplace=True)
        
        # Split features and target
        if target_col in df_model.columns:
            y = df_model[target_col]
            X = df_model.drop(target_col, axis=1)
        else:
            y = None
            X = df_model
        
        # Ensure all remaining columns are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            excluded_columns.append(col)
            X.drop(col, axis=1, inplace=True)
        
        print(f"Modeling preparation: Excluded {len(excluded_columns)} non-numeric columns")
        if excluded_columns:
            print(f"Excluded columns: {excluded_columns[:5]}{'...' if len(excluded_columns) > 5 else ''}")
        
        return X, y, excluded_columns
    
    def process_pipeline(self, df, config=None):
        """
        Complete data processing pipeline.
        
        Args:
            df (pd.DataFrame): Raw data to process
            config (dict): Configuration for processing steps
            
        Returns:
            tuple: (processed_df, processing_report)
        """
        if config is None:
            config = {
                'clean_text': True,
                'standardize_gender': True,
                'standardize_insurance': True,
                'parse_dates': True,
                'handle_outliers': True,
                'impute_missing': True,
                'validate_quality': True,
                'prepare_for_modeling': True
            }
        
        processing_report = {
            'original_shape': df.shape,
            'steps_completed': [],
            'issues_found': [],
            'final_shape': None,
            'excluded_columns': []
        }
        
        processed_df = df.copy()
        
        print("Starting EHR data processing pipeline...")
        
        # Step 1: Initial data quality validation
        if config.get('validate_quality', True):
            quality_report = self.validate_data_quality(processed_df)
            processing_report['initial_quality'] = quality_report
            processing_report['steps_completed'].append('Initial quality validation')
            print(f"Initial quality check: {len(quality_report['issues'])} issues identified")
        
        # Step 2: Clean text fields
        if config.get('clean_text', True):
            processed_df = self.clean_text_fields(processed_df)
            processing_report['steps_completed'].append('Text cleaning')
            print("Text fields cleaned and standardized")
        
        # Step 3: Standardize gender
        if config.get('standardize_gender', True):
            gender_columns = [col for col in processed_df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
            for col in gender_columns:
                processed_df[col] = self.standardize_gender(processed_df[col])
                processing_report['steps_completed'].append(f'Gender standardization: {col}')
            if gender_columns:
                print(f"Standardized {len(gender_columns)} gender columns")
        
        # Step 4: Standardize insurance
        if config.get('standardize_insurance', True):
            insurance_columns = [col for col in processed_df.columns if 'insurance' in col.lower()]
            for col in insurance_columns:
                processed_df[col] = self.standardize_insurance(processed_df[col])
                processing_report['steps_completed'].append(f'Insurance standardization: {col}')
            if insurance_columns:
                print(f"Standardized {len(insurance_columns)} insurance columns")
        
        # Step 5: Parse dates
        if config.get('parse_dates', True):
            date_columns = [col for col in processed_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                if processed_df[col].dtype == 'object':
                    processed_df[col] = self.parse_dates(processed_df[col], col)
                    processing_report['steps_completed'].append(f'Date parsing: {col}')
            if date_columns:
                print(f"Parsed {len(date_columns)} date columns")
        
        # Step 6: Handle outliers
        if config.get('handle_outliers', True):
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            outlier_columns = [col for col in numeric_columns if 'length' in col.lower() or 'stay' in col.lower()]
            for col in outlier_columns:
                processed_df[col] = self.handle_outliers(processed_df[col], method='clip')
                processing_report['steps_completed'].append(f'Outlier handling: {col}')
            if outlier_columns:
                print(f"Handled outliers in {len(outlier_columns)} columns")
        
        # Step 7: Impute missing values
        if config.get('impute_missing', True):
            processed_df, imputation_report = self.impute_missing_values(processed_df)
            processing_report['imputation_details'] = imputation_report
            processing_report['steps_completed'].append('Missing value imputation')
            print(f"Imputed missing values in {len(imputation_report)} columns")
        
        # Final quality validation
        if config.get('validate_quality', True):
            final_quality = self.validate_data_quality(processed_df)
            processing_report['final_quality'] = final_quality
            processing_report['steps_completed'].append('Final quality validation')
            print(f"Final quality check: {len(final_quality['issues'])} issues remaining")
        
        processing_report['final_shape'] = processed_df.shape
        
        print(f"Processing pipeline complete: {df.shape} -> {processed_df.shape}")
        print(f"Steps completed: {len(processing_report['steps_completed'])}")
        
        return processed_df, processing_report


def load_sample_data():
    """
    Generate sample EHR data for testing purposes.
    This mimics the data quality issues found in real EHR systems.
    """
    np.random.seed(42)
    n_patients = 1000
    
    # Generate sample data with realistic issues
    data = {
        'patient_id': [f"PID_{str(i).zfill(6)}" for i in range(1, n_patients + 1)],
        'age': np.random.normal(65, 15, n_patients),
        'gender': np.random.choice(['M', 'F', 'Male', 'Female', 'MALE', 'FEMALE', '', 'Unknown'], n_patients),
        'admission_date': [
            np.random.choice([
                '2014-01-15', '01/15/2014', '15-01-2014', '20140115', 
                'INVALID_DATE', '', None
            ]) for _ in range(n_patients)
        ],
        'length_of_stay': np.random.exponential(5, n_patients),
        'insurance_type': np.random.choice([
            'Medicare', 'Medicaid', 'Private', 'MEDICARE', 'medicare', 
            'Commercial', 'Self-Pay', '', 'Unknown'
        ], n_patients),
        'previous_admissions': np.random.poisson(2, n_patients)
    }
    
    # Add some missing values
    missing_indices = np.random.choice(n_patients, int(0.1 * n_patients), replace=False)
    for idx in missing_indices:
        if np.random.random() < 0.5:
            data['age'][idx] = np.nan
        if np.random.random() < 0.3:
            data['previous_admissions'][idx] = np.nan
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    print("EHR Data Processing Module - Example Usage")
    print("=" * 50)
    
    # Load sample data
    print("Loading sample data...")
    sample_data = load_sample_data()
    print(f"Sample data shape: {sample_data.shape}")
    
    # Initialize processor
    processor = EHRDataProcessor()
    
    # Run processing pipeline
    print("\nRunning processing pipeline...")
    processed_data, report = processor.process_pipeline(sample_data)
    
    print(f"\nProcessing complete!")
    print(f"Original shape: {report['original_shape']}")
    print(f"Final shape: {report['final_shape']}")
    print(f"Steps completed: {len(report['steps_completed'])}")
    
    # Test modeling preparation
    print("\nTesting modeling preparation...")
    X, y, excluded = processor.prepare_for_modeling(processed_data)
    print(f"Modeling data: {X.shape if X is not None else 'None'}")
    print(f"Target: {y.shape if y is not None else 'None'}")
    print(f"Excluded columns: {len(excluded)}")
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(processed_data.head())