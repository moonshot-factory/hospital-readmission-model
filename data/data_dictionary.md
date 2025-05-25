# Hospital Readmission Dataset - Data Dictionary

## Overview
This synthetic dataset mimics the real Electronic Health Record (EHR) data encountered during the 2015 hospital readmission risk prediction internship project in Southeast Texas. The data demonstrates typical data quality challenges found in multi-hospital EHR systems.

**Dataset**: `sample_data.csv`  
**Records**: 100 patient admissions  
**Time Period**: January 2014 - December 2015  
**Source**: Synthetic data based on real EHR patterns  

## Data Quality Issues Represented

This dataset intentionally includes common EHR data quality problems:

- **Inconsistent Date Formats**: Multiple date formats across records
- **Missing Values**: Empty cells in various columns
- **Inconsistent Categorical Values**: Multiple representations of the same category
- **Mixed Medical Coding Systems**: ICD-9, ICD-10, and local hospital codes
- **Varying Text Cases**: Mixed uppercase, lowercase, and title case

## Column Definitions

### Core Identifiers
| Column | Description | Data Type | Example Values |
|--------|-------------|-----------|----------------|
| `patient_id` | Unique patient identifier | String | PID_000001, PID_000002 |

### Demographics
| Column | Description | Data Type | Example Values | Notes |
|--------|-------------|-----------|----------------|-------|
| `age` | Patient age at admission | Numeric | 67.3, 45.8, 78.2 | Decimal values in years |
| `gender` | Patient gender | String | F, Male, FEMALE, m | **Inconsistent formatting** |

### Admission Details
| Column | Description | Data Type | Example Values | Notes |
|--------|-------------|-----------|----------------|-------|
| `admission_date` | Date of hospital admission | String | 2014-03-15, 01/22/2014, 15-06-2014 | **Multiple date formats** |
| `discharge_date` | Date of hospital discharge | String | 2014-03-18, 01/25/2014, 18-06-2014 | **Multiple date formats** |
| `length_of_stay` | Days in hospital | Numeric | 3, 5, 8 | Calculated field |
| `emergency_admission` | Emergency vs scheduled admission | Binary | 0, 1 | 1 = Emergency, 0 = Scheduled |

### Medical Information
| Column | Description | Data Type | Example Values | Notes |
|--------|-------------|-----------|----------------|-------|
| `diagnosis_codes` | Medical diagnosis codes | String | 250.00;401.9, E11.9;I10;I50.9, DM2;HTN;HF | **Mixed coding systems** |
| `previous_admissions` | Count of prior admissions | Numeric | 0, 2, 4 | Some missing values |

### Insurance and Administrative
| Column | Description | Data Type | Example Values | Notes |
|--------|-------------|-----------|----------------|-------|
| `insurance_type` | Insurance provider type | String | Medicare, Private, MEDICARE, medicare | **Inconsistent formatting** |

### Target Variable
| Column | Description | Data Type | Example Values | Notes |
|--------|-------------|-----------|----------------|-------|
| `readmission_30_day` | 30-day readmission occurred | Binary | 0, 1 | 1 = Readmitted, 0 = No readmission |

## Medical Coding Systems Represented

### ICD-9 Codes
- `250.00`, `250.01`, `250.02`, `250.03` - Diabetes mellitus
- `401.9`, `401.0`, `401.1` - Hypertension
- `428.0` - Congestive heart failure
- `585.9`, `585.3` - Chronic kidney disease
- `272.4`, `272.0`, `272.1` - Hyperlipidemia
- `414.01` - Coronary artery disease
- `496` - COPD

### ICD-10 Codes
- `E11.9`, `E11.0` - Type 2 diabetes mellitus
- `I10` - Essential hypertension
- `I50.9` - Heart failure
- `N18.9`, `N18.3` - Chronic kidney disease
- `E78.5`, `E78.0` - Hyperlipidemia
- `I25.10`, `I25.9` - Coronary artery disease
- `I21.9` - Myocardial infarction

### Local Hospital Codes
- `DM2`, `DIABETES` - Diabetes
- `HTN`, `HYPERTENSION` - Hypertension
- `HF`, `CHF`, `HEART_FAILURE`, `HEART` - Heart failure
- `KIDNEY`, `CKD`, `RENAL` - Kidney disease
- `CHOL`, `LIPID`, `CHOLESTEROL` - Cholesterol disorders
- `CAD` - Coronary artery disease
- `COPD` - Chronic obstructive pulmonary disease

## Data Quality Challenges

### 1. Date Format Inconsistencies
- **ISO Format**: 2014-03-15
- **US Format**: 01/22/2014
- **European Format**: 15-06-2014
- **Compact Format**: 20140812
- **Invalid Entries**: INVALID_DATE, empty cells

### 2. Gender Field Variations
- **Male**: M, Male, MALE, m
- **Female**: F, Female, FEMALE, f, female
- **Missing**: Empty cells

### 3. Insurance Type Inconsistencies
- **Medicare**: Medicare, MEDICARE, medicare
- **Private**: Private, PRIVATE, Commercial
- **Medicaid**: Medicaid
- **Self-Pay**: Self-Pay
- **Unknown**: Unknown, empty cells

### 4. Missing Data Patterns
- Random missing values across all columns
- Some patients missing critical information
- Systematic missing patterns in certain fields

## Clinical Context

### Readmission Risk Factors Represented
- **Age**: Elderly patients (>75) at higher risk
- **Comorbidities**: Multiple chronic conditions
- **Previous Admissions**: History of frequent readmissions
- **Emergency Admissions**: Unplanned hospital visits
- **Length of Stay**: Extended hospital stays

### Expected Preprocessing Steps
1. **Date Standardization**: Parse multiple date formats
2. **Gender Standardization**: Normalize to Male/Female/Unknown
3. **Insurance Standardization**: Group similar categories
4. **Medical Code Translation**: Map to standard conditions
5. **Missing Value Imputation**: Domain-appropriate strategies
6. **Outlier Detection**: Identify unrealistic values

## Usage Notes

### For Training Purposes
This dataset is designed to demonstrate:
- Real-world EHR data challenges
- Data preprocessing techniques
- Feature engineering strategies
- Model development workflows

### Privacy and Ethics
- **Synthetic Data**: No real patient information
- **Realistic Patterns**: Based on actual EHR data structures
- **Educational Use**: Suitable for learning and demonstration
- **HIPAA Compliant**: No protected health information

## File Structure
```
data/
├── sample_data.csv          # Main dataset
├── data_dictionary.md       # This documentation
├── processed/              # Cleaned data outputs
│   └── (generated by processing pipeline)
└── raw/                    # Raw data inputs
    └── (placeholder - no actual patient data)
```

## Processing Pipeline Compatibility

This dataset is designed to work with the project's processing modules:

- **`src/data_processing.py`**: Handles all data quality issues
- **`src/feature_engineering.py`**: Creates clinical features
- **`src/models.py`**: Trains prediction models
- **`src/visualization.py`**: Creates analytical visualizations

## Example Usage

```python
import pandas as pd
from src.data_processing import EHRDataProcessor

# Load the dataset
df = pd.read_csv('data/sample_data.csv')

# Initialize processor
processor = EHRDataProcessor()

# Process the data
processed_df, report = processor.process_pipeline(df)

print(f"Original: {df.shape}, Processed: {processed_df.shape}")
```

## Dataset Statistics

- **Total Records**: 100 patient admissions
- **Date Range**: 2014-2015
- **Readmission Rate**: ~15% (realistic for hospital populations)
- **Average Age**: ~64 years
- **Average Length of Stay**: ~5 days
- **Emergency Admissions**: ~60%
- **Missing Values**: ~5-10% across various columns

This dataset effectively demonstrates the complexity and challenges of working with real-world healthcare data while maintaining complete privacy compliance.