import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, schema_config: Dict[str, Any] = None):
        self.schema_config = schema_config or self._default_schema()
        self.validation_results = []
        
    def _default_schema(self) -> Dict[str, Any]:
        """Define default data schema"""
        return {
            'required_columns': ['age', 'tenure', 'monthly_charges', 'total_charges'],
            'optional_columns': ['churn'],
            'column_types': {
                'age': 'int64',
                'tenure': 'int64',
                'monthly_charges': 'float64',
                'total_charges': 'float64',
                'churn': 'int64'
            },
            'ranges': {
                'age': (18, 100),
                'tenure': (0, 120),
                'monthly_charges': (0, 200),
                'total_charges': (0, 20000)
            },
            'missing_threshold': 0.1,  # Max 10% missing values
            'outlier_threshold': 0.05   # Max 5% outliers
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema"""
        validation_result = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Check required columns
        missing_required = set(self.schema_config['required_columns']) - set(df.columns)
        if missing_required:
            validation_result['passed'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check data types
        for col, expected_type in self.schema_config['column_types'].items():
            if col in df.columns:
                if not df[col].dtype.name.startswith(expected_type.split('_')[0]):
                    validation_result['warnings'].append(
                        f"Column {col} has type {df[col].dtype}, expected {expected_type}"
                    )
        
        # Check data ranges
        for col, (min_val, max_val) in self.schema_config['ranges'].items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    validation_result['warnings'].append(
                        f"Column {col} has {out_of_range} values outside range [{min_val}, {max_val}]"
                    )
        
        logger.info(f"Schema validation completed: {'PASSED' if validation_result['passed'] else 'FAILED'}")
        return validation_result
    
    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
        validation_result = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check missing values
        missing_pct = df.isnull().sum() / len(df)
        validation_result['metrics']['missing_percentage'] = missing_pct.to_dict()
        
        high_missing_cols = missing_pct[missing_pct > self.schema_config['missing_threshold']]
        if not high_missing_cols.empty:
            validation_result['passed'] = False
            validation_result['errors'].append(
                f"High missing values in columns: {high_missing_cols.to_dict()}"
            )
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        validation_result['metrics']['duplicate_rows'] = duplicate_count
        if duplicate_count > 0:
            validation_result['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Check for outliers
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_pct = outliers / len(df)
                outlier_info[col] = outlier_pct
                
                if outlier_pct > self.schema_config['outlier_threshold']:
                    validation_result['warnings'].append(
                        f"High outlier rate in {col}: {outlier_pct:.2%}"
                    )
        
        validation_result['metrics']['outlier_percentage'] = outlier_info
        
        # Check data distribution
        for col in numerical_cols:
            if col in df.columns:
                # Check for extreme skewness
                skewness = df[col].skew()
                if abs(skewness) > 2:
                    validation_result['warnings'].append(
                        f"High skewness in {col}: {skewness:.2f}"
                    )
        
        logger.info(f"Quality validation completed: {'PASSED' if validation_result['passed'] else 'FAILED'}")
        return validation_result
    
    def validate_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate for data drift between reference and current data"""
        validation_result = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'drift_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check column consistency
        ref_cols = set(reference_df.columns)
        curr_cols = set(current_df.columns)
        
        if ref_cols != curr_cols:
            validation_result['passed'] = False
            validation_result['errors'].append(
                f"Column mismatch. Missing: {ref_cols - curr_cols}, Extra: {curr_cols - ref_cols}"
            )
        
        # Statistical drift detection for numerical columns
        numerical_cols = reference_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in current_df.columns:
                # Mean drift
                ref_mean = reference_df[col].mean()
                curr_mean = current_df[col].mean()
                mean_drift = abs(curr_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
                
                # Standard deviation drift
                ref_std = reference_df[col].std()
                curr_std = current_df[col].std()
                std_drift = abs(curr_std - ref_std) / ref_std if ref_std != 0 else 0
                
                validation_result['drift_metrics'][col] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift
                }
                
                if mean_drift > 0.1:  # 10% threshold
                    validation_result['warnings'].append(
                        f"Significant mean drift in {col}: {mean_drift:.2%}"
                    )
                
                if std_drift > 0.2:  # 20% threshold
                    validation_result['warnings'].append(
                        f"Significant std drift in {col}: {std_drift:.2%}"
                    )
        
        logger.info(f"Drift validation completed: {'PASSED' if validation_result['passed'] else 'FAILED'}")
        return validation_result
    
    def generate_report(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive data validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'validation_results': {}
        }
        
        # Schema validation
        report['validation_results']['schema'] = self.validate_schema(df)
        
        # Quality validation
        report['validation_results']['quality'] = self.validate_quality(df)
        
        # Drift validation (if reference data provided)
        if reference_df is not None:
            report['validation_results']['drift'] = self.validate_drift(reference_df, df)
        
        # Overall status
        all_passed = all(
            result.get('passed', True) 
            for result in report['validation_results'].values()
        )
        report['overall_status'] = 'PASSED' if all_passed else 'FAILED'
        
        # Summary statistics
        report['summary_statistics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        logger.info(f"Validation report generated: {report['overall_status']}")
        return report
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save validation report to file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to {filepath}")