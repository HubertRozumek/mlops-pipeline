import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None, is_synthetic: bool = True):
        self.config = config or {}
        self.is_synthetic = is_synthetic
        self.encoders = {}
        self.imputers = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessor on training data"""
        logger.info("Fitting preprocessor on training data")
        
        if self.is_synthetic:
            self._fit_imputers(df)
            self._fit_encoders(df)
        else:
            df_clean = self._clean_telco_data(df)
            self._fit_telco_encoders(df_clean)
        
        self.is_fitted = True
        logger.info("Preprocessor fitting completed")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if self.is_synthetic:
            df = self._apply_imputation(df)
            df = self._apply_encoding(df)
            df = self._feature_engineering(df)
        else:
            df = self._clean_telco_data(df)
            df = self._apply_telco_encoding(df)
            df = self._feature_engineering(df)
  
        logger.info(f"Data transformed: {df.shape}")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def _fit_imputers(self, df: pd.DataFrame):
        """Fit imputers for missing values (synthetic data)"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numerical_cols) > 0:
            self.imputers['numerical'] = SimpleImputer(strategy='median')
            self.imputers['numerical'].fit(df[numerical_cols])
        
        if len(categorical_cols) > 0:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'].fit(df[categorical_cols])
    
    def _fit_encoders(self, df: pd.DataFrame):
        """Fit label encoders for categorical variables (synthetic data)"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Churn':
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str))
    
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers (synthetic data)"""
        df_imputed = df.copy()
        
        if 'numerical' in self.imputers:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                df_imputed[numerical_cols] = self.imputers['numerical'].transform(df[numerical_cols])
        
        if 'categorical' in self.imputers:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                df_imputed[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])
        
        return df_imputed
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders (synthetic data)"""
        df_encoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df_encoded[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    logger.warning(f"Unseen categories in {col}, using most frequent class")
                    df_encoded[col] = encoder.transform([encoder.classes_[0]] * len(df))
        
        return df_encoded
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features (synthetic data)"""
        df_features = df.copy()
        
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df_features['avg_monthly_charges'] = df_features['MonthlyCharges'] / (df_features['tenure'] + 1)
        
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df_features['charges_per_month'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)
        
        if 'age' in df.columns:
            df_features['is_senior'] = (df_features['age'] > 65).astype(int)
        
        if 'tenure' in df.columns:
            df_features['is_new_customer'] = (df_features['tenure'] <= 3).astype(int)
        
        return df_features
    
    def _clean_telco_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning for Telco dataset"""
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df = df.dropna(subset=['TotalCharges'])
        
        return df.drop_duplicates()
    
    def _fit_telco_encoders(self, df: pd.DataFrame):
        """Fit encoders for Telco categorical features"""
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'Churn'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col])
                self.encoders[col] = le
    
    def _apply_telco_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to Telco data"""
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    logger.warning(f"Unseen categories in {col}, using first class")
                    df[col] = encoder.transform([encoder.classes_[0]] * len(df))
        return df
    
    def save(self, filepath: str):
        """Save preprocessor to disk"""
        preprocessor_data = {
            'encoders': self.encoders,
            'imputers': self.imputers,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'is_synthetic': self.is_synthetic
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from disk"""
        preprocessor_data = joblib.load(filepath)
        
        instance = cls(preprocessor_data['config'], preprocessor_data['is_synthetic'])
        instance.encoders = preprocessor_data['encoders']
        instance.imputers = preprocessor_data['imputers']
        instance.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return instance