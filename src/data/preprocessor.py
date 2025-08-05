import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessor on training data"""
        logger.info("Fitting preprocessor on training data")
        
        # Handle missing values
        self._fit_imputers(df)
        
        # Handle categorical encoding
        self._fit_encoders(df)
        
        # Handle numerical scaling
        self._fit_scalers(df)
        
        self.is_fitted = True
        logger.info("Preprocessor fitting completed")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_transformed = df.copy()
        
        # Apply imputation
        df_transformed = self._apply_imputation(df_transformed)
        
        # Apply encoding
        df_transformed = self._apply_encoding(df_transformed)
        
        # Apply scaling
        df_transformed = self._apply_scaling(df_transformed)
        
        # Feature engineering
        df_transformed = self._feature_engineering(df_transformed)
        
        logger.info(f"Data transformed: {df_transformed.shape}")
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def _fit_imputers(self, df: pd.DataFrame):
        """Fit imputers for missing values"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Numerical imputation (median)
        if len(numerical_cols) > 0:
            self.imputers['numerical'] = SimpleImputer(strategy='median')
            self.imputers['numerical'].fit(df[numerical_cols])
        
        # Categorical imputation (most frequent)
        if len(categorical_cols) > 0:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'].fit(df[categorical_cols])
    
    def _fit_encoders(self, df: pd.DataFrame):
        """Fit label encoders for categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'churn':  # Don't encode target
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str))
    
    def _fit_scalers(self, df: pd.DataFrame):
        """Fit scalers for numerical variables"""
        numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
        existing_cols = [col for col in numerical_cols if col in df.columns]
        
        if existing_cols:
            self.scalers['standard'] = StandardScaler()
            self.scalers['standard'].fit(df[existing_cols])
    
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers"""
        df_imputed = df.copy()
        
        # Numerical imputation
        if 'numerical' in self.imputers:
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                df_imputed[numerical_cols] = self.imputers['numerical'].transform(df[numerical_cols])
        
        # Categorical imputation
        if 'categorical' in self.imputers:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                df_imputed[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])
        
        return df_imputed
    
    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders"""
        df_encoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # Handle unseen categories
                try:
                    df_encoded[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # For unseen categories, use the most frequent class
                    logger.warning(f"Unseen categories in {col}, using most frequent class")
                    df_encoded[col] = encoder.transform([encoder.classes_[0]] * len(df))
        
        return df_encoded
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers"""
        df_scaled = df.copy()
        
        if 'standard' in self.scalers:
            numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
            existing_cols = [col for col in numerical_cols if col in df.columns]
            
            if existing_cols:
                df_scaled[existing_cols] = self.scalers['standard'].transform(df[existing_cols])
        
        return df_scaled
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features"""
        df_features = df.copy()
        
        # Create derived features
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            df_features['avg_monthly_charges'] = df_features['monthly_charges'] / (df_features['tenure'] + 1)
        
        if 'total_charges' in df.columns and 'tenure' in df.columns:
            df_features['charges_per_month'] = df_features['total_charges'] / (df_features['tenure'] + 1)
        
        if 'age' in df.columns:
            df_features['is_senior'] = (df_features['age'] > 65).astype(int)
        
        if 'tenure' in df.columns:
            df_features['is_new_customer'] = (df_features['tenure'] <= 12).astype(int)
        
        if 'monthly_charges' in df.columns:
            df_features['high_charges'] = (df_features['monthly_charges'] > df_features['monthly_charges'].mean()).astype(int)
        
        return df_features
    
    def save(self, filepath: str):
        """Save preprocessor to disk"""
        preprocessor_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from disk"""
        preprocessor_data = joblib.load(filepath)
        
        instance = cls(preprocessor_data['config'])
        instance.scalers = preprocessor_data['scalers']
        instance.encoders = preprocessor_data['encoders']
        instance.imputers = preprocessor_data['imputers']
        instance.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return instance