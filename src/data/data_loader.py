import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from CSV file"""
        try:
            df = pd.read_csv(self.data_path / "train.csv")
            logger.info(f"Loaded training data: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning("Training data not found, generating synthetic data")
            return self._generate_synthetic_data(1000)
    
    def load_validation_data(self) -> pd.DataFrame:
        """Load validation data"""
        try:
            df = pd.read_csv(self.data_path / "validation.csv")
            logger.info(f"Loaded validation data: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning("Validation data not found, generating synthetic data")
            return self._generate_synthetic_data(200)
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic customer churn data"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'Age': np.random.randint(18, 80, n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 100, n_samples),
            'TotalCharges': np.random.uniform(100, 8000, n_samples),
        })
        
        # Generate target with some correlation
        score = (
            0.3 * (df['MonthlyCharges'] - 20) / 80 + 
            0.3 * (1 - df['tenure'] / 72) +             
            0.2 * (df['Age'] > 65).astype(float)        
        )

        churn_prob = 0.1 + 0.8 * score.clip(0, 1)
     
        df['Churn'] = np.random.binomial(1, churn_prob)
        
        return df