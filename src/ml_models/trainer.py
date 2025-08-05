import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.data_config = self.config['data']
    
    def train(self, df: pd.DataFrame) -> str:
        """Train model and log to MLflow"""
        
        with mlflow.start_run(run_name=f"{self.model_config['name']}_training") as run:
            # Prepare data
            X = df[self.data_config['feature_columns']]
            y = df[self.data_config['target_column']]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.data_config['test_size'],
                random_state=self.model_config['hyperparameters']['random_state']
            )
            
            # Train model
            model = RandomForestClassifier(**self.model_config['hyperparameters']) 
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred)
            }
            
            # Log to MLflow
            mlflow.log_params(self.model_config['hyperparameters'])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=self.model_config['name']
            )
            
            logger.info(f"Model trained with accuracy: {metrics['accuracy']:.4f}")
            
            return run.info.run_id