import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
import logging
import time
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_uri: str = None, preprocessor_path: str = None):
        self.model = None
        self.preprocessor = None
        self.model_uri = model_uri
        self.preprocessor_path = preprocessor_path
        self.model_metadata = {}
        self.is_loaded = False
        
    def load_model(self, model_uri: str = None) -> bool:
        """Load model from MLflow or local path"""
        uri = model_uri or self.model_uri
        
        if not uri:
            logger.error("No model URI provided")
            return False
        
        try:
            start_time = time.time()
            
            if uri.startswith('models:/'):
                # Load from MLflow model registry
                self.model = mlflow.sklearn.load_model(uri)
                logger.info(f"Model loaded from registry: {uri}")
            elif uri.startswith('runs:/'):
                # Load from MLflow run
                self.model = mlflow.sklearn.load_model(uri)
                logger.info(f"Model loaded from run: {uri}")
            else:
                # Load from local path
                self.model = joblib.load(uri)
                logger.info(f"Model loaded from local path: {uri}")
            
            # Load model metadata if available
            self._load_model_metadata(uri)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_preprocessor(self, preprocessor_path: str = None) -> bool:
        """Load data preprocessor"""
        path = preprocessor_path or self.preprocessor_path
        
        if not path:
            logger.warning("No preprocessor path provided, skipping preprocessing")
            return True
        
        try:
            from ..data.preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor.load(path)
            logger.info(f"Preprocessor loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return False
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict[str, Any]:
        """Make predictions on input data"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Input data must be dict, list of dicts, or DataFrame")
            
            # Preprocess data if preprocessor is available
            if self.preprocessor:
                df_processed = self.preprocessor.transform(df)
            else:
                df_processed = df
            
            # Ensure required columns exist
            required_columns = ['age', 'tenure', 'monthly_charges', 'total_charges']
            missing_cols = [col for col in required_columns if col not in df_processed.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Make predictions
            predictions = self.model.predict(df_processed[required_columns])
            
            # Get prediction probabilities if available
            try:
                probabilities = self.model.predict_proba(df_processed[required_columns])
                prob_positive = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            except AttributeError:
                # Model doesn't support predict_proba
                prob_positive = np.full(len(predictions), 0.5)
            
            prediction_time = time.time() - start_time
            
            # Format results
            if len(predictions) == 1:
                # Single prediction
                result = {
                    'prediction': int(predictions[0]),
                    'probability': float(prob_positive[0]),
                    'model_version': self.model_metadata.get('version', 'unknown'),
                    'prediction_time_ms': prediction_time * 1000,
                    'features_used': required_columns
                }
            else:
                # Batch predictions
                result = {
                    'predictions': [int(p) for p in predictions],
                    'probabilities': [float(p) for p in prob_positive],
                    'model_version': self.model_metadata.get('version', 'unknown'),
                    'prediction_time_ms': prediction_time * 1000,
                    'batch_size': len(predictions),
                    'features_used': required_columns
                }
            
            logger.info(f"Prediction completed in {prediction_time*1000:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, data_list: List[Dict]) -> List[Dict[str, Any]]:
        """Make predictions on batch of data"""
        df = pd.DataFrame(data_list)
        batch_result = self.predict(df)
        
        # Convert batch result to list of individual results
        individual_results = []
        for i in range(len(data_list)):
            individual_results.append({
                'prediction': batch_result['predictions'][i],
                'probability': batch_result['probabilities'][i],
                'model_version': batch_result['model_version'],
                'features_used': batch_result['features_used']
            })
        
        return individual_results
    
    def explain_prediction(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        """Provide explanation for prediction (basic feature importance)"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_names = ['age', 'tenure', 'monthly_charges', 'total_charges']
                importances = self.model.feature_importances_
                
                feature_importance = dict(zip(feature_names, importances))
                
                # Sort by importance
                sorted_importance = dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True))
                
                return {
                    'feature_importance': sorted_importance,
                    'most_important_feature': max(feature_importance.items(), key=lambda x: x[1])[0],
                    'explanation_available': True
                }
            else:
                return {
                    'feature_importance': {},
                    'explanation_available': False,
                    'reason': 'Model does not support feature importance'
                }
                
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {
                'feature_importance': {},
                'explanation_available': False,
                'reason': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if not self.is_loaded:
            return {'loaded': False}
        
        info = {
            'loaded': True,
            'model_type': type(self.model).__name__,
            'model_uri': self.model_uri,
            'preprocessor_loaded': self.preprocessor is not None,
            'metadata': self.model_metadata
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        
        if hasattr(self.model, 'classes_'):
            info['classes'] = self.model.classes_.tolist()
        
        return info
    
    def _load_model_metadata(self, model_uri: str):
        """Load model metadata from MLflow"""
        try:
            if model_uri.startswith(('models:/', 'runs:/')):
                client = mlflow.tracking.MlflowClient()
                
                if model_uri.startswith('models:/'):
                    # Parse model registry URI
                    parts = model_uri.replace('models:/', '').split('/')
                    model_name = parts[0]
                    model_version = parts[1] if len(parts) > 1 else 'latest'
                    
                    if model_version == 'latest':
                        model_version_obj = client.get_latest_versions(model_name)[0]
                    else:
                        model_version_obj = client.get_model_version(model_name, model_version)
                    
                    self.model_metadata = {
                        'name': model_name,
                        'version': model_version_obj.version,
                        'stage': model_version_obj.current_stage,
                        'run_id': model_version_obj.run_id
                    }
                
                elif model_uri.startswith('runs:/'):
                    # Parse run URI
                    run_id = model_uri.split('/')[1]
                    run = client.get_run(run_id)
                    
                    self.model_metadata = {
                        'run_id': run_id,
                        'experiment_id': run.info.experiment_id,
                        'metrics': run.data.metrics,
                        'params': run.data.params
                    }
                    
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
            self.model_metadata = {}