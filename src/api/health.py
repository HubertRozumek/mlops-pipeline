import psutil
import time
import requests
import pandas
import sklearn
import mlflow
from typing import Dict, Any
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthChecker:
    def __init__(self):
        self.start_time = time.time()
        self.mlflow_uri = "http://mlflow:5000"  # Default MLflow URI
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'checks': {}
        }
        
        # Check individual components
        checks = [
            ('api', self._check_api_health()),
            ('model', self._check_model_health()),
            ('mlflow', self._check_mlflow_health()),
            ('system', self._check_system_health()),
            ('dependencies', self._check_dependencies())
        ]
        
        # Run all checks
        for check_name, check_func in checks:
            try:
                health_status['checks'][check_name] = await check_func
            except Exception as e:
                health_status['checks'][check_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'degraded'
        
        # Determine overall status
        if any(check.get('status') == 'unhealthy' for check in health_status['checks'].values()):
            health_status['status'] = 'unhealthy'
        elif any(check.get('status') == 'degraded' for check in health_status['checks'].values()):
            health_status['status'] = 'degraded'
        
        return health_status
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        return {
            'status': 'healthy',
            'uptime_seconds': time.time() - self.start_time,
            'memory_usage_mb': psutil.virtual_memory().used / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }
    
    async def _check_model_health(self) -> Dict[str, Any]:
        from .main import model
        """Check if model is loaded and functional"""
        try:
            # This would check if global model is loaded
            # For now, we'll simulate the check
            return {
                'status': 'healthy' if model else 'unhealthy',
                'model_loaded': bool(model),
                'model_version': getattr(model, 'version', 'none'),
                'last_prediction_time': None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_loaded': False,
                'error': str(e)
            }
    
    async def _check_mlflow_health(self) -> Dict[str, Any]:
        """Check MLflow connectivity"""
        try:
            # Try to connect to MLflow
            response = requests.get(f"{self.mlflow_uri}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'mlflow_uri': self.mlflow_uri,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
            else:
                return {
                    'status': 'degraded',
                    'mlflow_uri': self.mlflow_uri,
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unhealthy',
                'mlflow_uri': self.mlflow_uri,
                'error': str(e)
            }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Define thresholds
            memory_threshold = 90  # 90%
            disk_threshold = 85    # 85%
            cpu_threshold = 80     # 80%
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = memory.percent
            disk_percent = (disk.used / disk.total) * 100
            
            status = 'healthy'
            issues = []
            
            if cpu_percent > cpu_threshold:
                status = 'degraded'
                issues.append(f'High CPU usage: {cpu_percent:.1f}%')
            
            if memory_percent > memory_threshold:
                status = 'degraded'
                issues.append(f'High memory usage: {memory_percent:.1f}%')
            
            if disk_percent > disk_threshold:
                status = 'degraded'
                issues.append(f'High disk usage: {disk_percent:.1f}%')
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'available_memory_gb': memory.available / (1024**3),
                'available_disk_gb': disk.free / (1024**3),
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies"""
        dependencies = {
            'status': 'healthy',
            'services': {}
        }
        
        # Check if we can import key libraries
        try:
            dependencies['services']['pandas'] = {'status': 'healthy', 'version': pandas.__version__}
        except ImportError as e:
            dependencies['services']['pandas'] = {'status': 'unhealthy', 'error': str(e)}
            dependencies['status'] = 'unhealthy'
        
        try:
            dependencies['services']['sklearn'] = {'status': 'healthy', 'version': sklearn.__version__}
        except ImportError as e:
            dependencies['services']['sklearn'] = {'status': 'unhealthy', 'error': str(e)}
            dependencies['status'] = 'unhealthy'
        
        try:
            dependencies['services']['mlflow'] = {'status': 'healthy', 'version': mlflow.__version__}
        except ImportError as e:
            dependencies['services']['mlflow'] = {'status': 'unhealthy', 'error': str(e)}
            dependencies['status'] = 'unhealthy'
        
        return dependencies
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Check if service is ready to serve requests"""
        # Simple readiness check - can be extended
        try:
            # Check if we can allocate memory
            test_data = [0] * 1000
            del test_data
            
            return {
                'ready': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_liveness_status(self) -> Dict[str, Any]:
        """Check if service is alive"""
        return {
            'alive': True,
            'uptime_seconds': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        }