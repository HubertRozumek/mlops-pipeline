import psutil
import time
import logging
from prometheus_client import Gauge, Counter, Info
from typing import Dict, Any
import threading

logger = logging.getLogger(__name__)

# System metrics
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percentage')

# Model metrics
MODEL_VERSION = Info('model_version_info', 'Current model version information')
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Current model accuracy score')
DRIFT_SCORE = Gauge('data_drift_score', 'Current data drift score')

class MetricsCollector:
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
        
    def start_collection(self):
        """Start metrics collection in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_metrics_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        # Memory usage
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        CPU_USAGE.set(cpu_percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        DISK_USAGE.set(disk_percent)
    
    def update_model_metrics(self, version: str, accuracy: float):
        """Update model-specific metrics"""
        MODEL_VERSION.info({'version': version, 'timestamp': str(time.time())})
        MODEL_ACCURACY.set(accuracy)
    
    def update_drift_score(self, drift_score: float):
        """Update data drift score"""
        DRIFT_SCORE.set(drift_score)