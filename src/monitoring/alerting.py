import smtplib
import requests
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.enabled = config.get('enabled', True)
    
    def send_drift_alert(self, drift_info: Dict[str, Any]):
        """Send alert when data drift is detected"""
        if not self.enabled:
            return
            
        subject = "Data Drift Detected - MLOps Pipeline"
        message = self._format_drift_message(drift_info)
        
        self._send_email(subject, message)
        self._send_slack_notification(subject, message)
    
    def send_performance_alert(self, performance_info: Dict[str, Any]):
        """Send alert when model performance degrades"""
        if not self.enabled:
            return
            
        subject = "⚠️ Model Performance Degradation - MLOps Pipeline"
        message = self._format_performance_message(performance_info)
        
        self._send_email(subject, message)
        self._send_slack_notification(subject, message)
    
    def send_system_alert(self, system_info: Dict[str, Any]):
        """Send system health alerts"""
        if not self.enabled:
            return
            
        subject = "System Alert - MLOps Pipeline"
        message = self._format_system_message(system_info)
        
        self._send_email(subject, message)
        self._send_slack_notification(subject, message)
    
    def _format_drift_message(self, drift_info: Dict[str, Any]) -> str:
        """Format drift alert message"""
        return f"""
Data Drift Alert

Drift Score: {drift_info.get('drift_score', 'N/A')}
Threshold: {drift_info.get('threshold', 'N/A')}
Affected Features: {', '.join([f['feature'] for f in drift_info.get('features_with_drift', [])])}

Time: {drift_info.get('timestamp', 'N/A')}

Action Required: Consider retraining the model with recent data.
        """.strip()
    
    def _format_performance_message(self, perf_info: Dict[str, Any]) -> str:
        """Format performance alert message"""
        return f"""
Model Performance Alert

Current Accuracy: {perf_info.get('current_accuracy', 'N/A')}
Threshold: {perf_info.get('threshold', 'N/A')}
Previous Accuracy: {perf_info.get('previous_accuracy', 'N/A')}

Time: {perf_info.get('timestamp', 'N/A')}

Action Required: Model performance has degraded below acceptable threshold.
        """.strip()
    
    def _send_email(self, subject: str, message: str):
        """Send email notification"""
        if not self.email_config.get('enabled', False):
            return
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['sender']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = subject
            
            msg.attach(MimeText(message, 'plain'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_notification(self, subject: str, message: str):
        """Send Slack notification"""
        if not self.slack_config.get('enabled', False):
            return
            
        try:
            payload = {
                'text': f"{subject}\n```{message}```",
                'channel': self.slack_config.get('channel', '#alerts'),
                'username': 'MLOps Bot',
                'icon_emoji': ':robot_face:'
            }
            
            response = requests.post(
                self.slack_config['webhook_url'],
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(f"Failed to send Slack notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")