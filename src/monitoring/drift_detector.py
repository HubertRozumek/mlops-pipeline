import pandas as pd
import logging
from typing import Dict, Any
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold

    def _get_metric_by_type(self, metrics, metric_type: str):
        for metric in metrics:
            if metric['metric'] == metric_type:
                return metric
        raise ValueError(f"Metric {metric_type} not found in report.")

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            # Define column mapping to specify categorical features
            column_mapping = ColumnMapping()
            column_mapping.categorical_features = self.reference_data.columns.tolist()

            # Initialize the report with DataDriftPreset and column_mapping
            data_drift_report = Report(
                metrics=[DataDriftPreset()],
                options={'column_mapping': column_mapping}
            )

            # Run the drift detection
            data_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )

            report_dict = data_drift_report.as_dict()
            metrics = report_dict['metrics']

            data_drift_table = self._get_metric_by_type(metrics, 'DataDriftTable')

            # Calculate drift metrics
            num_features = len(data_drift_table['result']['drift_by_columns'])
            num_drifted = sum(
                1 for res in data_drift_table['result']['drift_by_columns'].values()
                if res['drift_detected']
            )
            drift_score = num_drifted / num_features if num_features > 0 else 0.0  # Compute drift score as proportion
            drift_detected = drift_score > self.threshold

            drift_result = {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'threshold': self.threshold,
                'features_with_drift': [
                    {'feature': name, 'drift_score': res.get('drift_score', 0.0)}
                    for name, res in data_drift_table['result']['drift_by_columns'].items()
                    if res['drift_detected']
                ],
                'number_of_features': num_features,
                'number_of_drifted_features': num_drifted
            }

            logger.info(f"Drift detected: {drift_detected} | Features drifted: {num_drifted}/{num_features}")
            return drift_result

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {'error': str(e)}