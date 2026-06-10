from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Optional
from cmdstanpy import disable_logging
from src.detector.base_detector import BaseDetector

class ProphetDetector(BaseDetector):
    """
    Prophet detector for anomaly detection
    """

    def __init__(
        self,
        anom_threshold: Optional[float] = None,
    ):
        self._threshold = anom_threshold
        super().__init__()

    def init_detector(self):
        self._detector_name = "prophet"
        self._anom_threshold = self._threshold

    def data_process(self, data):
        data.reset_index(inplace=True)
        data.columns = ['ds', 'y']

        return data
    
    def anomaly_score(self, data):
        data = self.data_process(data)
        model = Prophet(changepoint_range=1)
        with disable_logging():
            model.fit(data)
        delta_trend = np.abs(np.nanmean(model.params["delta"], axis=0))

        changepoint_timestamps = model.changepoints
        anomalous_timestamps = changepoint_timestamps[delta_trend > self._anom_threshold]
        
        score = pd.DataFrame({"value": False}, index=data['ds'])
        score.loc[anomalous_timestamps, "value"] = True

        return score









    
