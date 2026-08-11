import pandas
import logging
from amp.utils import form_multi_output
from amp.preprocessing import split_samples_with_feature_windows

logger = logging.getLogger(__name__)


class WindowForecastMixin:
    """Mixin for window-based time-series forecasters.

    Provides ``predict`` (with feature-column filtering) and ``_target_predict``
    (with gap-safe regular-grid re-indexing) for models that slice a DataFrame
    into overlapping sample windows and produce multi-step forecasts.

    Requires the concrete class to provide:
        - ``self.features``             — feature dict (keys = required column names)
        - ``self.outputs``              — list of output column names
        - ``self.max_lag``              — most negative window bound (negative int)
        - ``self.max_future``           — most positive window bound (non-neg int)
        - ``self.forecast_len``         — number of forecast steps
        - ``self.update_rate``          — stride between successive forecast origins (steps)
        - ``self.lead_time``            — lead time in steps
        - ``self.model_resolution_min`` — step size in minutes

    And must implement:
        - ``_single_predict(data)`` — predict for one sample window
    """

    def predict(self, data, output_mode='single'):
        # Keep only feature columns the model actually uses,
        # ignoring extra columns present for other models.
        
        # Some models require ordered columns
        if hasattr(self, 'ordered_features'):
            features = [c for c in self.ordered_features if c in data.columns]
        else:
            features = [c for c in self.features if c in data.columns]

        feature_cols = [c for c in features if c in data.columns]
        data = data[feature_cols].copy()

        if output_mode == 'single':
            return self._single_predict(data)
        elif output_mode == 'target':
            return self._target_predict(data)
        else:
            logger.warning("Unknown output_mode '%s'", output_mode)

    def _target_predict(self, df):
        samples = split_samples_with_feature_windows(
            df, self.features, (self.max_lag, self.max_future), self.update_rate
        )
        warmup_len = -self.max_lag
        freq_td = pandas.Timedelta(minutes=self.model_resolution_min)
        t0 = df.index[warmup_len]  # expected start of first forecast window
        output = []
        for i, sample in enumerate(samples):
            pred = self._single_predict(sample)
            # Re-index to a regular grid so data gaps don't misalign timestamps
            # and cause form_multi_output to crash.
            expected_start = t0 + i * self.update_rate * freq_td
            pred.index = pandas.date_range(
                start=expected_start, periods=self.forecast_len, freq=freq_td
            )
            output.append(pred)
        return form_multi_output(
            output,
            self.outputs,
            self.lead_time,
            self.forecast_len,
            self.update_rate * self.model_resolution_min,
        )
