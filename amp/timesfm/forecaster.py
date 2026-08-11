import warnings
import joblib
import time
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from amp.base import BaseModel
from amp.efp.preprocessing import add_temporal_columns, HOUR, WEEKDAY, HOLIDAY, MONTH

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # TimesFM uses JAX under the hood, and setting this environment variable forces it to use CPU, which is more compatible with most environments. If you have a compatible GPU and want to use it, you can remove this line.

BATCH_SIZE = 64  # 64 is optimal for most CPUs; increase to 128/256 if using GPU


class TimesFMForecaster(BaseModel):
    """
    TimesFM forecaster with optional covariates.
    Supports sliding window predictions in 'target' or 'single' mode.
    If no covariates are provided, operates in target-only mode.
    """

    def __init__(self, targets, lead_time, forecast_len, data_freq, features=None, update_freq=1):
        self.targets = targets
        self.lead_time = lead_time
        self.horizon = forecast_len
        self.data_freq = data_freq
        self._features = features
        self.context_len = abs(features['lagged_target']['windows'][0][0])
        self.forecast_len = self.horizon
        self.update_rate = update_freq

    def _load_model(self):
        self.model_name = "google/timesfm-2.5-200m-pytorch"
        try:
            import timesfm
            import torch
            import os
            torch.set_float32_matmul_precision("high")
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        except ImportError:
            warnings.warn("TimesFM not installed. Please install timesfm.")
            raise

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
        print(f'Loaded TimesFM model: {self.model_name} with context length {self.context_len} and horizon {self.horizon}')
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=True
            )
        )

    @property
    def input_window(self):
        return -self.context_len, self.horizon

    @property
    def outputs(self):
        return [t.output for t in self.targets]

    def predict(self, input_df, output_mode="target"):
        """
        Generate sliding-window forecasts using TimesFM with optional covariates.

        Parameters
        ----------
        input_df : pd.DataFrame
            Input dataframe containing target columns and optional covariates
        output_mode : str
            "target" returns multiple forecasts per timestamp
            "single" returns one forecast per timestamp (online mode)

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of target output name -> forecast DataFrame


        """

        batch_size = BATCH_SIZE

        results = {}

        # copy input_df to avoid modifying original
        input_df = input_df.copy()

        input_df = self._add_temporal_features(input_df)

        if input_df.index.freq is None:
            input_df = input_df.asfreq(f'{self.data_freq}min', method='ffill')

        covariate_cols = [k for k in self._features.keys() if k != "lagged_target"]
        has_covariates = len(covariate_cols) > 0

        # Clean data
        for col_name in covariate_cols + [t.output for t in self.targets]:
            if col_name in input_df.columns and input_df[col_name].isna().any():
                input_df[col_name] = input_df[col_name].ffill().bfill()

        # Load model once
        if not hasattr(self, 'model') or self.model is None:
            self._load_model()

        for target_cfg in self.targets:
            output = target_cfg.output
            target_series = input_df[output].values.astype(np.float32)

            # 1. VECTORIZED WINDOWING
            # Create all possible windows of context_len
            # sliding_window_view returns a view (no memory copy), extremely fast
            all_target_windows = sliding_window_view(target_series, window_shape=self.context_len)

            if has_covariates:
                # We need space for the horizon in the future for covariates
                num_windows = len(target_series) - self.context_len - self.horizon + 1
                targets = all_target_windows[:num_windows]
            else:
                num_windows = all_target_windows.shape[0]
                targets = all_target_windows

            if num_windows <= 0:
                raise ValueError("Not enough data for TimesFM context + horizon.")

            # 2. VECTORIZED COVARIATES
            dyn_covs = {}
            if has_covariates:
                for cov in covariate_cols:
                    if cov not in input_df.columns:
                        continue

                    # Covariate windows must span context_len (past) + horizon (future)
                    # to align with the target context windows. The window start from
                    # the feature config is ignored here — past_len is always context_len.
                    total_len = self.context_len + self.horizon

                    cov_series = input_df[cov].values.astype(np.float32)
                    # Create a sliding window for the covariate series
                    all_cov_windows = sliding_window_view(cov_series, window_shape=total_len)

                    # Align covariate windows with the target windows
                    # This replaces the nested loop 'for i in range(num_windows)'
                    dyn_covs[f"cov_{cov}"] = all_cov_windows[:num_windows]

            # 3. BATCHED INFERENCE
            # Running 1000s of windows at once crashes RAM; batching keeps it stable.

            all_forecasts = []
            inference_start = time.perf_counter()
            for i in range(0, num_windows, batch_size):
                b_targets = targets[i: i + batch_size]

                if has_covariates:
                    b_covs = {k: v[i: i + batch_size] for k, v in dyn_covs.items()}
                    # Pass as numpy array, avoid list() if possible
                    f, _ = self.model.forecast_with_covariates(
                        inputs=b_targets,
                        dynamic_numerical_covariates=b_covs,
                        xreg_mode="xreg + timesfm",
                        ridge=0.01,
                    )
                else:
                    f, _ = self.model.forecast(
                        horizon=self.horizon,
                        inputs=b_targets
                    )
                all_forecasts.append(np.asarray(f))
            inference_done = time.perf_counter()
            print(f"Model Inference time:  {inference_done - inference_start:.4f}s for target '{output}' with {num_windows} windows and batch size {batch_size}")
            # Combine batches and extract the forecast portion.
            # With return_backcast=True, model.forecast() returns (batch, context_len + horizon):
            # the backcast prefix comes first, so we take the last `horizon` columns.
            # forecast_with_covariates() already returns only the horizon, so [:horizon] is safe there too.
            raw = np.concatenate(all_forecasts, axis=0)
            forecasts = raw[:, -self.horizon:] if not has_covariates else raw[:, :self.horizon]

            # 4. DATA FRAME BUILDING
            start_time = input_df.index[self.context_len - 1] + pd.Timedelta(minutes=self.data_freq)

            # Determine index length
            idx_len = forecasts.shape[0] if not has_covariates else num_windows + self.horizon
            index = pd.date_range(start=start_time, periods=idx_len, freq=f'{self.data_freq}min')

            columns = [f'forecast_{i}' for i in range(self.lead_time, self.lead_time + self.horizon)]
            ftime_df = pd.DataFrame(index=index, columns=columns, dtype=float)

            # Fill forecast matrix using vectorized assignment where possible
            # Since this is a lead-time shifted matrix, we use a small loop
            for h in range(self.horizon):
                col_name = columns[h]
                start_idx = h + self.lead_time
                # Align the forecast column into the dataframe
                # This is significantly faster than iat in a double loop
                val_len = min(len(ftime_df) - start_idx, forecasts.shape[0])
                ftime_df.iloc[start_idx: start_idx + val_len, h] = forecasts[:val_len, h]

            if has_covariates:
                ftime_df = ftime_df.iloc[self.lead_time:]

            # Output mode
            if output_mode == "target":
                results[output] = ftime_df
            elif output_mode == "single":
                results[output] = pd.DataFrame({"forecast": ftime_df.max(axis=1)})
            else:
                raise ValueError(f"Invalid output_mode: {output_mode}")

        return results

    def fit(self, df):
        print("TimesFMForecaster does not require fitting. Model is pre-trained.")

    def save(self, filepath):
        """Save metadata only (model weights are pre-trained)."""
        joblib.dump({
            "class": "TimesFMForecaster",
            "targets": self.targets,
            "lead_time": self.lead_time,
            "forecast_len": self.horizon,
            "data_freq": self.data_freq,
            "features": self._features
        }, filepath)

    @classmethod
    def load(cls, filepath):
        state = joblib.load(filepath)
        obj = cls(
            targets=state["targets"],
            lead_time=state["lead_time"],
            forecast_len=state["forecast_len"],
            data_freq=state["data_freq"],
            features=state["features"]
        )
        obj._load_model()
        return obj

    def _add_temporal_features(self, input_df):
        """
        Adds temporal features (weekday, hour, month, holiday) to input_df if specified in self._features.
        Modifies input_df in-place.
        """
        #TODO replace with efp preocessing functions
        import holidays

        temporal_features = ["weekday", "hour", "month", "holiday"]
        for feat in temporal_features:
            if feat in self._features and feat not in input_df.columns:
                if feat == "weekday":
                    input_df["weekday"] = input_df.index.weekday
                elif feat == "hour":
                    input_df["hour"] = input_df.index.hour
                elif feat == "month":
                    input_df["month"] = input_df.index.month
                elif feat == "holiday":
                    fi_holidays = holidays.FI()
                    input_df["holiday"] = [int(x in fi_holidays) for x in input_df.index.date]
        return input_df
