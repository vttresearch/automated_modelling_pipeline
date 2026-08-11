"""
AMP-compatible TimesFM forecaster that loads a LoRA-fine-tuned adapter.

Subclasses the interface of ``TimesFMForecaster`` from the AMP library, using
the Transformers-based ``TimesFm2_5ModelForPrediction`` + PEFT LoRA adapter
instead of the native ``timesfm`` checkpoint.

The ``predict()`` method uses the Transformers inference API, which differs
slightly from the native ``timesfm`` package.

Model name convention in configs: ``timesfm_lora_ctx_{N}`` / ``timesfm_lora_ctx_{N}_control``
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BATCH_SIZE = 32  # smaller than base TimesFM due to Transformers overhead


class TimesFMLoRAForecaster:
    """
    TimesFM 2.5 forecaster that loads a pre-trained LoRA adapter.

    Identical public interface to ``TimesFMForecaster``:
        - ``fit(df)``           — no-op (base model is pre-trained, adapter is pre-loaded)
        - ``predict(input_df)`` — sliding-window forecast
        - ``save(filepath)``    — saves metadata only
        - ``load(filepath)``    — class method, reconstructs and loads adapter

    Parameters
    ----------
    targets : list[Target]
        AMP Target objects (same format as TimesFMForecaster).
    lead_time : int
        Forecast lead time in steps.
    forecast_len : int
        Forecast horizon in steps.
    data_freq : int
        Data frequency in minutes.
    features : dict
        Feature config dict (same format as TimesFMForecaster).
    update_freq : int
        Update frequency (unused, kept for interface compatibility).
    adapter_dir : str
        Path to the saved LoRA adapter directory (required).
    """

    def __init__(
        self,
        targets,
        lead_time: int,
        forecast_len: int,
        data_freq: int,
        features: dict | None = None,
        update_freq: int = 1,
        adapter_dir: str | None = None,
    ):
        if adapter_dir is None:
            raise ValueError(
                "adapter_dir is required for TimesFMLoRAForecaster. "
                "Pass the path to the directory containing the LoRA adapter weights."
            )
        self.targets = targets
        self.lead_time = lead_time
        self.horizon = forecast_len
        self.data_freq = data_freq
        self._features = features or {}
        self.context_len = abs(features["lagged_target"]["windows"][0][0])
        self.forecast_len = self.horizon
        self.update_rate = update_freq
        self.adapter_dir = adapter_dir
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Properties (mirror TimesFMForecaster interface)
    # ------------------------------------------------------------------

    @property
    def input_window(self):
        return -self.context_len, self.horizon

    @property
    def outputs(self):
        return [t.output for t in self.targets]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the Transformers TimesFM model with LoRA adapter applied."""
        import torch
        from transformers import TimesFm2_5ModelForPrediction
        from peft import PeftModel

        torch.set_float32_matmul_precision("high")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        if not os.path.isdir(self.adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {self.adapter_dir}\n"
                "Ensure the adapter has been trained and saved before running inference."
            )

        print(f"Loading base TimesFM model (transformers) …")
        base = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers"
        )

        print(f"Loading LoRA adapter from: {self.adapter_dir}")
        self._model = PeftModel.from_pretrained(base, self.adapter_dir)
        self._model.eval()

        # Merge LoRA weights into base for faster inference
        self._model = self._model.merge_and_unload()
        print(f"LoRA adapter merged. Model ready.")

    def _ensure_model(self):
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_df: pd.DataFrame, output_mode: str = "target") -> dict:
        """
        Generate sliding-window forecasts using the LoRA-fine-tuned TimesFM.

        Parameters
        ----------
        input_df : pd.DataFrame
            Input dataframe with target columns.
        output_mode : str
            ``"target"`` — multi-step forecast matrix per target.
            ``"single"`` — single forecast value per timestamp.

        Returns
        -------
        dict[str, pd.DataFrame]
        """
        import torch

        self._ensure_model()
        device = next(self._model.parameters()).device

        input_df = input_df.copy()
        if input_df.index.freq is None:
            input_df = input_df.asfreq(f"{self.data_freq}min", method="ffill")

        # Fill NaNs in all used columns
        all_cols = [t.output for t in self.targets]
        for col in all_cols:
            if col in input_df.columns and input_df[col].isna().any():
                input_df[col] = input_df[col].ffill().bfill()

        results = {}

        for target_cfg in self.targets:
            output = target_cfg.output
            series = input_df[output].values.astype(np.float32)

            # Build all context windows at once
            all_windows = sliding_window_view(series, window_shape=self.context_len)
            num_windows = all_windows.shape[0]

            if num_windows <= 0:
                raise ValueError(f"Not enough data for context length {self.context_len}.")

            all_forecasts = []
            inference_start = time.perf_counter()

            for i in range(0, num_windows, BATCH_SIZE):
                batch = all_windows[i : i + BATCH_SIZE]  # (B, context)
                batch_tensors = [
                    torch.from_numpy(batch[j]).to(device) for j in range(len(batch))
                ]

                with torch.no_grad():
                    out = self._model(
                        past_values=batch_tensors,
                        forecast_context_len=self.context_len,
                    )

                # point_forecasts: (batch, max_horizon) — slice to self.horizon
                preds = out.point_forecasts.cpu().numpy()[:, : self.horizon]
                all_forecasts.append(preds)

            inference_done = time.perf_counter()
            print(
                f"Model (LoRA) Inference time: {inference_done - inference_start:.2f}s "
                f"for target '{output}' with {num_windows} windows"
            )

            forecasts = np.concatenate(all_forecasts, axis=0)  # (num_windows, horizon)

            # Build output DataFrame (same structure as TimesFMForecaster)
            start_time = input_df.index[self.context_len - 1] + pd.Timedelta(
                minutes=self.data_freq
            )
            index = pd.date_range(
                start=start_time, periods=num_windows + self.horizon, freq=f"{self.data_freq}min"
            )
            columns = [
                f"forecast_{i}" for i in range(self.lead_time, self.lead_time + self.horizon)
            ]
            ftime_df = pd.DataFrame(index=index, columns=columns, dtype=float)

            for h in range(self.horizon):
                col_name = columns[h]
                start_idx = h + self.lead_time
                val_len = min(len(ftime_df) - start_idx, forecasts.shape[0])
                ftime_df.iloc[start_idx : start_idx + val_len, h] = forecasts[:val_len, h]

            if output_mode == "target":
                results[output] = ftime_df
            elif output_mode == "single":
                results[output] = pd.DataFrame({"forecast": ftime_df.max(axis=1)})
            else:
                raise ValueError(f"Invalid output_mode: {output_mode}")

        return results

    # ------------------------------------------------------------------
    # Fit / Save / Load (AMP interface)
    # ------------------------------------------------------------------

    def fit(self, df):
        print("TimesFMLoRAForecaster does not require fitting — adapter is pre-loaded.")

    def save(self, filepath):
        import joblib

        joblib.dump(
            {
                "class": "TimesFMLoRAForecaster",
                "targets": self.targets,
                "lead_time": self.lead_time,
                "forecast_len": self.horizon,
                "data_freq": self.data_freq,
                "features": self._features,
                "adapter_dir": self.adapter_dir,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        import joblib

        state = joblib.load(filepath)
        obj = cls(
            targets=state["targets"],
            lead_time=state["lead_time"],
            forecast_len=state["forecast_len"],
            data_freq=state["data_freq"],
            features=state["features"],
            adapter_dir=state.get("adapter_dir"),
        )
        obj._load_model()
        return obj
