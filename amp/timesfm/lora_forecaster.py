"""AMP TimesFM forecaster backed by LoRA-fine-tuned weights, with covariate support.

``TimesFMLoRAForecaster`` subclasses :class:`~amp.timesfm.forecaster.TimesFMForecaster`
and overrides only model loading. A LoRA adapter (trained on the Transformers
variant of TimesFM 2.5) is converted into the *native* checkpoint layout via
:mod:`amp.timesfm.lora_convert`, then loaded into the native ``timesfm`` model.

Because the native model is used for inference, the forecaster inherits the full
covariate pipeline of the base forecaster — including ``forecast_with_covariates``
(XReg) for weather / flex-event covariates — verbatim. This guarantees an
apples-to-apples comparison with base TimesFM: identical inference logic, the only
difference being the LoRA-fine-tuned backbone.

Model name convention in configs: ``timesfm_lora_ctx_{N}`` / ``timesfm_lora_ctx_{N}_control``
"""

from __future__ import annotations

import os

from amp.timesfm.forecaster import TimesFMForecaster
from amp.timesfm.lora_convert import (
    NATIVE_MODEL,
    load_native_finetuned_state_dict,
)


class TimesFMLoRAForecaster(TimesFMForecaster):
    """TimesFM forecaster that loads LoRA-fine-tuned weights with covariate support.

    Identical public interface and inference behaviour to ``TimesFMForecaster``;
    only the model weights differ (base backbone replaced by the LoRA-fine-tuned
    one). Covariates declared in ``features`` are handled exactly as in the base
    forecaster (XReg residual model).

    Parameters
    ----------
    targets, lead_time, forecast_len, data_freq, features, update_freq
        Same as :class:`~amp.timesfm.forecaster.TimesFMForecaster`.
    adapter_dir : str
        Path to the saved LoRA adapter directory (required). If a cached
        ``native_finetuned.safetensors`` is present it is loaded directly;
        otherwise the native weights are built on the fly from the PEFT adapter.
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
        super().__init__(
            targets=targets,
            lead_time=lead_time,
            forecast_len=forecast_len,
            data_freq=data_freq,
            features=features,
            update_freq=update_freq,
        )
        self.adapter_dir = adapter_dir

    def _load_model(self):
        """Load the native TimesFM model and apply LoRA-fine-tuned weights.

        Mirrors ``TimesFMForecaster._load_model`` (same ``ForecastConfig``) but
        replaces the attention projections with the fine-tuned ones before
        compiling, so the covariate inference path is reused unchanged.
        """
        self.model_name = NATIVE_MODEL
        try:
            import timesfm
            import torch
        except ImportError:
            import warnings
            warnings.warn("TimesFM not installed. Please install timesfm.")
            raise

        torch.set_float32_matmul_precision("high")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        if not os.path.isdir(self.adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {self.adapter_dir}\n"
                "Ensure the adapter has been trained and saved before running inference."
            )

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)

        print(f"Applying LoRA-fine-tuned weights from: {self.adapter_dir}")
        native_sd = load_native_finetuned_state_dict(self.adapter_dir)
        missing, unexpected = self.model.model.load_state_dict(native_sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading fine-tuned weights: {unexpected[:5]} …"
            )
        if missing:
            # All native keys should be present; warn rather than fail in case the
            # cached export omitted unchanged buffers.
            import warnings
            warnings.warn(
                f"{len(missing)} native keys not provided by fine-tuned weights "
                f"(kept base values), e.g. {missing[:3]}"
            )

        print(
            f"Loaded LoRA-fine-tuned TimesFM ({self.model_name}) with context "
            f"length {self.context_len} and horizon {self.horizon}"
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=True,
            )
        )

    # ------------------------------------------------------------------
    # Save / Load (AMP interface) — persist the adapter directory.
    # ------------------------------------------------------------------

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
        # Model loading is lazy — _load_model() is called on first predict()
        # (inherited from TimesFMForecaster). This avoids requiring the adapter
        # directory / weights to exist at load time.
        return obj
