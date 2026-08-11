"""
Baseline MLP model for time-series forecasting.

A simple multi-layer perceptron that takes the full feature window (history + future)
flattened, and predicts multi-step-ahead targets. Designed as a data-driven baseline
for comparison against physics-informed models such as TorchLKF.

Batch format expected from TorchDataLoader:
    features : (batch, window_size, n_features)
    targets  : (batch, fcast_len, n_targets)
"""

import torch
import torch.nn as nn
import lightning.pytorch as L
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from amp.torch.base import BaseTorchModel
from amp.base import BaseModel

import logging
logger = logging.getLogger(__name__)


class BaselineMLP(L.LightningModule, BaseTorchModel, BaseModel):
    """
    MLP baseline for time-series forecasting.

    Takes a flattened window of features (history + future horizon) and produces
    multi-step target predictions.  Intended as a pure data-driven comparison
    baseline against structured / physics-informed models like TorchLKF.

    Architecture
    ------------
    Input  : (batch, window_size * n_features)   [flattened]
    Hidden : configurable depth / width with ReLU + optional Dropout
    Output : (batch, n_targets, forecast_len)

    Parameters
    ----------
    targets : list of Target
        AMP Target objects defining the outputs.
    amp_features : dict
        AMP features dict ``{name: {'windows': [...], 'type': ...}}``.
        Determines input dimensions.
    target_features : list of str
        Column names to predict (must be a subset of the feature columns present
        in the data provided to the DataLoader).
    hidden_dims : list of int, optional
        Width of each hidden layer.  Default: ``[128, 64]``.
    dropout : float, optional
        Dropout probability applied after each hidden activation.  Default: 0.
    learning_rate : float, optional
        Adam learning rate.  Default: 1e-3.
    lead_time : int, optional
        Lead time in steps (passed through to the AMP evaluation pipeline).
    forecast_len : int
        Forecast horizon length in steps.
    data_freq : int
        Data resolution in minutes.
    norm_params : dict, optional
        ``{'mean': pd.Series, 'std': pd.Series}`` used to normalise inputs and
        denormalise outputs when ``normalize=True``.
    normalize : bool, optional
        Whether to normalise inputs before the forward pass.  Default: False.
    trainer_kwargs : dict, optional
        Keyword arguments forwarded to ``amp.torch.trainer.Trainer``.

    Examples
    --------
    >>> model = BaselineMLP(
    ...     targets=[Target('t_301_2'), Target('t_floor_heating_out')],
    ...     amp_features=feature_properties,
    ...     target_features=['t_301_2', 't_floor_heating_out'],
    ...     hidden_dims=[256, 128, 64],
    ...     forecast_len=24,
    ...     data_freq=60,
    ... )
    """

    def __init__(
        self,
        targets,
        amp_features: dict,
        target_features: List[str],
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        lead_time: int = 0,
        forecast_len: int = 24,
        data_freq: int = 60,
        norm_params: Optional[Dict] = None,
        normalize: bool = False,
        batch_size: Optional[int] = None,
        trainer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        # ── 1. initialise nn.Module first (required before any nn.* assignment) ──
        L.LightningModule.__init__(self)

        # ── 2. AMP base classes ────────────────────────────────────────────────
        BaseModel.__init__(self, targets=targets, features=amp_features)
        tb_name = kwargs.pop('tb_name', 'default')
        BaseTorchModel.__init__(self, trainer_kwargs=trainer_kwargs, tb_name=tb_name)

        # ── 3. store model attributes ─────────────────────────────────────────
        self.target_features = target_features
        self.normalize = normalize
        self.norm_params = norm_params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # self.outputs is provided by BaseModel as a property (reads self._outputs)

        # ── 4. derive dimensions from amp_features ────────────────────────────
        base_input_window = BaseModel.input_window.fget(self)
        history_len = abs(base_input_window[0])
        window_size = history_len + base_input_window[1] + 1
        n_features = len(amp_features)
        n_targets = len(target_features)

        input_size = window_size * n_features
        output_size = forecast_len * n_targets

        # ── 5. build MLP network ──────────────────────────────────────────────
        _hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]

        layers: List[nn.Module] = []
        in_size = input_size
        for h in _hidden_dims:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))

        self.net = nn.Sequential(*layers)

        # ── 6. register scalars as buffers for Lightning checkpoint safety ────
        self.register_buffer('_lead_time_buf', torch.tensor(lead_time))
        self.register_buffer('_forecast_len_buf', torch.tensor(forecast_len))
        self.register_buffer('_data_freq_buf', torch.tensor(data_freq))
        self.register_buffer('_input_window_buf', torch.tensor(list(base_input_window)))

    # ── property shims so BaseTorchModel.predict() resolves scalars correctly ──

    @property
    def lead_time(self) -> int:
        return int(self._lead_time_buf.item())

    @property
    def forecast_len(self) -> int:
        return int(self._forecast_len_buf.item())

    @property
    def data_freq(self) -> int:
        return int(self._data_freq_buf.item())

    @property
    def input_window(self):
        """Override BaseModel property to use registered buffer values."""
        return tuple(int(v) for v in self._input_window_buf)

    # ── nn.Module forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, window_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets, forecast_len)``.
        """
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)                      # (batch, window_size * n_features)
        out = self.net(x_flat)                             # (batch, forecast_len * n_targets)
        out = out.reshape(batch, self.forecast_len, len(self.target_features))
        return out.permute(0, 2, 1)                        # (batch, n_targets, forecast_len)

    # ── Lightning training/validation/test steps ───────────────────────────────

    def _compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss with optional horizon-window weighting and
        curriculum-based horizon truncation.

        Reads two optional attributes injected by CurriculumLearningCallback:
          - ``curriculum_steps`` (int): truncate both preds and targets to this
            many forecast steps before computing loss.
          - ``horizon_loss_windows`` / ``horizon_loss_weights``: list of (start, end)
            index tuples and corresponding scalar weights.  Gaps between windows are
            filled with weight 1.0.  If not set, plain MSE over the full horizon is used.

        Parameters
        ----------
        preds   : (batch, n_targets, fcast_len)
        targets : (batch, n_targets, fcast_len)
        """
        # 1. Horizon truncation (curriculum window_stages)
        steps = getattr(self, 'curriculum_steps', None)
        if steps is not None:
            preds = preds[..., :steps]
            targets = targets[..., :steps]

        total_steps = preds.shape[-1]

        # 2. Windowed weighted loss
        windows = getattr(self, 'horizon_loss_windows', None)
        weights = getattr(self, 'horizon_loss_weights', None)

        if not windows:
            return nn.functional.mse_loss(preds, targets)

        # Build explicit window list, clip to actual horizon
        explicit = []
        for (s, e), w in zip(windows, weights if weights else [1.0] * len(windows)):
            if e is None:
                e = total_steps
            s = min(s, total_steps)
            e = min(e, total_steps)
            if s < e:
                explicit.append((s, e, w))

        # Fill uncovered timesteps with weight 1.0
        covered = set()
        for s, e, _ in explicit:
            covered.update(range(s, e))
        gaps = sorted(set(range(total_steps)) - covered)
        if gaps:
            g_start = gaps[0]
            prev = gaps[0]
            for g in gaps[1:]:
                if g != prev + 1:
                    explicit.append((g_start, prev + 1, 1.0))
                    g_start = g
                prev = g
            explicit.append((g_start, prev + 1, 1.0))

        explicit.sort(key=lambda x: x[0])

        total_loss = torch.tensor(0.0, device=preds.device)
        weight_sum = sum(w for _, _, w in explicit)
        for s, e, w in explicit:
            total_loss = total_loss + w * nn.functional.mse_loss(preds[..., s:e], targets[..., s:e])

        return total_loss / weight_sum

    def training_step(self, batch, batch_idx):
        features, targets = batch
        # features : (batch, window_size, n_features)
        # targets  : (batch, fcast_len, n_targets)
        preds = self(features)                             # (batch, n_targets, fcast_len)
        targets_t = targets.permute(0, 2, 1)               # (batch, n_targets, fcast_len)
        loss = self._compute_loss(preds, targets_t)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss, 'predictions': preds.detach(), 'targets': targets_t.detach()}

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        preds = self(features)
        targets_t = targets.permute(0, 2, 1)
        loss = nn.functional.mse_loss(preds, targets_t)   # full horizon, no curriculum
        self.log('val_loss', loss, prog_bar=True)
        return {'loss': loss, 'predictions': preds.detach(), 'targets': targets_t.detach()}

    def test_step(self, batch, batch_idx):
        features, targets = batch
        preds = self(features)
        targets_t = targets.permute(0, 2, 1)
        loss = nn.functional.mse_loss(preds, targets_t)   # full horizon, no curriculum
        self.log('test_loss', loss, prog_bar=True)
        return {'loss': loss, 'predictions': preds.detach(), 'targets': targets_t.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # ── AMP interface: get_fit_data / fit / predict ────────────────────────────

    def _get_dataloaders(self, training_set, validation_set=None, testing_set=None):
        """
        Create plain (features, targets) PyTorch DataLoaders from lists of DataFrames.

        Called internally by ``BaseTorchModel.fit()``.
        """
        from amp.torch.dataloader import TorchDataLoader
        from torch.utils.data import DataLoader as TorchDataLoaderClass

        required_features = list(self._features.keys())

        dl = TorchDataLoader(
            resample_freq=self.data_freq,
            update_freq=self.data_freq,
            targets=self.target_features,
            normalize=True,
        )
        dl.set_split_data(training_set, validation_set, testing_set)
        # Save normalization parameters so _single_predict can normalize/denormalize
        if dl.is_fitted and dl.norm_params:
            self.norm_params = dl.norm_params
            self.normalize = True
        datasets = dl._create_datasets(
            features=required_features,
            targets=self.target_features,
            input_window=self.input_window,
        )

        loaders = []
        for split_name in ['training', 'validation', 'testing']:
            dataset = datasets.get(split_name)
            if dataset is not None:
                loader = TorchDataLoaderClass(
                    dataset,
                    batch_size=self.batch_size if self.batch_size is not None else len(dataset),
                    shuffle=False,
                    num_workers=0,
                    drop_last=True,
                )
            else:
                loader = None
            loaders.append(loader)

        return loaders

    def fit(self, training_set, validation_set=None, testing_set=None):
        """Delegate to BaseTorchModel.fit()."""
        BaseTorchModel.fit(self, training_set, validation_set, testing_set)
        return self

    def predict(self, df, output_mode='single', history_len=None, forecast_len=None, current_index=None):
        """Delegate to BaseTorchModel.predict()."""
        return BaseTorchModel.predict(
            self, df,
            output_mode=output_mode,
            history_len=history_len,
            forecast_len=forecast_len,
            current_index=current_index,
        )

    def _single_predict(self, df: pd.DataFrame, history_len: int, forecast_len: int, current_index: int) -> pd.DataFrame:
        """
        Predict a single forecast window.

        Parameters
        ----------
        df : pd.DataFrame
            Full input dataframe containing all feature columns.
        history_len : int
            Number of historical steps before ``current_index``.
        forecast_len : int
            Number of forecast steps after ``current_index``.
        current_index : int
            Row index where the forecast begins.

        Returns
        -------
        pd.DataFrame
            Predictions with ``target_features`` as columns and the forecast
            datetime index as the index.
        """
        self.eval()

        all_features = list(self._features.keys())
        window_df = df[all_features].iloc[current_index - history_len: current_index + forecast_len].copy()

        # Normalise if requested
        if self.normalize and self.norm_params is not None:
            for col in window_df.columns:
                if col in self.norm_params['mean'].index:
                    window_df[col] = (
                        (window_df[col] - self.norm_params['mean'][col])
                        / self.norm_params['std'][col]
                    )

        with torch.no_grad():
            x = torch.FloatTensor(window_df.fillna(0.0).values).unsqueeze(0)  # (1, window_size, n_features)
            preds = self(x)                              # (1, n_targets, forecast_len)
            preds_np = preds.squeeze(0).permute(1, 0).cpu().numpy()  # (forecast_len, n_targets)

        # Denormalise
        if self.normalize and self.norm_params is not None:
            for i, t in enumerate(self.target_features):
                if t in self.norm_params['mean'].index:
                    preds_np[:, i] = (
                        preds_np[:, i] * self.norm_params['std'][t]
                        + self.norm_params['mean'][t]
                    )

        forecast_index = df.index[current_index: current_index + forecast_len]
        return pd.DataFrame(preds_np, index=forecast_index, columns=self.target_features)

    def _target_predict(self, df: pd.DataFrame, history_len: int, forecast_len: int, current_index: int) -> list:
        """
        Slide a window across ``df`` and return one prediction per window.

        Parameters
        ----------
        df : pd.DataFrame
            Full input dataframe.
        history_len, forecast_len, current_index : int
            Window configuration (see ``_single_predict``).

        Returns
        -------
        list of pd.DataFrame
        """
        window_size = history_len + forecast_len
        predictions = []
        for start_idx in range(0, len(df) - window_size + 1):
            window_df = df.iloc[start_idx: start_idx + window_size]
            pred_df = self._single_predict(window_df, history_len, forecast_len, current_index)
            predictions.append(pred_df)
        return predictions
