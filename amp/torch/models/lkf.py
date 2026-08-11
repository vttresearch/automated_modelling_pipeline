"""
Unified Linear Kalman Filter (LKF) with trainable parameters.

Single model supporting:
    x[t+1] = A @ x[t] + B @ u[t] + E @ d[t] + b_x + w[t]   (state dynamics)
    y[t] = C @ x[t] + b_y + v[t]                           (observation model)

where:
    x[t] - latent state vector
    u[t] - control inputs (controllable)
    d[t] - disturbance inputs (uncontrollable, optional)
    y[t] - observations/outputs
    w[t] ~ N(0, Q) - process noise
    v[t] ~ N(0, R) - measurement noise
    
    A - state transition matrix (trainable)
    B - control influence matrix (trainable)
    E - disturbance influence matrix (trainable, optional)
    C - observation matrix (trainable)
    b_x - state bias vector (trainable, optional)
    b_y - observation bias vector (trainable, optional)
"""

from abc import abstractmethod
import time
import warnings
import torch
import torch.nn as nn
import lightning.pytorch as L
from typing import Optional, List


class LinearKalmanFilter(nn.Module):
    """
    Linear Kalman Filter for state estimation.
    
    State space model:
        x[t+1] = A @ x[t] + B @ u[t] + w[t]    (process model)
        y[t] = C @ x[t] + v[t]                  (observation model)
    
    where:
        w[t] ~ N(0, Q) - process noise
        v[t] ~ N(0, R) - measurement noise
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        process_noise_cov: float = 1e-3,
        measurement_noise_cov: float = 1e-2,
        learnable_noise: bool = False,
        noise_constraint_type: str = 'diagonal_positive',
    ):
        """
        Args:
            state_dim: Dimension of latent state
            input_dim: Dimension of control inputs
            output_dim: Dimension of observations
            process_noise_cov: Initial process noise covariance (Q)
            measurement_noise_cov: Initial measurement noise covariance (R)
            learnable_noise: If True, Q and R are learnable parameters
            noise_constraint_type: Type of constraint for learnable noise covariances.
                         Options: 'cholesky', 'diagonal_positive', 'none'
                         - 'cholesky': Full covariance via Cholesky decomposition (allows correlations)
                         - 'diagonal_positive': Diagonal matrices only (no correlations, forces independence)
                         - 'none': No constraint (use with caution, may not stay positive definite)
                         Default: 'diagonal_positive' for robustness
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Validate constraint type
        valid_types = ['cholesky', 'diagonal_positive', 'none']
        if noise_constraint_type not in valid_types:
            raise ValueError(f"noise_constraint_type must be one of {valid_types}, got '{noise_constraint_type}'")
        
        self.noise_constraint_type = noise_constraint_type
        
        self.noise_constraint_type = noise_constraint_type
        
        # Process noise covariance Q
        if learnable_noise:
            if noise_constraint_type == 'cholesky':
                # Parameterize Q = L @ L.T where L is lower triangular
                # Initialize L as Cholesky of initial Q
                L_Q_init = torch.linalg.cholesky(torch.eye(state_dim) * process_noise_cov)
                self.L_Q = nn.Parameter(L_Q_init.clone())
            elif noise_constraint_type == 'diagonal_positive':
                # Use log-scale diagonal for better optimization (ensures positivity)
                self.log_Q_diag = nn.Parameter(
                    torch.log(torch.ones(state_dim) * process_noise_cov)
                )
            else:  # 'none'
                # Direct parameterization - WARNING: may not stay positive definite
                self.Q_param = nn.Parameter(torch.eye(state_dim) * process_noise_cov)
        else:
            self.register_buffer(
                'Q', torch.eye(state_dim) * process_noise_cov
            )
        
        # Measurement noise covariance R
        if learnable_noise:
            if noise_constraint_type == 'cholesky':
                # Parameterize R = L @ L.T where L is lower triangular
                L_R_init = torch.linalg.cholesky(torch.eye(output_dim) * measurement_noise_cov)
                self.L_R = nn.Parameter(L_R_init.clone())
            elif noise_constraint_type == 'diagonal_positive':
                self.log_R_diag = nn.Parameter(
                    torch.log(torch.ones(output_dim) * measurement_noise_cov)
                )
            else:  # 'none'
                # Direct parameterization - WARNING: may not stay positive definite
                self.R_param = nn.Parameter(torch.eye(output_dim) * measurement_noise_cov)
        else:
            self.register_buffer(
                'R', torch.eye(output_dim) * measurement_noise_cov
            )
        
        # Initial state covariance P0
        self.learnable_noise = learnable_noise
        if learnable_noise:
            # Use log-scale diagonal for P0 (ensures positivity)
            # Initialize to log(1.0) = 0 for each dimension
            self.log_P0_diag = nn.Parameter(
                torch.zeros(state_dim)
            )
        else:
            # Fixed P0 as identity matrix
            self.register_buffer(
                'P0_diag', torch.ones(state_dim)
            )
        
        self.learnable_noise = learnable_noise
    
    @property
    def Q(self):
        """Process noise covariance."""
        if self.learnable_noise:
            if self.noise_constraint_type == 'cholesky':
                # Q = L @ L.T ensures positive definiteness
                # Use tril to enforce lower triangular structure
                L = torch.tril(self.L_Q)
                return L @ L.T
            elif self.noise_constraint_type == 'diagonal_positive':
                return torch.diag(torch.exp(self.log_Q_diag))
            else:  # 'none'
                return self.Q_param
        else:
            return self._buffers['Q']
    
    @property
    def R(self):
        """Measurement noise covariance."""
        if self.learnable_noise:
            if self.noise_constraint_type == 'cholesky':
                # R = L @ L.T ensures positive definiteness
                L = torch.tril(self.L_R)
                return L @ L.T
            elif self.noise_constraint_type == 'diagonal_positive':
                return torch.diag(torch.exp(self.log_R_diag))
            else:  # 'none'
                return self.R_param
        else:
            return self._buffers['R']
    
    @property
    def P0(self):
        """Initial state covariance matrix."""
        if self.learnable_noise:
            # Diagonal covariance with learned variances
            return torch.diag(torch.exp(self.log_P0_diag))
        else:
            # Fixed identity covariance
            return torch.diag(self.P0_diag)
    
    def predict(self, mean, cov, A, B, u=None, A_expanded=None):
        """
        Prediction step: propagate state through dynamics.
        
        Args:
            mean: State mean (batch_size, state_dim)
            cov: State covariance (batch_size, state_dim, state_dim)
            A: State transition matrix (state_dim, state_dim)
            B: Control matrix (state_dim, input_dim) or None
            u: Control input (batch_size, input_dim) or None
            A_expanded: Pre-expanded A matrix (batch_size, state_dim, state_dim) - optional for optimization
            
        Returns:
            mean_pred: Predicted state mean
            cov_pred: Predicted state covariance
        """
        # Mean prediction: x_pred = A @ x
        mean_pred = mean @ A.T  # (batch, state_dim)
        
        # Add control if provided
        if u is not None and B is not None:
            mean_pred = mean_pred + u @ B.T
        
        # Covariance prediction: P_pred = A @ P @ A.T + Q
        # Use pre-expanded A if provided, otherwise expand here
        if A_expanded is None:
            batch_size = mean.shape[0]
            A_expanded = A.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, state_dim, state_dim)
        
        cov_pred = A_expanded @ cov @ A_expanded.transpose(1, 2)  # (batch, state_dim, state_dim)
        
        # Add process noise
        cov_pred = cov_pred + self.Q.unsqueeze(0)  # Broadcasting handles batch dimension
        
        return mean_pred, cov_pred
    
    def update(self, mean_pred, cov_pred, measurement, C, 
               C_expanded=None, R_expanded=None, I_expanded=None):
        """
        Update step: incorporate measurement.
        
        Args:
            mean_pred: Predicted state mean (batch_size, state_dim)
            cov_pred: Predicted state covariance (batch_size, state_dim, state_dim)
            measurement: Measurement y (batch_size, output_dim)
            C: Observation matrix (output_dim, state_dim)
            C_expanded: Pre-expanded C matrix (batch_size, output_dim, state_dim) - optional for optimization
            R_expanded: Pre-expanded R matrix (batch_size, output_dim, output_dim) - optional for optimization
            I_expanded: Pre-expanded identity matrix (batch_size, state_dim, state_dim) - optional for optimization
            
        Returns:
            mean_updated: Updated state mean
            cov_updated: Updated state covariance
        """
        batch_size = mean_pred.shape[0]
        
        # Predicted measurement: y_pred = C @ x_pred
        y_pred = mean_pred @ C.T  # (batch, output_dim)
        
        # Innovation: y - y_pred
        innovation = measurement - y_pred  # (batch, output_dim)
        
        # Innovation covariance: S = C @ P_pred @ C.T + R
        # Use pre-expanded matrices if provided
        if C_expanded is None:
            C_expanded = C.unsqueeze(0).expand(batch_size, -1, -1)
        if R_expanded is None:
            R_expanded = self.R.unsqueeze(0).expand(batch_size, -1, -1)
        
        S = C_expanded @ cov_pred @ C_expanded.transpose(1, 2) + R_expanded
        
        # Kalman gain: K = P_pred @ C.T @ S^-1
        # Use solve instead of inverse for numerical stability and speed
        PC_T = cov_pred @ C_expanded.transpose(1, 2)  # (batch, state_dim, output_dim)
        
        if S.device.type == 'mps':
            # MPS doesn't support linalg.solve, use inverse as fallback
            K = PC_T @ torch.inverse(S)  # (batch, state_dim, output_dim)
        else:
            # Use solve for better numerical stability on CPU/CUDA
            K = torch.linalg.solve(S.transpose(1, 2), PC_T.transpose(1, 2)).transpose(1, 2)
        
        # State update: x = x_pred + K @ innovation
        mean_updated = mean_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        
        # Covariance update: P = (I - K @ C) @ P_pred
        if I_expanded is None:
            I_expanded = torch.eye(self.state_dim, device=mean_pred.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        cov_updated = (I_expanded - K @ C_expanded) @ cov_pred
        
        return mean_updated, cov_updated


class BatchedLinearKalmanFilter(LinearKalmanFilter):
    """
    Linear Kalman Filter that handles batched observation matrices.
    
    Extends LinearKalmanFilter to support time-varying (batched) C matrices
    for models like PWLKF where the observation matrix changes at each timestep.
    
    Key difference: The update() method can handle C matrices that are already
    batched (3D tensors) instead of requiring 2D matrices that get expanded.
    """
    
    def update(self, mean_pred, cov_pred, measurement, C, 
               C_expanded=None, R_expanded=None, I_expanded=None):
        """
        Update step: incorporate measurement with support for batched C matrices.
        
        Args:
            mean_pred: Predicted state mean (batch_size, state_dim)
            cov_pred: Predicted state covariance (batch_size, state_dim, state_dim)
            measurement: Measurement y (batch_size, output_dim)
            C: Observation matrix - can be either:
                - 2D: (output_dim, state_dim) - will be expanded
                - 3D: (batch_size, output_dim, state_dim) - already batched
            C_expanded: Pre-expanded C matrix (batch_size, output_dim, state_dim) - optional
            R_expanded: Pre-expanded R matrix (batch_size, output_dim, output_dim) - optional
            I_expanded: Pre-expanded identity matrix (batch_size, state_dim, state_dim) - optional
            
        Returns:
            mean_updated: Updated state mean
            cov_updated: Updated state covariance
        """
        batch_size = mean_pred.shape[0]
        
        # Handle batched or non-batched C matrix
        if C.ndim == 3:
            # C is already batched (batch, output_dim, state_dim)
            C_batched = C
            # Predicted measurement: y_pred = C @ x_pred (batch matmul)
            y_pred = (C_batched @ mean_pred.unsqueeze(-1)).squeeze(-1)
        else:
            # C is 2D (output_dim, state_dim) - use standard approach
            y_pred = mean_pred @ C.mT  # (batch, output_dim)
            C_batched = C.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Innovation: y - y_pred
        innovation = measurement - y_pred  # (batch, output_dim)
        
        # Innovation covariance: S = C @ P_pred @ C.T + R
        # Use pre-expanded matrices if provided, otherwise use C_batched
        if C_expanded is None:
            C_expanded = C_batched
        if R_expanded is None:
            R_expanded = self.R.unsqueeze(0).expand(batch_size, -1, -1)
        
        S = C_expanded @ cov_pred @ C_expanded.transpose(1, 2) + R_expanded
        
        # Kalman gain: K = P_pred @ C.T @ S^-1
        # Use solve instead of inverse for numerical stability and speed
        PC_T = cov_pred @ C_expanded.transpose(1, 2)  # (batch, state_dim, output_dim)
        
        if S.device.type == 'mps':
            # MPS doesn't support linalg.solve, use inverse as fallback
            K = PC_T @ torch.inverse(S)  # (batch, state_dim, output_dim)
        else:
            # Use solve for better numerical stability on CPU/CUDA
            K = torch.linalg.solve(S.transpose(1, 2), PC_T.transpose(1, 2)).transpose(1, 2)
        
        # State update: x = x_pred + K @ innovation
        mean_updated = mean_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
        
        # Covariance update: P = (I - K @ C) @ P_pred
        if I_expanded is None:
            I_expanded = torch.eye(self.state_dim, device=mean_pred.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        cov_updated = (I_expanded - K @ C_expanded) @ cov_pred
        
        return mean_updated, cov_updated
    
class AbstractLinearDynamics:
    """ Abstract base class for linear dynamics models with Kalman filter support. 
    Contains abstract methods that must be implemented by subclasses.
    
    Note: All methods accept **kwargs to allow model-specific extensions
    (e.g., matrix_generation_features for PWLKF) without breaking the interface.
    """

    @abstractmethod
    def encode_with_kf(self, control_inputs, measurements, disturbance_inputs=None, **kwargs):
        """
        Encode initial state using Kalman filter with measurements.
        
        Args:
            control_inputs: Historical control inputs (batch, control_dim, window)
            measurements: Historical observations (batch, output_dim, window)
            disturbance_inputs: Historical disturbances (batch, disturbance_dim, window) - optional
            **kwargs: Model-specific parameters (e.g., matrix_generation_features)

        Returns:
            x_mean: Estimated initial state mean (batch, latent_dim)
            x_cov: Estimated initial state covariance (batch, latent_dim, latent_dim)
        """

    @abstractmethod
    def predict_with_kf(self, x_mean, x_cov, control_inputs, disturbance_inputs=None, **kwargs):
        """
        Predict future states using Kalman filter.
        
        Args:
            x_mean: Initial state mean (batch, latent_dim)
            x_cov: Initial state covariance (batch, latent_dim, latent_dim)
            control_inputs: Control inputs (batch, control_dim, time_steps)
            disturbance_inputs: Disturbance inputs (batch, disturbance_dim, time_steps) - optional
            **kwargs: Model-specific parameters (e.g., matrix_generation_features)
        
        Returns:
            predictions: Predicted outputs (batch, output_dim, time_steps)
            uncertainties: Prediction uncertainties (batch, output_dim, time_steps)
        """

    @abstractmethod
    def forward(self, control_inputs, historical_controls=None, historical_observations=None, 
                disturbance_inputs=None, historical_disturbances=None, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            control_inputs: Control inputs for prediction (batch, control_dim, time_steps)
            historical_controls: Historical controls (batch, control_dim, window) - optional
            historical_observations: Historical observations (batch, output_dim, window) - optional
            disturbance_inputs: Disturbance inputs (batch, disturbance_dim, time_steps) - optional
            historical_disturbances: Historical disturbances (batch, disturbance_dim, window) - optional
            **kwargs: Model-specific parameters (e.g., matrix_generation_features)

        Returns:
            predictions: Predicted outputs (batch, output_dim, time_steps)
            uncertainties: Prediction uncertainties (batch, output_dim, time_steps)
        """

    @abstractmethod
    def predict_full_trajectory(self, control_inputs, historical_controls=None, historical_observations=None,
                disturbance_inputs=None, historical_disturbances=None, **kwargs):
        """
        Predict full trajectory of states and covariances.
        
        Args:
            control_inputs: Control inputs for prediction (batch, control_dim, time_steps)
            historical_controls: Historical controls (batch, control_dim, window) - optional
            historical_observations: Historical observations (batch, output_dim, window) - optional
            disturbance_inputs: Disturbance inputs (batch, disturbance_dim, time_steps) - optional
            historical_disturbances: Historical disturbances (batch, disturbance_dim, window) - optional
            **kwargs: Model-specific parameters (e.g., matrix_generation_features)

        Returns:
            predictions_full: Predicted outputs (batch, output_dim, window + time_steps)
            uncertainties_full: Prediction uncertainties (batch, output_dim, window + time_steps)
            window_size: Number of timesteps in historical window
        """


class TorchLinearDynamicsWithKF(L.LightningModule, AbstractLinearDynamics):
    """
    Unified Trainable Linear Kalman Filter supporting:
        x[t+1] = A @ x[t] + B @ u[t] + E @ d[t] + b_x + w[t]
        y[t] = C @ x[t] + b_y + v[t]
    
    Key features:
    - Single model for both control-only and control+disturbance scenarios
    - All matrices (A, B, C, E) are nn.Parameter (trainable tensors)
    - Optional bias terms (b_x, b_y) can be enabled independently
    - Uses KF update during encoding (with measurements)
    - Uses prediction-only during forward simulation
    - Learns noise covariances Q and R (optional)
    """
    
    def __init__(
        self,
        control_features: list,
        disturbance_features: list = None,
        latent_dim: int = 3,
        observation_features: list = None,
        target_features: list = None,
        output_dim: int = 1,
        process_noise_cov: float = 1e-3,
        measurement_noise_cov: float = 1e-2,
        learnable_noise: bool = True,
        noise_constraint_type: str = 'diagonal_positive',
        learning_rate: float = 1e-3,
        num_encoding_measurements: int = 1,
        n_substeps: int = 1,
        use_state_bias: bool = True,
        use_observation_bias: bool = True,
        use_feedforward: bool = False,
        learnable_x0: bool = False,
        horizon_loss_windows: list = None,
        horizon_loss_weights: list = None,
        n4sid_init: bool = False,
        n4sid_num_block_rows: int = 10,
        ss_init: str = None,
        ss_init_kwargs: dict = None,
    ):
        """
        Args:
            control_features: List of control input feature names (controllable)
            disturbance_features: List of disturbance feature names (uncontrollable, optional)
            latent_dim: Dimension of latent state
            observation_features: List of measurement feature names (what C matrix observes).
                                 If None, uses target_features. Can be superset of targets.
            target_features: List of target feature names (what we predict/evaluate).
                            If None, uses output_dim.
            output_dim: Dimension of observations/targets (deprecated - use target_features)
            process_noise_cov: Initial process noise covariance
            measurement_noise_cov: Initial measurement noise covariance
            learnable_noise: If True, learn Q and R during training
            noise_constraint_type: Type of constraint for learnable noise covariances.
                                  Options: 'cholesky', 'diagonal_positive', 'none'
                                  - 'cholesky': Full covariance via Cholesky decomposition (allows correlations)
                                  - 'diagonal_positive': Diagonal matrices only (no correlations)
                                  - 'none': No constraint
                                  Default: 'diagonal_positive' for robustness
            learning_rate: Learning rate for optimizer
            num_encoding_measurements: Number of recent measurements to use for encoding (1=last only, -1=all)
            n_substeps: Number of substeps per data interval (default=1). Uses zero-order hold on inputs.
                       Model learns dynamics at substep time scale.
            use_state_bias: If True, add learnable bias term b_x to state dynamics
            use_observation_bias: If True, add learnable bias term b_y to observations
            use_feedforward: If True, add learnable feedforward matrix D so that
                             y[t] = C @ x[t] + D @ u[t] + b_y (default: False)
            learnable_x0: If True, learn initial state mean x0 as a parameter (default: False, uses observation mean)
            horizon_loss_windows: List of (start, end) tuples for multi-horizon loss splits.
                                e.g., [(0, 4), (4, 8), (8, 12)] for 12-step forecast with 3 horizons.
                                If None, single loss over entire horizon is used: [(0, None)].
            horizon_loss_weights: Weights for each horizon window, e.g., [2.0, 1.5, 1.0].
                                Must have same length as horizon_loss_windows. 
                                If None, all weights default to 1.0 for each window.
            n4sid_init: If True, run N4SID subspace identification on training data before
                        gradient training to initialize A, B, C (and E, D if applicable).
                        Requires the `nfoursid` package to be installed. (default: False)
                        Deprecated — use ss_init='nfoursid' instead.
            n4sid_num_block_rows: Number of block rows for the N4SID Hankel matrix.
                                  Larger values improve accuracy but increase memory and
                                  compute cost. Only used when n4sid_init=True. (default: 10)
                                  Deprecated — pass via ss_init_kwargs={'num_block_rows': N}.
            ss_init: Subspace identification method to use for matrix initialization before
                     gradient training. Options: 'nfoursid', 'sippy', or None (default: None).
            ss_init_kwargs: Keyword arguments forwarded to the subspace init method.
                            For 'nfoursid': accepts 'num_block_rows' (default: n4sid_num_block_rows).
                            For 'sippy': accepts 'id_method', 'ss_f', 'ss_fixed_order',
                            'ss_threshold', 'centering'.
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Multi-horizon loss configuration
        self.horizon_loss_windows = horizon_loss_windows or [(0, None)]  # Default: single window over full horizon
        if horizon_loss_weights is None:
            self.horizon_loss_weights = [1.0] * len(self.horizon_loss_windows)
        else:
            self.horizon_loss_weights = horizon_loss_weights
            if len(self.horizon_loss_weights) != len(self.horizon_loss_windows):
                raise ValueError(f"horizon_loss_weights length ({len(horizon_loss_weights)}) must match "
                               f"horizon_loss_windows length ({len(self.horizon_loss_windows)})")
        
        # Store feature lists
        self.control_features = control_features
        self.control_dim = len(control_features)
        self.num_encoding_measurements = num_encoding_measurements
        self.n_substeps = n_substeps
        self.use_state_bias = use_state_bias
        self.use_observation_bias = use_observation_bias
        self.learnable_x0 = learnable_x0
        
        # Subspace identification initialization configuration
        # Backward-compat: n4sid_init=True → ss_init='nfoursid'
        if n4sid_init and ss_init is None:
            import warnings as _warnings
            _warnings.warn(
                "n4sid_init=True is deprecated; use ss_init='nfoursid' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            ss_init = 'nfoursid'
        if ss_init == 'nfoursid' and ss_init_kwargs is None:
            ss_init_kwargs = {'num_block_rows': n4sid_num_block_rows}
        self.ss_init = ss_init
        self.ss_init_kwargs = ss_init_kwargs or {}
        # Keep for checkpoint compatibility
        self.n4sid_init = n4sid_init
        self.n4sid_num_block_rows = n4sid_num_block_rows
        
        # Handle disturbance features (optional)
        self.disturbance_features = disturbance_features if disturbance_features else []
        self.disturbance_dim = len(self.disturbance_features)
        self.has_disturbances = self.disturbance_dim > 0
        
        # Handle observation_features: what C matrix observes (for state estimation)
        if observation_features is not None:
            self.observation_features = observation_features if isinstance(observation_features, list) else [observation_features]
            measurement_dim = len(self.observation_features)
        else:
            self.observation_features = None
            measurement_dim = output_dim
        
        # Handle target_features: what we predict/evaluate (can be subset of measurements)
        if target_features is not None:
            self.target_features = target_features if isinstance(target_features, list) else [target_features]
            output_dim = len(self.target_features)
        else:
            self.target_features = None
        
        # Store dimensions
        self.latent_dim = latent_dim
        self.measurement_dim = measurement_dim  # C matrix output dimension
        self.output_dim = output_dim            # Prediction output dimension
        self.lr = learning_rate
        
        # Define trainable matrices as nn.Parameter
        # A: state transition (latent_dim, latent_dim)
        self.A = nn.Parameter(torch.eye(latent_dim) + torch.randn(latent_dim, latent_dim) * 0.01)
        
        # B: control influence (latent_dim, control_dim)
        self._B = nn.Parameter(torch.randn(latent_dim, self.control_dim) * 0.1)
        
        # C: observation matrix (measurement_dim, latent_dim) - observes all measurements
        self._C = nn.Parameter(torch.randn(measurement_dim, latent_dim) * 0.1)
        
        # If targets are subset of measurements, create projection matrix
        # to select target outputs from full C @ x
        if self.observation_features and self.target_features:
            # Create indices to map targets within measurements
            self.target_indices = torch.tensor([
                self.observation_features.index(t) for t in self.target_features
            ])
        else:
            self.target_indices = None
        
        # E: disturbance influence (latent_dim, disturbance_dim) - only if disturbances provided
        if self.has_disturbances:
            self._E = nn.Parameter(torch.randn(latent_dim, self.disturbance_dim) * 0.1)
        else:
            self._E = None
        
        # b_x: state bias vector (latent_dim,) - optional
        if use_state_bias:
            self.b_x = nn.Parameter(torch.zeros(latent_dim))
        else:
            self.register_buffer('b_x', torch.zeros(latent_dim))
        
        # b_y: observation bias vector (measurement_dim,) - optional
        if use_observation_bias:
            self.b_y = nn.Parameter(torch.zeros(measurement_dim))
        else:
            self.register_buffer('b_y', torch.zeros(measurement_dim))
        
        # D: control feedforward matrix (measurement_dim, control_dim) - optional
        # y[t] = C @ x[t] + D @ u[t] + b_y
        self.use_feedforward = use_feedforward
        if use_feedforward:
            self._D = nn.Parameter(torch.randn(measurement_dim, self.control_dim) * 0.1)
        else:
            self._D = None
        
        # x0: initial state mean (latent_dim,) - optional learnable parameter
        if learnable_x0:
            # Initialize to zeros - will be set to observation mean on first forward pass
            self.x0 = nn.Parameter(torch.zeros(latent_dim))
            self._x0_initialized = False  # Flag to track if x0 has been initialized with data
        else:
            self.x0 = None
        
        # Create Kalman filter
        # Total input dim is control_dim (B matrix only, E handled separately)
        self.kf = LinearKalmanFilter(
            state_dim=latent_dim,
            input_dim=self.control_dim,  # Only for B matrix operations
            output_dim=measurement_dim,  # C matrix output dimension (all measurements)
            process_noise_cov=process_noise_cov,
            measurement_noise_cov=measurement_noise_cov,
            learnable_noise=learnable_noise,
            noise_constraint_type=noise_constraint_type,
        )
        
        # Curriculum learning support - will be set by CurriculumLearningCallback
        self.curriculum_fraction = 1.0  # Default to full difficulty (100% of horizon) - for linear strategy
        self.curriculum_steps = None  # Absolute step count - for window_stages strategy
        
        # Timing tracking
        self.timing_stats = {
            'encode_with_kf': {'total': 0.0, 'count': 0},
            'predict_with_kf': {'total': 0.0, 'count': 0},
        }
    
    @property
    def B(self):
        """Control influence matrix."""
        return self._B
    
    @property
    def C(self):
        """Observation matrix."""
        return self._C
    
    @property
    def E(self):
        """Disturbance influence matrix."""
        return self._E
    
    @property
    def D(self):
        """Control feedforward matrix (observation equation)."""
        return self._D
    
    def _compute_state_mean(self, x_mean, u_t, A, B, d_t=None, E=None):
        """
        Compute predicted state mean.
        
        x[t+1] = A @ x[t] + B @ u[t] + E @ d[t] + b_x
        
        Args:
            x_mean: Current state mean (batch, latent_dim)
            u_t: Control input at time t (batch, control_dim)
            A: State transition matrix (latent_dim, latent_dim) or (batch, latent_dim, latent_dim)
            B: Control matrix (latent_dim, control_dim) or (batch, latent_dim, control_dim)
            d_t: Disturbance input at time t (batch, disturbance_dim) - optional
            E: Disturbance matrix (latent_dim, disturbance_dim) or (batch, latent_dim, disturbance_dim) - optional
            
        Returns:
            x_mean_pred: Predicted state mean (batch, latent_dim)
        """
        # State transition: x_mean @ A.T
        x_mean_pred = x_mean @ A.T
        
        # Control influence: u @ B.T
        x_mean_pred = x_mean_pred + u_t @ B.T
        
        # Disturbance influence: d @ E.T (if provided)
        if d_t is not None and E is not None:
            x_mean_pred = x_mean_pred + d_t @ E.T
        
        # State bias (broadcasting handles batch dimension automatically)
        x_mean_pred = x_mean_pred + self.b_x
        
        return x_mean_pred
    
    def _compute_state_covariance(self, x_cov, A_expanded):
        """
        Compute predicted state covariance.
        
        P[t+1] = A @ P[t] @ A.T + Q
        
        Args:
            x_cov: Current state covariance (batch, latent_dim, latent_dim)
            A_expanded: State transition matrix (batch, latent_dim, latent_dim)
            
        Returns:
            x_cov_pred: Predicted state covariance (batch, latent_dim, latent_dim)
        """
        x_cov_pred = A_expanded @ x_cov @ A_expanded.transpose(1, 2) + self.Q.unsqueeze(0)
        return x_cov_pred
    
    def _compute_observation(self, x_mean, project_to_targets=False, u_t=None):
        """
        Compute observation from state.
        
        y = C @ x + b_y                    (no feedforward)
        y = C @ x + D @ u + b_y            (with feedforward)
        
        Args:
            x_mean: State mean (batch, latent_dim)
            project_to_targets: If True and target_indices exist, project to target subset
            u_t: Control input (batch, control_dim) - required when use_feedforward=True
            
        Returns:
            y_pred: Predicted observation (batch, measurement_dim) or (batch, output_dim)
        """
        y_pred = x_mean @ self.C.T  # (batch, measurement_dim)
        
        # Add feedforward term D @ u if enabled
        if self.D is not None and u_t is not None:
            y_pred = y_pred + u_t @ self.D.T  # (batch, measurement_dim)
        
        # Add observation bias (broadcasting handles batch dimension automatically)
        y_pred = y_pred + self.b_y
        
        # Project to targets if requested and mapping exists
        if project_to_targets and self.target_indices is not None:
            y_pred = y_pred[:, self.target_indices]  # (batch, output_dim)
        
        return y_pred
    
    def _compute_observation_uncertainty(self, x_cov, C_expanded, R_diag, project_to_targets=False):
        """
        Compute observation uncertainty.
        
        Var(y) = C @ Cov(x) @ C.T + R
        std(y) = sqrt(Var(y))
        
        Args:
            x_cov: State covariance (batch, latent_dim, latent_dim)
            C_expanded: Pre-expanded observation matrix (batch, measurement_dim, latent_dim)
            R_diag: Diagonal of measurement noise covariance (1, measurement_dim)
            project_to_targets: If True and target_indices exist, project to target subset
            
        Returns:
            y_std: Observation standard deviation (batch, measurement_dim) or (batch, output_dim)
        """
        C_x_cov = C_expanded @ x_cov  # (batch, measurement_dim, latent_dim)
        y_var = (C_x_cov * C_expanded).sum(dim=-1) + R_diag  # (batch, measurement_dim)
        
        # Project to targets if requested and mapping exists
        if project_to_targets and self.target_indices is not None:
            y_var = y_var[:, self.target_indices]  # (batch, output_dim)
        
        y_std = torch.sqrt(y_var)
        return y_std
    
    def _run_substep_prediction(self, x_mean, x_cov, u_t, d_t, A, B, E, A_expanded):
        """
        Run substep prediction loop with zero-order hold on inputs.
        
        Applies the same control and disturbance inputs for n_substeps iterations,
        propagating the state and covariance forward. This allows for finer temporal
        resolution in simulation while maintaining the same data time scale.
        
        Args:
            x_mean: State mean at start of major time step (batch, latent_dim)
            x_cov: State covariance at start of major time step (batch, latent_dim, latent_dim)
            u_t: Control input (held constant over substeps) (batch, control_dim)
            d_t: Disturbance input (held constant over substeps) (batch, disturbance_dim) or None
            A: State transition matrix (latent_dim, latent_dim)
            B: Control matrix (latent_dim, control_dim)
            E: Disturbance matrix (latent_dim, disturbance_dim) or None
            A_expanded: Pre-expanded A for batch operations (batch, latent_dim, latent_dim)
            
        Returns:
            x_mean: State mean after all substeps (batch, latent_dim)
            x_cov: State covariance after all substeps (batch, latent_dim, latent_dim)
        """
        for substep in range(self.n_substeps):
            x_mean = self._compute_state_mean(x_mean, u_t, A, B, d_t, E)
            x_cov = self._compute_state_covariance(x_cov, A_expanded)
        
        return x_mean, x_cov
    
    def encode_with_kf(self, control_inputs, measurements, disturbance_inputs=None, **kwargs):
        """
        Encode state trajectory using Kalman filter with measurements.
        
        Performs sequential filtering through all measurements in the given horizon,
        applying predict-update cycles at each timestep.
        
        Args:
            control_inputs: Historical control inputs (batch, control_dim, window)
            measurements: Historical observations (batch, output_dim, window)
            disturbance_inputs: Historical disturbances (batch, disturbance_dim, window) - optional
            **kwargs: Model-specific parameters (unused in base LKF, available for subclasses)
            
        Returns:
            x_mean_traj: Filtered state mean trajectory (batch, latent_dim, window)
            x_cov_traj: Filtered state covariance trajectory (batch, latent_dim, latent_dim, window)
        """
        start_time = time.time()
        
        batch_size = control_inputs.shape[0]
        window_size = control_inputs.shape[-1]
        device = control_inputs.device
        
        # Compute observation mean for initialization
        # Shape: (output_dim,) - mean across batch and time for each observation
        obs_mean_per_feature = torch.mean(measurements, dim=(0, 2))  # (output_dim,)
        
        # Initialize learnable x0 with observation means on first call
        if self.learnable_x0 and not self._x0_initialized:
            with torch.no_grad():
                # If latent_dim == output_dim, use one-to-one mapping
                # Otherwise, use mean of all observations for each state
                if self.latent_dim == self.measurement_dim:
                    self.x0.copy_(obs_mean_per_feature)
                else:
                    # Fallback: use scalar mean for all states
                    obs_mean_scalar = torch.mean(obs_mean_per_feature).item()
                    self.x0.fill_(obs_mean_scalar)
            self._x0_initialized = True
        
        # Initialize state mean: use learnable x0 if available, otherwise observation means
        if self.learnable_x0:
            x_mean = self.x0.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            # Use per-feature observation means if dimensions match, otherwise use scalar mean
            if self.latent_dim == self.measurement_dim:
                x_mean = obs_mean_per_feature.unsqueeze(0).expand(batch_size, -1).clone()
            else:
                obs_mean_scalar = torch.mean(obs_mean_per_feature).item()
                x_mean = torch.full((batch_size, self.latent_dim), obs_mean_scalar, device=device)
        
        x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        x_mean_traj = []
        x_cov_traj = []
        
        # Pre-expand matrices for batch operations (optimization)
        A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)
        R_expanded = self.R.unsqueeze(0).expand(batch_size, -1, -1)
        I_expanded = torch.eye(self.latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Sequential filtering through all measurements in the horizon
        for t in range(window_size):
            u_t = control_inputs[:, :, t]  # (batch, control_dim)
            y_t = measurements[:, :, t]    # (batch, output_dim)
            d_t = disturbance_inputs[:, :, t] if self.has_disturbances and disturbance_inputs is not None else None
            
            # Predict state with substeps (zero-order hold on u_t, d_t)
            x_mean, x_cov = self._run_substep_prediction(x_mean, x_cov, u_t, d_t, self.A, self.B, self.E, A_expanded)
            
            # Update with measurement
            x_mean, x_cov = self.kf.update(x_mean, x_cov, y_t, self.C,
                                          C_expanded=C_expanded, R_expanded=R_expanded, I_expanded=I_expanded)
            x_mean_traj.append(x_mean)
            x_cov_traj.append(x_cov)
        
        # Record timing
        elapsed = time.time() - start_time
        self.timing_stats['encode_with_kf']['total'] += elapsed
        self.timing_stats['encode_with_kf']['count'] += 1
        
        return torch.stack(x_mean_traj, dim=-1), torch.stack(x_cov_traj, dim=-1)
    
    def predict_with_kf(self, x_mean, x_cov, control_inputs, disturbance_inputs=None, **kwargs):
        """
        Predict forward using Kalman filter without measurements.
        
        Args:
            x_mean: Initial state mean (batch, latent_dim)
            x_cov: Initial state covariance (batch, latent_dim, latent_dim)
            control_inputs: Control inputs (batch, control_dim, time_steps)
            disturbance_inputs: Disturbance inputs (batch, disturbance_dim, time_steps) - optional
            **kwargs: Model-specific parameters (unused in base LKF, available for subclasses)
            
        Returns:
            predictions: Predicted outputs (batch, output_dim, time_steps)
            uncertainties: Prediction uncertainties (batch, output_dim, time_steps)
        """
        start_time = time.time()
        
        batch_size = control_inputs.shape[0]
        time_steps = control_inputs.shape[-1]
        
        # Pre-expand matrices (optimization)
        A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)
        R_diag = torch.diagonal(self.R, dim1=0, dim2=1).unsqueeze(0)  # (1, output_dim)
        
        predictions = []
        uncertainties = []
        
        for t in range(time_steps):
            u_t = control_inputs[:, :, t]  # (batch, control_dim)
            d_t = disturbance_inputs[:, :, t] if self.has_disturbances and disturbance_inputs is not None else None
            
            # Predict state with substeps (zero-order hold on u_t, d_t)
            x_mean, x_cov = self._run_substep_prediction(x_mean, x_cov, u_t, d_t, self.A, self.B, self.E, A_expanded)
            
            # Predict observation (project to targets for output)
            y_pred = self._compute_observation(x_mean, project_to_targets=True, u_t=u_t)
            predictions.append(y_pred)
            
            # Compute uncertainty (project to targets for output)
            y_std = self._compute_observation_uncertainty(x_cov, C_expanded, R_diag, project_to_targets=True)
            uncertainties.append(y_std)
        
        # Stack results
        predictions = torch.stack(predictions, dim=-1)  # (batch, output_dim, time_steps)
        uncertainties = torch.stack(uncertainties, dim=-1)  # (batch, output_dim, time_steps)
        
        # Record timing
        elapsed = time.time() - start_time
        self.timing_stats['predict_with_kf']['total'] += elapsed
        self.timing_stats['predict_with_kf']['count'] += 1
        
        return predictions, uncertainties
    
    def forward(self, control_inputs, historical_controls=None, historical_observations=None, 
                disturbance_inputs=None, historical_disturbances=None, **kwargs):
        """
        Forward pass: encode with measurements (if provided) then predict.
        
        Args:
            control_inputs: Control inputs for prediction (batch, control_dim, time_steps)
            historical_controls: Historical controls for encoding (batch, control_dim, window)
            historical_observations: Historical observations for encoding (batch, output_dim, window)
            disturbance_inputs: Disturbance inputs for prediction (batch, disturbance_dim, time_steps)
            historical_disturbances: Historical disturbances (batch, disturbance_dim, window)
            **kwargs: Model-specific parameters (e.g., matrix_generation_features for PWLKF)
            
        Returns:
            predictions: Predicted outputs (batch, output_dim, time_steps)
            uncertainties: Prediction uncertainties (batch, output_dim, time_steps)
        """
        # Encode state from historical measurements (if provided)
        if historical_controls is not None and historical_observations is not None:
            x_mean_traj, x_cov_traj = self.encode_with_kf(
                historical_controls, historical_observations, historical_disturbances, **kwargs
            )
            x_mean, x_cov = x_mean_traj[:, :, -1], x_cov_traj[:, :, :, -1]  # Use last time step
        else:
            # No historical data - use zero initialization
            batch_size = control_inputs.shape[0]
            device = control_inputs.device
            x_mean = torch.zeros(batch_size, self.latent_dim, device=device)
            x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Predict forward
        predictions, uncertainties = self.predict_with_kf(
            x_mean, x_cov, control_inputs, disturbance_inputs, **kwargs
        )
        
        return predictions, uncertainties
    
    def predict_full_trajectory(self, control_inputs, historical_controls=None, historical_observations=None,
                disturbance_inputs=None, historical_disturbances=None, **kwargs):
        """
        Predict method that returns full trajectory including historical reconstruction.
        
        Args:
            control_inputs: Control inputs for prediction (batch, control_dim, time_steps)
            historical_controls: Historical controls (batch, control_dim, window)
            historical_observations: Historical observations (batch, output_dim, window)
            disturbance_inputs: Disturbance inputs for prediction (batch, disturbance_dim, time_steps)
            historical_disturbances: Historical disturbances (batch, disturbance_dim, window)
            **kwargs: Model-specific parameters (e.g., matrix_generation_features for PWLKF)
            
        Returns:
            predictions_full: Predicted outputs (batch, output_dim, window + time_steps)
            uncertainties_full: Prediction uncertainties (batch, output_dim, window + time_steps)
            window_size: Number of timesteps in historical window
        """
        batch_size = control_inputs.shape[0]
        device = control_inputs.device
        window_size = historical_controls.shape[-1] if historical_controls is not None else 0
        
        hist_predictions = []
        hist_uncertainties = []
        
        # Capture historical predictions by running encode step-by-step
        if historical_controls is not None and historical_observations is not None:
            # Initialize state with learnable P0
            x_mean = torch.zeros(batch_size, self.latent_dim, device=device)
            x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()
            
            # Pre-expand matrices
            A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)
            C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)
            R_expanded = self.R.unsqueeze(0).expand(batch_size, -1, -1)
            I_expanded = torch.eye(self.latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            R_diag = torch.diagonal(self.R, dim1=0, dim2=1).unsqueeze(0)
            
            # Sequential filtering through historical window
            for t in range(window_size):
                u_t = historical_controls[:, :, t]
                y_t = historical_observations[:, :, t]
                d_t = historical_disturbances[:, :, t] if self.has_disturbances and historical_disturbances is not None else None
                
                # Predict state with substeps (zero-order hold on u_t, d_t)
                x_mean, x_cov = self._run_substep_prediction(x_mean, x_cov, u_t, d_t, self.A, self.B, self.E, A_expanded)
                
                # Compute predicted observation (project to targets for output)
                y_pred = self._compute_observation(x_mean, project_to_targets=True, u_t=u_t)
                
                # Compute uncertainty (project to targets for output)
                y_std = self._compute_observation_uncertainty(x_cov, C_expanded, R_diag, project_to_targets=True)
                
                hist_predictions.append(y_pred)
                hist_uncertainties.append(y_std)
                
                # Update with measurement
                x_mean, x_cov = self.kf.update(x_mean, x_cov, y_t, self.C,
                                              C_expanded=C_expanded, R_expanded=R_expanded, I_expanded=I_expanded)
        else:
            # No historical data
            x_mean = torch.zeros(batch_size, self.latent_dim, device=device)
            x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Predict forward
        predictions, uncertainties = self.predict_with_kf(
            x_mean, x_cov, control_inputs, disturbance_inputs, **kwargs
        )
        
        # Concatenate historical and prediction phases
        if window_size > 0:
            hist_predictions = torch.stack(hist_predictions, dim=-1)
            hist_uncertainties = torch.stack(hist_uncertainties, dim=-1)
            
            predictions_full = torch.cat([hist_predictions, predictions], dim=-1)
            uncertainties_full = torch.cat([hist_uncertainties, uncertainties], dim=-1)
        else:
            predictions_full = predictions
            uncertainties_full = uncertainties
        
        return predictions_full, uncertainties_full, window_size
    
    def _compute_loss(self, predictions, targets, uncertainties):
        """
        Compute negative log-likelihood loss with multi-horizon support.
        
        For a diagonal Gaussian distribution, the NLL is:
        NLL = 0.5 * sum_over_dims(log(2π) + log(σ²) + (y - ŷ)²/σ²)
        
        With multi-horizon loss, we compute separate losses for different time windows
        and combine them with configurable weights. This allows emphasizing near-term
        vs. long-term predictions differently.
        
        Timesteps not covered by any window automatically get weight 1.0.
        
        NOTE: Curriculum learning is applied BEFORE this method is called (in training_step
        and validation_step via apply_curriculum_to_inputs). The inputs here are already
        at the curriculum horizon length.
        
        Args:
            predictions: (batch, output_dim, time_steps) - already sliced to curriculum horizon
            targets: (batch, output_dim, time_steps) - already sliced to curriculum horizon
            uncertainties: (batch, output_dim, time_steps) - already sliced to curriculum horizon
            
        Returns:
            dict with 'nll' (negative log-likelihood), 'mse' (mean squared error),
            and per-horizon losses ('nll_horizon_X_Y', 'mse_horizon_X_Y')
        """
        total_nll = 0.0
        total_mse = 0.0
        losses_by_horizon = {}
        
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-6
        
        total_time_steps = predictions.shape[-1]
        
        # Build complete window list with gap filling (weight 1.0 for gaps)
        windows_with_weights = self._fill_window_gaps(total_time_steps)
        
        # Compute loss for each horizon window
        for (start, end), weight in windows_with_weights:
            # Slice predictions/targets/uncertainties for this horizon
            pred_slice = predictions[..., start:end]
            target_slice = targets[..., start:end]
            unc_slice = uncertainties[..., start:end]
            
            # Compute squared error
            squared_error = (pred_slice - target_slice) ** 2
            sigma = unc_slice + epsilon
            
            # Negative log-likelihood per element
            # NLL = 0.5 * (log(2π) + log(σ²) + (y - ŷ)²/σ²)
            nll_per_element = 0.5 * (torch.log(2 * torch.pi * (sigma ** 2)) + squared_error / (sigma ** 2))
            
            # Sum over dimensions (output_dim, time_steps), average over batch
            # This gives NLL per sample, then average across batch
            nll_horizon = nll_per_element.sum(dim=(1, 2)).mean()
            mse_horizon = squared_error.mean()
            
            # Apply weight and accumulate
            weighted_nll = nll_horizon * weight
            total_nll += weighted_nll
            total_mse += mse_horizon
            
            # Store per-horizon metrics (unweighted for analysis)
            horizon_name = f'horizon_{start}_{end}'
            losses_by_horizon[f'nll_{horizon_name}'] = nll_horizon.item()
            losses_by_horizon[f'mse_{horizon_name}'] = mse_horizon.item()
            losses_by_horizon[f'weighted_nll_{horizon_name}'] = weighted_nll.item()
        
        # Normalize total NLL by sum of weights to maintain comparable scale
        weight_sum = sum(weight for _, weight in windows_with_weights)
        total_nll /= weight_sum
        total_mse /= len(windows_with_weights)
        
        return {'nll': total_nll, 'mse': total_mse, **losses_by_horizon}
    
    def _fill_window_gaps(self, total_time_steps):
        """
        Fill gaps in horizon windows with weight 1.0.
        
        If windows don't cover all timesteps, creates additional windows for gaps
        with default weight 1.0. Also handles overlaps by keeping original windows.
        Clips windows that exceed the actual forecast length.
        
        Args:
            total_time_steps: Total number of timesteps in the forecast
            
        Returns:
            List of ((start, end), weight) tuples covering all timesteps
        """
        # Convert windows with None end to explicit end, and clip windows to actual forecast length
        explicit_windows = []
        for (start, end), weight in zip(self.horizon_loss_windows, self.horizon_loss_weights):
            if end is None:
                end = total_time_steps
            
            # Clip window to actual forecast length
            if start >= total_time_steps:
                # Window starts beyond forecast - skip it entirely
                continue
            if end > total_time_steps:
                # Window extends beyond forecast - clip it
                end = total_time_steps
            
            # Only add non-empty windows
            if start < end:
                explicit_windows.append(((start, end), weight))
        
        # Find all covered timesteps
        covered = set()
        for (start, end), _ in explicit_windows:
            covered.update(range(start, end))
        
        # Find gaps (uncovered timesteps)
        all_steps = set(range(total_time_steps))
        gaps = sorted(all_steps - covered)
        
        # Convert gaps to contiguous windows
        gap_windows = []
        if gaps:
            gap_start = gaps[0]
            prev_step = gaps[0]
            
            for step in gaps[1:]:
                if step != prev_step + 1:
                    # End of contiguous gap
                    gap_windows.append(((gap_start, prev_step + 1), 1.0))
                    gap_start = step
                prev_step = step
            
            # Add final gap window
            gap_windows.append(((gap_start, prev_step + 1), 1.0))
        
        # Combine original windows and gap windows, sorted by start position
        all_windows = explicit_windows + gap_windows
        all_windows.sort(key=lambda x: x[0][0])
        
        return all_windows
    
    def apply_curriculum_to_inputs(self, control_inputs, disturbance_inputs, targets):
        """
        Apply curriculum learning by slicing inputs to reduce forward pass computation.
        
        Supports two modes:
        1. Absolute steps (curriculum_steps) - from window_stages strategy
        2. Fraction-based (curriculum_fraction) - from linear strategy
        
        During early training (low curriculum_fraction or curriculum_steps), only predict
        first portion of the forecast horizon. This reduces computational cost of the 
        forward pass (Kalman filter predictions) in addition to loss computation.
        
        Args:
            control_inputs: (batch, control_dim, time_steps) or None
            disturbance_inputs: (batch, disturbance_dim, time_steps) or None
            targets: (batch, output_dim, time_steps)
            
        Returns:
            Tuple of (control_inputs, disturbance_inputs, targets) - truncated if curriculum is active
        """
        # Check for absolute steps first (window_stages strategy)
        if hasattr(self, 'curriculum_steps') and self.curriculum_steps is not None:
            mask_steps = self.curriculum_steps
        elif hasattr(self, 'curriculum_fraction') and self.curriculum_fraction < 1.0:
            # Fall back to fraction-based (linear strategy)
            horizon_steps = targets.size(-1)
            mask_steps = int(horizon_steps * self.curriculum_fraction)
            mask_steps = max(1, mask_steps)
        else:
            # No curriculum - use full horizon
            return control_inputs, disturbance_inputs, targets
        
        # Slice control inputs if present
        if control_inputs is not None:
            control_inputs = control_inputs[..., :mask_steps]
        
        # Slice disturbance inputs if present
        if disturbance_inputs is not None:
            disturbance_inputs = disturbance_inputs[..., :mask_steps]
        
        # Slice targets
        targets = targets[..., :mask_steps]
        
        return control_inputs, disturbance_inputs, targets
    
    def apply_curriculum_mask(self, predictions, targets, uncertainties):
        """
        Apply curriculum learning by only computing loss on a fraction of the forecast horizon.
        
        NOTE: This method is now primarily for backward compatibility. The main curriculum
        slicing happens in apply_curriculum_to_inputs() which slices BEFORE the forward pass.
        This method handles any residual slicing needed (e.g., if predictions came from
        a different code path).
        
        During early training (low curriculum_fraction), only the first portion of the 
        forecast horizon is used for loss computation. As training progresses, 
        curriculum_fraction increases to 1.0, using the full horizon.
        
        Args:
            predictions: (batch, output_dim, time_steps)
            targets: (batch, output_dim, time_steps)
            uncertainties: (batch, output_dim, time_steps)
            
        Returns:
            Tuple of (predictions, targets, uncertainties) - truncated if curriculum_fraction < 1.0
        """
        if self.curriculum_fraction >= 1.0:
            # Full difficulty - use all timesteps
            return predictions, targets, uncertainties
        
        # Only use first X% of the forecast horizon during curriculum training
        horizon_steps = predictions.size(-1)
        mask_steps = int(horizon_steps * self.curriculum_fraction)
        mask_steps = max(1, mask_steps)  # At least 1 step
        
        return predictions[..., :mask_steps], targets[..., :mask_steps], uncertainties[..., :mask_steps]
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        print("\n" + "="*70)
        print("TRAINING LOSS: Negative Log-Likelihood (NLL)")
        print("  - train_loss and val_loss use NLL (Gaussian likelihood)")
        print("  - Also logging MSE for reference (train_mse, val_mse)")
        if len(self.horizon_loss_windows) > 1:
            print(f"\nMULTI-HORIZON LOSS ENABLED:")
            print(f"  - Horizon loss windows: {self.horizon_loss_windows}")
            print(f"  - Loss weights: {self.horizon_loss_weights}")
            print(f"  - Per-horizon metrics logged as: train/val_nll_horizon_X_Y")
        if self.curriculum_fraction < 1.0:
            print(f"\nCURRICULUM LEARNING ENABLED:")
            print(f"  - Starting curriculum fraction: {self.curriculum_fraction:.2%}")
            print(f"  - Forward pass predicts only first {self.curriculum_fraction:.2%} of forecast horizon")
            print(f"  - Reduces both computation cost AND memory usage")
            print(f"  - Will progressively increase to 100% during training")
        print("="*70 + "\n")
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        control_inputs = batch.get('control_inputs')
        disturbance_inputs = batch.get('disturbance_inputs')
        historical_controls = batch.get('historical_controls')
        historical_disturbances = batch.get('historical_disturbances')
        historical_observations = batch.get('historical_observations')
        targets = batch['targets']  # (batch, n_targets, fcast_len) - transposed in collate_fn
        
        # Apply curriculum learning: slice inputs to reduce forward pass computation
        control_inputs, disturbance_inputs, targets = self.apply_curriculum_to_inputs(
            control_inputs, disturbance_inputs, targets
        )
        
        # Forward pass (now predicts only curriculum horizon, not full horizon)
        predictions, uncertainties = self(
            control_inputs, historical_controls, historical_observations,
            disturbance_inputs, historical_disturbances
        )
        # predictions: (batch, output_dim, curriculum_steps) where curriculum_steps <= fcast_len
        # uncertainties: (batch, output_dim, curriculum_steps)
        
        # Compute loss (no additional slicing needed - already at curriculum horizon)
        losses = self._compute_loss(predictions, targets, uncertainties)
        
        # Log main losses
        self.log('train_loss', losses['nll'], prog_bar=True)
        self.log('train_nll', losses['nll'], prog_bar=True)
        self.log('train_mse', losses['mse'], prog_bar=True)
        
        # Log per-horizon losses (only if multi-horizon is enabled)
        if len(self.horizon_loss_windows) > 1:
            for key, value in losses.items():
                if key.startswith(('nll_horizon', 'mse_horizon', 'weighted_nll')):
                    self.log(f'train_{key}', value)
        
        return {
            'loss': losses['nll'],
            'predictions': predictions,
            'targets': targets,
            'uncertainties': uncertainties
        }
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        control_inputs = batch.get('control_inputs')
        disturbance_inputs = batch.get('disturbance_inputs')
        historical_controls = batch.get('historical_controls')
        historical_disturbances = batch.get('historical_disturbances')
        historical_observations = batch.get('historical_observations')
        targets = batch['targets']  # (batch, n_targets, fcast_len) - transposed in collate_fn
        
        # Apply curriculum learning: slice inputs to reduce forward pass computation
        control_inputs, disturbance_inputs, targets = self.apply_curriculum_to_inputs(
            control_inputs, disturbance_inputs, targets
        )
        
        # Forward pass (now predicts only curriculum horizon, not full horizon)
        predictions, uncertainties = self(
            control_inputs, historical_controls, historical_observations,
            disturbance_inputs, historical_disturbances
        )
        # predictions: (batch, output_dim, curriculum_steps) where curriculum_steps <= fcast_len
        # uncertainties: (batch, output_dim, curriculum_steps)
        
        # Compute loss (no additional slicing needed - already at curriculum horizon)
        losses = self._compute_loss(predictions, targets, uncertainties)
        
        # Log main losses
        self.log('val_loss', losses['nll'], prog_bar=True)
        self.log('val_nll', losses['nll'], prog_bar=True)
        self.log('val_mse', losses['mse'], prog_bar=True)
        self.log('val_mean_uncertainty', uncertainties.mean())
        
        # Log per-horizon losses (only if multi-horizon is enabled)
        if len(self.horizon_loss_windows) > 1:
            for key, value in losses.items():
                if key.startswith(('nll_horizon', 'mse_horizon', 'weighted_nll')):
                    self.log(f'val_{key}', value)
        
        return {
            'loss': losses['nll'],
            'predictions': predictions,
            'targets': targets,
            'uncertainties': uncertainties
        }
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        control_inputs = batch.get('control_inputs')
        disturbance_inputs = batch.get('disturbance_inputs')
        historical_controls = batch.get('historical_controls')
        historical_disturbances = batch.get('historical_disturbances')
        historical_observations = batch.get('historical_observations')
        targets = batch['targets']  # (batch, n_targets, fcast_len) - transposed in collate_fn
        
        # Forward pass (NOT predict, since targets only contain forward predictions)
        predictions, uncertainties = self(
            control_inputs, historical_controls, historical_observations,
            disturbance_inputs, historical_disturbances
        )
        # predictions: (batch, output_dim, time_steps)
        # uncertainties: (batch, output_dim, time_steps)
        
        # Compute loss (returns dict with main losses + per-horizon losses)
        losses = self._compute_loss(predictions, targets, uncertainties)
        
        # Log main losses
        self.log('test_loss', losses['nll'])
        self.log('test_nll', losses['nll'])
        self.log('test_mse', losses['mse'])
        self.log('test_mean_uncertainty', uncertainties.mean())
        
        # Log per-horizon losses (only if multi-horizon is enabled)
        if len(self.horizon_loss_windows) > 1:
            for key, value in losses.items():
                if key.startswith(('nll_horizon', 'mse_horizon', 'weighted_nll')):
                    self.log(f'test_{key}', value)
        
        return {
            'loss': losses['nll'],
            'predictions': predictions,
            'targets': targets,
            'uncertainties': uncertainties
        }
    
    def print_timing_stats(self):
        """Print timing statistics for encode and predict methods and log to TensorBoard."""
        print("\n" + "="*70)
        print("LKF TIMING STATISTICS")
        print("="*70)
        
        total_time = 0.0
        for method_name, stats in self.timing_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total'] / stats['count']
                total_time += stats['total']
                print(f"\n{method_name}:")
                print(f"  Total time: {stats['total']:.4f}s")
                print(f"  Count: {stats['count']}")
                print(f"  Average: {avg_time:.6f}s")
        
        print(f"\nTOTAL LKF TIME: {total_time:.4f}s")
        
        # Calculate percentages
        if total_time > 0:
            print("\nRelative Time Distribution:")
            for method_name, stats in self.timing_stats.items():
                if stats['count'] > 0:
                    percentage = (stats['total'] / total_time) * 100
                    print(f"  {method_name}: {percentage:.1f}%")
        
        print("="*70 + "\n")
        
        # Log to TensorBoard
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment'):
            logger = self.logger.experiment
            
            # Log absolute times
            for method_name, stats in self.timing_stats.items():
                if stats['count'] > 0:
                    logger.add_scalar(f'lkf_timing/{method_name}_total_seconds', stats['total'], 0)
                    logger.add_scalar(f'lkf_timing/{method_name}_avg_seconds', 
                                    stats['total'] / stats['count'], 0)
                    logger.add_scalar(f'lkf_timing/{method_name}_count', stats['count'], 0)
            
            # Log total and percentages
            logger.add_scalar('lkf_timing/total_seconds', total_time, 0)
            logger.add_scalar('lkf_timing/total_minutes', total_time / 60, 0)
            
            for method_name, stats in self.timing_stats.items():
                if stats['count'] > 0 and total_time > 0:
                    percentage = (stats['total'] / total_time) * 100
                    logger.add_scalar(f'lkf_timing/{method_name}_percentage', percentage, 0)
            
            print("[LKF Timing] Performance metrics logged to TensorBoard under 'lkf_timing/*'")

    
    def reset_timing_stats(self):
        """Reset timing statistics."""
        for stats in self.timing_stats.values():
            stats['total'] = 0.0
            stats['count'] = 0
    
    def on_train_end(self):
        """Called at the end of training to print timing statistics."""
        self.print_timing_stats()
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    @property
    def R(self):
        """Measurement noise covariance."""
        return self.kf.R
    
    @property
    def Q(self):
        """Process noise covariance."""
        return self.kf.Q
    
    def print_learned_matrices(self):
        """
        Print all learned matrices and bias vectors from the trained model.
        
        Prints:
        - A: state transition matrix (latent_dim, latent_dim)
        - B: control influence matrix (latent_dim, control_dim)
        - C: observation matrix (measurement_dim, latent_dim)
        - E: disturbance influence matrix (latent_dim, disturbance_dim) [if exists]
        - b_x: state bias vector (latent_dim,) [if learnable]
        - b_y: observation bias vector (measurement_dim,) [if learnable]
        - Q: process noise covariance (latent_dim, latent_dim)
        - R: measurement noise covariance (measurement_dim, measurement_dim)
        """
        print("\n" + "="*80)
        print("LEARNED STATE-SPACE MATRICES AND PARAMETERS")
        print("="*80)
        
        # Print dimensions
        print(f"\nModel Dimensions:")
        print(f"  Latent state dimension: {self.latent_dim}")
        print(f"  Control dimension: {self.control_dim}")
        print(f"  Disturbance dimension: {self.disturbance_dim}")
        print(f"  Observation dimension: {self.measurement_dim}")
        print(f"  Output dimension: {self.output_dim}")
        
        # Print feature names
        print(f"\nFeature Names:")
        print(f"  Control features: {self.control_features}")
        print(f"  Disturbance features: {self.disturbance_features}")
        if self.observation_features:
            print(f"  Observation features: {self.observation_features}")
        if self.target_features:
            print(f"  Target features: {self.target_features}")
        
        # Print A matrix (state transition)
        print(f"\n{'='*80}")
        print("A Matrix (State Transition): x[t+1] = A @ x[t] + ...")
        print(f"Shape: {self.A.shape}")
        print("="*80)
        with torch.no_grad():
            A_np = self.A.cpu().numpy()
            for i in range(self.latent_dim):
                row_str = "  " + "  ".join([f"{val:8.4f}" for val in A_np[i]])
                print(row_str)
        
        # Print B matrix (control influence)
        print(f"\n{'='*80}")
        print("B Matrix (Control Influence): x[t+1] = ... + B @ u[t] + ...")
        print(f"Shape: {self.B.shape}")
        print(f"Control features: {self.control_features}")
        print("="*80)
        with torch.no_grad():
            B_np = self.B.cpu().numpy()
            for i in range(self.latent_dim):
                row_str = "  " + "  ".join([f"{val:8.4f}" for val in B_np[i]])
                print(row_str)
        
        # Print E matrix (disturbance influence) if it exists
        if self.has_disturbances and self.E is not None:
            print(f"\n{'='*80}")
            print("E Matrix (Disturbance Influence): x[t+1] = ... + E @ d[t] + ...")
            print(f"Shape: {self.E.shape}")
            print(f"Disturbance features: {self.disturbance_features}")
            print("="*80)
            with torch.no_grad():
                E_np = self.E.cpu().numpy()
                for i in range(self.latent_dim):
                    row_str = "  " + "  ".join([f"{val:8.4f}" for val in E_np[i]])
                    print(row_str)
        
        # Print C matrix (observation)
        print(f"\n{'='*80}")
        print("C Matrix (Observation): y[t] = C @ x[t] + ...")
        print(f"Shape: {self.C.shape}")
        if self.observation_features:
            print(f"Observation features: {self.observation_features}")
        print("="*80)
        with torch.no_grad():
            C_np = self.C.cpu().numpy()
            for i in range(self.measurement_dim):
                row_str = "  " + "  ".join([f"{val:8.4f}" for val in C_np[i]])
                obs_name = self.observation_features[i] if self.observation_features else f"Obs_{i+1}"
                print(f"  {obs_name}: {row_str}")
        
        # Print b_x (state bias) if learnable
        if self.use_state_bias:
            print(f"\n{'='*80}")
            print("b_x Vector (State Bias): x[t+1] = ... + b_x")
            print(f"Shape: {self.b_x.shape}")
            print("="*80)
            with torch.no_grad():
                b_x_np = self.b_x.cpu().numpy()
                b_x_str = "  " + "  ".join([f"{val:8.4f}" for val in b_x_np])
                print(b_x_str)
        
        # Print b_y (observation bias) if learnable
        if self.use_observation_bias:
            print(f"\n{'='*80}")
            print("b_y Vector (Observation Bias): y[t] = ... + b_y")
            print(f"Shape: {self.b_y.shape}")
            if self.observation_features:
                print(f"Observation features: {self.observation_features}")
            print("="*80)
            with torch.no_grad():
                b_y_np = self.b_y.cpu().numpy()
                for i, val in enumerate(b_y_np):
                    obs_name = self.observation_features[i] if self.observation_features else f"Obs_{i+1}"
                    print(f"  {obs_name}: {val:8.4f}")
        
        # Print x0 (initial state) if learnable
        if self.learnable_x0:
            print(f"\n{'='*80}")
            print("x0 Vector (Initial State Mean): x[0] = x0")
            print(f"Shape: {self.x0.shape}")
            print("="*80)
            with torch.no_grad():
                x0_np = self.x0.cpu().numpy()
                x0_str = "  " + "  ".join([f"{val:8.4f}" for val in x0_np])
                print(x0_str)
        
        # Print Q matrix (process noise covariance)
        print(f"\n{'='*80}")
        print("Q Matrix (Process Noise Covariance): w[t] ~ N(0, Q)")
        print(f"Shape: {self.Q.shape}")
        print("="*80)
        with torch.no_grad():
            Q_np = self.Q.cpu().numpy()
            # Print diagonal values (most relevant for interpretation)
            print("  Diagonal values:")
            for i in range(self.latent_dim):
                print(f"    State_{i+1}: {Q_np[i, i]:8.6f}")
            
            # Only print full matrix if small enough
            if self.latent_dim <= 5:
                print("  Full matrix:")
                for i in range(self.latent_dim):
                    row_str = "    " + "  ".join([f"{val:8.6f}" for val in Q_np[i]])
                    print(row_str)
        
        # Print R matrix (measurement noise covariance)
        print(f"\n{'='*80}")
        print("R Matrix (Measurement Noise Covariance): v[t] ~ N(0, R)")
        print(f"Shape: {self.R.shape}")
        print("="*80)
        with torch.no_grad():
            R_np = self.R.cpu().numpy()
            # Print diagonal values (most relevant for interpretation)
            print("  Diagonal values:")
            for i in range(self.measurement_dim):
                obs_name = self.observation_features[i] if self.observation_features else f"Obs_{i+1}"
                print(f"    {obs_name}: {R_np[i, i]:8.6f}")
            
            # Only print full matrix if small enough
            if self.measurement_dim <= 5:
                print("  Full matrix:")
                for i in range(self.measurement_dim):
                    row_str = "    " + "  ".join([f"{val:8.6f}" for val in R_np[i]])
                    print(row_str)
        
        print("\n" + "="*80)
        print("END OF LEARNED MATRICES")
        print("="*80 + "\n")

    def fit(self, training_set, validation_set=None, testing_set=None):
        """
        Fit the model, optionally preceded by subspace matrix initialization.

        If ``ss_init`` was set in the constructor, the selected subspace
        identification method is run first on the concatenated training data to
        initialize A, B, C (and E, D) before gradient-based training begins.

        Args:
            training_set: List of DataFrames containing training data.
            validation_set: List of DataFrames containing validation data.
            testing_set: List of DataFrames containing testing data.
        """
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        if self.ss_init == 'nfoursid':
            _logger.info(
                "N4SID subspace initialization enabled. "
                f"kwargs={self.ss_init_kwargs} ..."
            )
            self.initialize_matrices_from_n4sid(training_set, **self.ss_init_kwargs)
            _logger.info("N4SID initialization complete. Starting gradient training.")
        elif self.ss_init == 'sippy':
            _logger.info(
                "SIPPY subspace initialization enabled. "
                f"kwargs={self.ss_init_kwargs} ..."
            )
            self.initialize_matrices_from_sippy(training_set, **self.ss_init_kwargs)
            _logger.info("SIPPY initialization complete. Starting gradient training.")

        super().fit(training_set, validation_set, testing_set)

    def initialize_matrices_from_n4sid(self, training_set, num_block_rows=10):
        """
        Identify initial state-space matrices from data using the N4SID algorithm.

        Concatenates all DataFrames in ``training_set``, drops rows with NaNs,
        then runs ``NFourSID.subspace_identification()`` followed by
        ``NFourSID.system_identification(rank=self.latent_dim)`` to obtain A,
        B, C (and D) matrices.  The identified matrices are then copied into the
        model parameters via :meth:`_apply_n4sid_matrices`.

        The combined input matrix returned by N4SID is split at ``control_dim``
        so that the first ``control_dim`` columns initialize B (control) and
        the remaining columns initialize E (disturbance), matching the
        model's separation of inputs.

        Args:
            training_set: List of DataFrames. Must contain all columns in
                ``observation_features`` (or ``target_features``) plus all
                ``control_features`` and ``disturbance_features``.
            num_block_rows: Number of block rows for the N4SID Hankel matrix.
                            Larger values give a better estimate at the cost of
                            more memory and compute.
        """
        import logging as _logging
        import pandas as pd
        from nfoursid.nfoursid import NFourSID

        _logger = _logging.getLogger(__name__)

        # Determine observation columns (y in N4SID = what C observes)
        obs_cols = (
            self.observation_features
            if self.observation_features is not None
            else self.target_features
        )
        if obs_cols is None:
            raise ValueError(
                "observation_features or target_features must be set for N4SID "
                "initialization. Provide at least one as a list of column names."
            )

        # Combined input columns: controls first, then disturbances
        input_cols = list(self.control_features) + list(self.disturbance_features)

        # Concatenate and clean training data
        all_cols = obs_cols + input_cols
        data = pd.concat(training_set, axis=0)[all_cols].dropna()

        if len(data) == 0:
            raise ValueError(
                "No valid data rows remain after dropping NaNs. "
                "N4SID initialization requires at least some non-NaN rows."
            )

        min_recommended = 2 * num_block_rows * len(obs_cols)
        if len(data) < min_recommended:
            _logger.warning(
                f"N4SID initialization: only {len(data)} clean rows available; "
                f"at least {min_recommended} are recommended for "
                f"num_block_rows={num_block_rows}."
            )

        _logger.info(
            f"Running N4SID: {len(data)} rows, "
            f"y={obs_cols}, u={input_cols}, "
            f"num_block_rows={num_block_rows}, rank={self.latent_dim}"
        )

        nfoursid = NFourSID(
            data,
            output_columns=obs_cols,
            input_columns=input_cols if input_cols else None,
            num_block_rows=num_block_rows,
        )
        nfoursid.subspace_identification()
        state_space, _ = nfoursid.system_identification(rank=self.latent_dim)

        _logger.info(
            f"N4SID identified: A{state_space.a.shape}, "
            f"B{state_space.b.shape}, C{state_space.c.shape}, "
            f"D{state_space.d.shape}"
        )

        # Convert to float32 tensors
        A = torch.tensor(state_space.a, dtype=torch.float32)
        C = torch.tensor(state_space.c, dtype=torch.float32)

        # Split the combined B/D matrices into control (B, D) and disturbance (E)
        B_ctrl, E_dist, D_ctrl = None, None, None
        if state_space.u_dim > 0:
            B_full = torch.tensor(state_space.b, dtype=torch.float32)
            D_full = torch.tensor(state_space.d, dtype=torch.float32)

            if self.control_dim > 0:
                B_ctrl = B_full[:, :self.control_dim]
                D_ctrl = D_full[:, :self.control_dim]

            if self.disturbance_dim > 0:
                E_dist = B_full[:, self.control_dim: self.control_dim + self.disturbance_dim]

        self._apply_n4sid_matrices(A, B_ctrl, C, E_dist, D_ctrl)
        _logger.info("N4SID matrix initialization applied to model parameters.")

    def initialize_matrices_from_sippy(
        self,
        training_set,
        id_method='N4SID',
        ss_f=15,
        ss_fixed_order=None,
        ss_threshold=0.1,
        centering='MeanVal',
        **kwargs,
    ):
        """
        Identify initial state-space matrices from data using the SIPPY package.

        Uses ``sippy_unipi.system_identification`` to fit a state-space model,
        then copies the identified matrices into the model parameters via
        :meth:`_apply_n4sid_matrices`.

        The combined B matrix returned by SIPPY is split at ``control_dim``
        so that the first ``control_dim`` columns initialize B (control) and
        the remaining columns initialize E (disturbance).

        Args:
            training_set: List of DataFrames. Must contain all columns in
                ``observation_features`` (or ``target_features``) plus all
                ``control_features`` and ``disturbance_features``.
            id_method: SIPPY identification method, e.g. 'N4SID', 'CVA',
                       'MOESP' (default: 'N4SID').
            ss_f: Number of future steps (horizon) for the Hankel matrix
                  (default: 15).
            ss_fixed_order: Fixed state-space order. If None, uses
                            ``self.latent_dim`` (default: None).
            ss_threshold: Threshold for automatic order selection when
                          ``ss_fixed_order`` is None (default: 0.1).
            centering: Data centering method, e.g. 'MeanVal', 'InitVal',
                       'None' (default: 'MeanVal').
            **kwargs: Additional keyword arguments forwarded to
                      ``sippy_unipi.system_identification``.
        """
        import logging as _logging
        import numpy as _np
        import pandas as pd
        from sippy_unipi import system_identification

        _logger = _logging.getLogger(__name__)

        # Determine observation columns
        obs_cols = (
            self.observation_features
            if self.observation_features is not None
            else self.target_features
        )
        if obs_cols is None:
            raise ValueError(
                "observation_features or target_features must be set for SIPPY "
                "initialization. Provide at least one as a list of column names."
            )

        input_cols = list(self.control_features) + list(self.disturbance_features)

        all_cols = obs_cols + input_cols
        data = pd.concat(training_set, axis=0)[all_cols].dropna()

        if len(data) == 0:
            raise ValueError(
                "No valid data rows remain after dropping NaNs. "
                "SIPPY initialization requires at least some non-NaN rows."
            )

        _logger.info(
            f"Running SIPPY ({id_method}): {len(data)} rows, "
            f"y={obs_cols}, u={input_cols}, "
            f"ss_f={ss_f}, order={ss_fixed_order or self.latent_dim}"
        )

        y = data[obs_cols].to_numpy(dtype=float).T   # (y_dim, T)
        u = data[input_cols].to_numpy(dtype=float).T if input_cols else None  # (u_dim, T)

        order = ss_fixed_order if ss_fixed_order is not None else self.latent_dim

        model = system_identification(
            y, u, id_method,
            centering=centering,
            tsample=1.0,
            SS_f=ss_f,
            SS_fixed_order=order,
            SS_threshold=ss_threshold,
            **kwargs,
        )

        _logger.info(
            f"SIPPY identified: A{model.A.shape}, B{model.B.shape}, "
            f"C{model.C.shape}, D{model.D.shape}"
        )

        A = torch.tensor(model.A, dtype=torch.float32)
        C = torch.tensor(model.C, dtype=torch.float32)

        B_ctrl, E_dist, D_ctrl = None, None, None
        if u is not None and model.B is not None:
            B_full = torch.tensor(model.B, dtype=torch.float32)
            D_full = torch.tensor(model.D, dtype=torch.float32)

            if self.control_dim > 0:
                B_ctrl = B_full[:, :self.control_dim]
                D_ctrl = D_full[:, :self.control_dim]

            if self.disturbance_dim > 0:
                E_dist = B_full[:, self.control_dim: self.control_dim + self.disturbance_dim]

        self._apply_n4sid_matrices(A, B_ctrl, C, E_dist, D_ctrl)
        _logger.info("SIPPY matrix initialization applied to model parameters.")

    def _apply_n4sid_matrices(self, A, B, C, E, D):
        """
        Copy N4SID-identified matrices into model parameters.

        Handles the unconstrained case where A, B, C, E, and D are stored as
        direct ``nn.Parameter`` tensors.  Each matrix is only copied when the
        corresponding parameter name exists in ``self._parameters``, so
        subclasses that replace parameters with alternative parameterizations
        (e.g. :class:`BuildingDynamics`, which uses ``log_B``, ``log_C``,
        ``log_A_offdiag`` / ``log_A_diag``, ``log_E``) are unaffected — N4SID
        initialization is simply skipped for those parameters.

        Args:
            A: State transition matrix ``(latent_dim, latent_dim)``.
            B: Control matrix ``(latent_dim, control_dim)`` or ``None``.
            C: Observation matrix ``(measurement_dim, latent_dim)``.
            E: Disturbance matrix ``(latent_dim, disturbance_dim)`` or ``None``.
            D: Feedforward matrix ``(measurement_dim, control_dim)`` or ``None``.
        """
        with torch.no_grad():
            # A: stored as nn.Parameter 'A' in the base class
            if 'A' in self._parameters:
                self._parameters['A'].copy_(A)

            # B: stored as '_B'
            if B is not None and '_B' in self._parameters:
                self._parameters['_B'].copy_(B)

            # C: stored as '_C'
            if C is not None and '_C' in self._parameters:
                self._parameters['_C'].copy_(C)

            # E: stored as '_E'
            if E is not None and '_E' in self._parameters:
                self._parameters['_E'].copy_(E)

            # D: stored as '_D' (only present when use_feedforward=True)
            if D is not None and '_D' in self._parameters:
                self._parameters['_D'].copy_(D)


class BuildingDynamics(TorchLinearDynamicsWithKF):
    """
    Building dynamics model with physical constraints.
    
    Extends TorchLinearDynamicsWithKF with options to enforce physical constraints
    on state-space matrices, which is meaningful for building thermal models:
    - A off-diagonal: non-negative (A_ij = exp(log_A_offdiag_ij), thermal coupling between states)
    - A diagonal: in [off-diagonal row sum, 1] (sigmoid-scaled headroom, energy-dissipative or conservative)
    - B elements: positive (B = exp(log_B), control heat transfer coefficients)
    - E elements: positive (E = exp(log_E), disturbance heat transfer coefficients)
    - D elements: positive (D = exp(log_D), feedforward gains; only when use_feedforward=True)
    - C elements: constrained via constraint_C parameter:
        * 'softmax': row-wise softmax (each output is a convex combination of states, weights sum to 1)
        * 'positive': all elements positive via exp (sensor gains)
        * 'identity': fixed as identity matrix (C = I, not learnable)
    
    The A constraint guarantees: off-diagonal >= 0, diagonal >= off-diagonal row sum, diagonal <= 1.
    This ensures no single state can amplify unboundedly (diagonal <= 1) while also ensuring
    the diagonal is at least as large as the total outflow to neighbours (energy cannot exceed
    what is available in the state). Note: the full row sum is NOT constrained to <= 1;
    it equals diagonal + sum(off-diagonals) which can exceed 1 when off-diagonals are large.
    Uses exponential parametrization for B and E; sigmoid-scaled headroom for A diagonal.
    Softmax parametrization ensures each C row is interpretable as a state-weighting vector.
    """
    
    def __init__(
        self,
        latent_dim: int = None,
        control_features: List[str] = None,
        observation_features: Optional[List[str]] = None,
        disturbance_features: Optional[List[str]] = None,
        output_dim: int = None,
        positive_B: bool = True,
        constraint_C: str = 'softmax',
        positive_E: bool = True,
        positive_D: bool = False,
        constrained_A: bool = True,
        **kwargs
    ):
        """
        Args:
            control_features: List of control input feature names (controllable)
            disturbance_features: List of disturbance feature names (uncontrollable, e.g. weather)
            latent_dim: Dimension of latent state
            observation_features: List of measurement feature names (what C matrix observes).
                                 If None, uses target_features. Can be superset of targets.
            target_features: List of target feature names (what we predict/evaluate).
                            If None, uses output_dim.
            output_dim: Dimension of observations/targets (deprecated - use target_features)
            process_noise_cov: Initial process noise covariance
            measurement_noise_cov: Initial measurement noise covariance
            learnable_noise: If True, learn Q and R during training
            noise_constraint_type: Type of constraint for learnable noise covariances
            learning_rate: Learning rate for optimizer
            num_encoding_measurements: Number of recent measurements to use for encoding (1=last only, -1=all)
            positive_B: If True, parameterize B to ensure all elements are positive (B = exp(log_B))
            constraint_C: Type of constraint for C matrix. Options:
                         - 'positive': C = exp(log_C), ensures all elements are positive
                         - 'identity': C is fixed as identity matrix (not learnable)
                         - 'softmax': Apply row-wise softmax so each output is a weighted average of states (weights sum to 1)
            positive_E: If True, parameterize E to ensure all elements are positive (E = exp(log_E))
            positive_D: If True, parameterize D to ensure all elements are positive (D = exp(log_D)).
                        Only applies when use_feedforward=True. Default: False.
            constrained_A: If True, constrain A: off-diagonal >= 0 (via exp), diagonal in
                [off-diagonal row sum, 1] (via sigmoid-scaled headroom). This ensures the
                diagonal is always at least as large as the sum of outgoing couplings, and
                never exceeds 1. The full row sum is not constrained to <= 1.
        """
        # Store constraint flags before calling parent init
        self.positive_B = positive_B
        self.constraint_C = constraint_C
        self.positive_E = positive_E
        self.positive_D = positive_D
        self.constrained_A = constrained_A
        
        # Initialize parent class (creates A, B, C, E matrices)
        super().__init__(
            latent_dim=latent_dim,
            control_features=control_features,
            observation_features=observation_features,
            disturbance_features=disturbance_features,
            output_dim=output_dim,
            **kwargs
        )
        
        # Apply positive parametrization to B if requested
        if positive_B:
            # Get current B parameter and convert to positive parametrization
            if hasattr(self, '_B'):
                B_current = self._B.data
            else:
                B_current = self.B.data if hasattr(self, 'B') else torch.randn(latent_dim, len(control_features)) * 0.1
            
            # Remove old parameter
            if hasattr(self, '_B'):
                delattr(self, '_B')
            
            # Initialize with small positive values
            B_init = B_current.abs() + 0.01
            self.log_B = nn.Parameter(torch.log(B_init))
        
        # Apply positive parametrization to C if requested
        if constraint_C == 'identity':
            # Remove learnable C parameter and replace with fixed identity matrix
            if hasattr(self, '_C'):
                delattr(self, '_C')
            if hasattr(self, 'C'):
                delattr(self, 'C')
            # Register identity matrix as a buffer (not a parameter, so not learnable)
            # C shape depends on output_dim and latent_dim
            output_size = output_dim if output_dim is not None else len(observation_features)
            
            # Validate dimensions for identity constraint
            if output_size != latent_dim:
                warnings.warn(
                    f"constraint_C='identity' with observation dimension ({output_size}) "
                    f"!= latent dimension ({latent_dim}). "
                    f"This creates an unobservable system where {abs(latent_dim - output_size)} state(s) "
                    f"cannot be directly updated from measurements. "
                    f"Consider using constraint_C='positive' or 'softmax' for better observability.",
                    UserWarning,
                    stacklevel=2
                )
            
            identity_C = torch.eye(output_size, latent_dim)
            self.register_buffer('_C_identity', identity_C)
            
        elif constraint_C in ['positive', 'softmax']:
            # Get current C parameter and convert to positive parametrization
            if hasattr(self, '_C'):
                C_current = self._C.data
            else:
                C_current = self.C.data if hasattr(self, 'C') else torch.randn(latent_dim, len(observation_features)) * 0.1
            
            # Remove old parameter
            if hasattr(self, '_C'):
                delattr(self, '_C')
            
            if constraint_C == 'softmax':
                # Initialise logits to zero → uniform softmax weights (1/n_states per output).
                # This is the physically neutral prior: each sensor equally observes all states.
                # Training will discover the actual sparse mixing structure from this baseline.
                self.log_C = nn.Parameter(torch.zeros_like(C_current))
            elif constraint_C == 'positive':
                # For positive constraint, initialize with small positive values in log space
                C_init = C_current.abs() + 0.01
                self.log_C = nn.Parameter(torch.log(C_init))
        
        # Apply constrained parametrization to A if requested
        if constrained_A:
            # Get current A parameter
            A_current = self.A.data
            
            # Remove old parameter
            delattr(self, 'A')
            
            # Parametrization for A constraint:
            # - log_A_offdiag: off-diagonal params (applied as exp in _apply_A_constraint → positive)
            # - log_A_diag: pre-sigmoid diagonal fraction (applied as sigmoid in _apply_A_constraint)
            # Result: off-diag >= 0, diagonal in [off-diagonal row sum, 1]
            # The full row sum (diagonal + sum of off-diagonals) is NOT constrained to <= 1.
            
            # Extract current diagonal (as positive dissipation)
            A_diag = torch.diagonal(A_current).abs()
            A_offdiag = A_current.clone()
            A_offdiag.fill_diagonal_(0.0)  # Zero out diagonal
            A_offdiag = A_offdiag.abs() + 0.01  # Initialize positive
            
            # Compute what extra dissipation would be needed
            offdiag_sums = torch.sum(A_offdiag, dim=1)
            extra_dissipation = torch.clamp(A_diag - offdiag_sums, min=0.01)  # Must be ≥ 0
            
            # Create learnable parameters
            self.log_A_diag = nn.Parameter(torch.log(extra_dissipation))  # Extra dissipation term
            self.log_A_offdiag = nn.Parameter(torch.log(A_offdiag))  # Off-diagonal elements
    
        # Apply positive parametrization to E if requested
        if positive_E:
            # Get current E parameter and convert to positive parametrization
            if hasattr(self, '_E'):
                E_current = self._E.data
            else:
                E_current = self.E.data if hasattr(self, 'E') else torch.randn(latent_dim, len(disturbance_features)) * 0.1
            
            # Remove old parameter
            if hasattr(self, '_E'):
                delattr(self, '_E')
            
            # Initialize with small positive values
            E_init = E_current.abs() + 0.01
            self.log_E = nn.Parameter(torch.log(E_init))

        # Apply positive parametrization to D if requested (feedforward only)
        if positive_D and self.use_feedforward:
            # Get current D parameter and convert to positive parametrization
            if hasattr(self, '_D') and self._D is not None:
                D_current = self._D.data
            else:
                D_current = torch.randn(self.measurement_dim, self.control_dim) * 0.1

            # Remove old parameter
            if hasattr(self, '_D'):
                delattr(self, '_D')

            # Initialize with small positive values
            D_init = D_current.abs() + 0.01
            self.log_D = nn.Parameter(torch.log(D_init))

    # ------------------------------------------------------------------
    # Static constraint helpers
    # These encode the physical constraint math once, so both this static
    # LKF variant and the piecewise BuildingPWLKF can share the identical
    # transformation logic without duplication.
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_A_constraint(log_offdiag_flat, log_diag_fraction, latent_dim, device):
        """
        Apply building A matrix constraint from flat off-diagonal params.

        Constraints enforced:
          - Off-diagonal elements >= 0 (via exp)
          - Diagonal in [off-diagonal row sum, 1] (via sigmoid-scaled headroom)
        Note: the full row sum (diagonal + sum of off-diagonals) is NOT constrained
        to <= 1; it can exceed 1 when off-diagonal elements are large.
        Supports both unbatched (1D params) and batched (2D params) inputs.

        Args:
            log_offdiag_flat: Log off-diagonal params.
                - Unbatched: (latent_dim*(latent_dim-1),)
                - Batched:   (batch, latent_dim*(latent_dim-1))
            log_diag_fraction: Pre-sigmoid diagonal fraction params.
                - Unbatched: (latent_dim,)
                - Batched:   (batch, latent_dim)
            latent_dim: Latent state dimension.
            device: Torch device.

        Returns:
            A: Constrained matrix, (latent_dim, latent_dim) or (batch, latent_dim, latent_dim).
        """
        batched = log_offdiag_flat.ndim == 2
        if not batched:
            log_offdiag_flat = log_offdiag_flat.unsqueeze(0)
            log_diag_fraction = log_diag_fraction.unsqueeze(0)

        batch_size = log_offdiag_flat.shape[0]

        # Off-diagonal: non-negative via exp
        A_offdiag = torch.exp(log_offdiag_flat)  # (batch, offdiag_size)

        # Place off-diagonal values into full matrix (diagonal stays 0)
        A = torch.zeros(batch_size, latent_dim, latent_dim, device=device)
        mask = ~torch.eye(latent_dim, dtype=torch.bool, device=device)
        mask_expanded = mask.unsqueeze(0).expand(batch_size, -1, -1)
        A[mask_expanded] = A_offdiag.reshape(-1)

        # Row sums of off-diagonal elements only (diagonal is still 0)
        offdiag_row_sums = A.sum(dim=2)  # (batch, latent_dim)

        # Diagonal: sigmoid fraction scales available headroom [offdiag_sum, 1]
        extra_fraction = torch.sigmoid(log_diag_fraction)  # (batch, latent_dim) ∈ [0, 1]
        A_diag = offdiag_row_sums + extra_fraction * (1.0 - offdiag_row_sums)
        A_diag = torch.clamp(A_diag, min=0.0, max=1.0)

        diag_idx = torch.arange(latent_dim, device=device)
        A[:, diag_idx, diag_idx] = A_diag

        if not batched:
            A = A.squeeze(0)
        return A

    @staticmethod
    def _apply_B_constraint(raw_B):
        """Apply positive constraint to B: B = exp(raw_B)."""
        return torch.exp(raw_B)

    @staticmethod
    def _apply_E_constraint(raw_E):
        """Apply positive constraint to E: E = exp(raw_E)."""
        return torch.exp(raw_E)

    @staticmethod
    def _apply_D_constraint(raw_D):
        """Apply positive constraint to D: D = exp(raw_D)."""
        return torch.exp(raw_D)

    @staticmethod
    def _apply_C_constraint(raw_C, constraint_C):
        """
        Apply constraint to C matrix.

        Works with both 2D (output_dim, latent_dim) and 3D (batch, output_dim, latent_dim).

        Args:
            raw_C: Raw C values.
            constraint_C: 'positive' (exp) or 'softmax' (row-wise over latent states, dim=-1).
        """
        if constraint_C == 'positive':
            return torch.exp(raw_C)
        elif constraint_C == 'softmax':
            # dim=-1 is the latent dimension for both 2D and 3D inputs
            return torch.softmax(raw_C, dim=-1)
        else:
            raise ValueError(f"Unknown constraint_C: {constraint_C!r}")

    @property
    def A(self):
        """State transition matrix."""
        if hasattr(self, 'constrained_A') and self.constrained_A:
            device = self.log_A_offdiag.device
            mask = ~torch.eye(self.latent_dim, dtype=torch.bool, device=device)
            # Extract only the off-diagonal entries (diagonal entries of log_A_offdiag unused)
            log_offdiag_flat = self.log_A_offdiag[mask]  # (latent_dim*(latent_dim-1),)
            return BuildingDynamics._apply_A_constraint(
                log_offdiag_flat, self.log_A_diag, self.latent_dim, device
            )
        else:
            try:
                return self._parameters['A']
            except KeyError:
                raise AttributeError('A')

    @property
    def B(self):
        """Control influence matrix."""
        if self.positive_B:
            return BuildingDynamics._apply_B_constraint(self.log_B)
        else:
            return self._B

    @property
    def C(self):
        """Observation matrix."""
        if self.constraint_C == 'identity':
            return self._C_identity
        elif self.constraint_C in ('positive', 'softmax'):
            return BuildingDynamics._apply_C_constraint(self.log_C, self.constraint_C)
        else:
            return self._C

    @property
    def E(self):
        """Disturbance influence matrix."""
        if self.has_disturbances:
            if self.positive_E:
                return BuildingDynamics._apply_E_constraint(self.log_E)
            return self._E
        else:
            return None

    @property
    def D(self):
        """Control feedforward matrix (observation equation)."""
        if self.use_feedforward:
            if self.positive_D:
                return BuildingDynamics._apply_D_constraint(self.log_D)
            return self._D
        else:
            return None

    def _apply_n4sid_matrices(self, A, B, C, E, D):
        """
        Copy subspace-identified matrices into BuildingDynamics constrained parameters.

        Each matrix is mapped to its log-parameterization via the inverse of the
        corresponding constraint transform:

        - ``constrained_A``: off-diagonal → ``log_A_offdiag = log(|A_offdiag| + ε)``;
          diagonal → ``log_A_diag = logit(clamp((diag − Σoffdiag)/(1 − Σoffdiag), ε, 1−ε))``.
          Warns when the identified diagonal is outside ``[Σoffdiag, 1]``.
        - ``positive_B``: ``log_B = log(|B| + ε)``
        - ``positive_E``: ``log_E = log(|E| + ε)``
        - ``positive_D``: ``log_D = log(|D| + ε)`` (only when ``use_feedforward=True``)
        - ``constraint_C in ('positive', 'softmax')``: ``log_C = log(|C| + ε)``
        - ``constraint_C == 'identity'``: skipped (fixed buffer, not learnable)
        - Any flag that is ``False`` falls through to the base-class implementation,
          which copies into the unconstrained ``nn.Parameter`` (``_B``, ``_C``, etc.).

        Args:
            A: State transition matrix ``(latent_dim, latent_dim)``.
            B: Control matrix ``(latent_dim, control_dim)`` or ``None``.
            C: Observation matrix ``(measurement_dim, latent_dim)``.
            E: Disturbance matrix ``(latent_dim, disturbance_dim)`` or ``None``.
            D: Feedforward matrix ``(measurement_dim, control_dim)`` or ``None``.
        """
        import logging as _logging
        import math as _math
        _logger = _logging.getLogger(__name__)
        eps = 1e-6

        with torch.no_grad():
            # --- A matrix ---
            if self.constrained_A and A is not None:
                A_np = A if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32)
                n = self.latent_dim
                device = self.log_A_offdiag.device
                mask = ~torch.eye(n, dtype=torch.bool, device=device)

                # Off-diagonal: inverse of exp → log(|value| + ε)
                A_offdiag_vals = A_np[mask.cpu()].to(device)
                log_offdiag = torch.log(torch.abs(A_offdiag_vals) + eps)
                self.log_A_offdiag[mask] = log_offdiag

                # Diagonal: inverse of sigmoid-scaled headroom
                # forward: diag = offdiag_sum + sigmoid(log_A_diag) * (1 - offdiag_sum)
                offdiag_pos = torch.exp(log_offdiag)  # positive off-diag values
                # Rebuild per-row sums
                A_off_full = torch.zeros(n, n, device=device)
                A_off_full[mask] = offdiag_pos
                offdiag_sums = A_off_full.sum(dim=1)  # (n,)

                A_diag = torch.diagonal(A_np.to(device))
                headroom = 1.0 - offdiag_sums  # denominator

                # Check if diagonal is in valid range
                out_of_range = (A_diag < offdiag_sums) | (A_diag > 1.0)
                if out_of_range.any():
                    _logger.warning(
                        f"BuildingDynamics init: identified A diagonal has "
                        f"{out_of_range.sum().item()} element(s) outside "
                        f"[offdiag_sum, 1]. Values will be clamped."
                    )

                fraction = torch.clamp(
                    (A_diag - offdiag_sums) / torch.clamp(headroom, min=eps),
                    min=eps, max=1.0 - eps,
                )
                # logit = log(p / (1-p))
                self.log_A_diag.copy_(torch.log(fraction / (1.0 - fraction)))

            # --- B matrix ---
            if self.positive_B and B is not None:
                B_t = B if isinstance(B, torch.Tensor) else torch.tensor(B, dtype=torch.float32)
                self.log_B.copy_(torch.log(torch.abs(B_t) + eps))

            # --- C matrix ---
            if self.constraint_C in ('positive', 'softmax') and C is not None:
                C_t = C if isinstance(C, torch.Tensor) else torch.tensor(C, dtype=torch.float32)
                self.log_C.copy_(torch.log(torch.abs(C_t) + eps))
            # 'identity' is a fixed buffer — skip

            # --- E matrix ---
            if self.positive_E and E is not None:
                E_t = E if isinstance(E, torch.Tensor) else torch.tensor(E, dtype=torch.float32)
                self.log_E.copy_(torch.log(torch.abs(E_t) + eps))

            # --- D matrix ---
            if self.positive_D and self.use_feedforward and D is not None:
                D_t = D if isinstance(D, torch.Tensor) else torch.tensor(D, dtype=torch.float32)
                self.log_D.copy_(torch.log(torch.abs(D_t) + eps))

        # Fall through to base class for any unconstrained parameters
        # (e.g., when positive_B=False, _B exists and should be initialized directly)
        A_for_super = None if self.constrained_A else A
        B_for_super = None if self.positive_B else B
        C_for_super = None if self.constraint_C in ('positive', 'softmax', 'identity') else C
        E_for_super = None if self.positive_E else E
        D_for_super = None if self.positive_D else D
        super()._apply_n4sid_matrices(A_for_super, B_for_super, C_for_super, E_for_super, D_for_super)
