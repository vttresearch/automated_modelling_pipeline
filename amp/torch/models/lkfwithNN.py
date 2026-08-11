"""
Linear Kalman Filter with Neural Network correction term.

The NN adds a learned correction to the state transition based on disturbances.

Key difference from standard LKF:
- Adds NN(d_t) correction term to state dynamics
- Two-stage training: warmup state-space matrices, then train NN
- Optional: freeze matrices after warmup or keep them trainable
"""

import torch
import torch.nn as nn
from amp.torch.models.lkf import TorchLinearDynamicsWithKF, BuildingDynamics


class LKFWithNNCorrection:
    """
    Linear Kalman Filter with neural network correction to the state transition.
    
    Key feature:
    - Adds NN correction term to state transition dynamics. The term is a function of disturbance inputs.
    - Training is modified so that there is a configuration to set the state space warmup period.
        After the warmup period, the NN correction term training starts, and the matrices are frozen.
    
    State space model:
        x[t+1] = A @ x[t] + B @ u[t] + E @ d[t] + NN(d[t]) + w[t]
        y[t] = C @ x[t] + v[t]
    
    Training stages:
    1. Warmup stage (0 to ss_warmup_epochs): Train only state-space matrices (A, B, C, E, Q, R)
    2. NN training stage (ss_warmup_epochs to ss_warmup_epochs + nn_warmup_epochs): 
       Train NN correction, optionally freeze matrices
    3. Final stage: All stopping conditions can be met
    """
    
    def __init__(
        self,
        control_features: list,
        disturbance_features: list = None,
        nn_features: list = None,
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
        # NN correction parameters
        nn_hidden_dims: list = None,
        nn_activation: str = 'relu',
        ss_warmup_epochs: int = 100,
        nn_warmup_epochs: int = 100,
        freeze_matrices_after_warmup: bool = True,
        freeze_noise_parameters: bool = False,
        joint_finetune: bool = False,
        **kwargs
    ):
        """
        Args:
            control_features: List of control input feature names
            disturbance_features: List of disturbance feature names (NN input)
            latent_dim: Dimension of latent state
            observation_features: List of measurement feature names
            target_features: List of target feature names
            output_dim: Dimension of observations/targets
            process_noise_cov: Initial process noise covariance
            measurement_noise_cov: Initial measurement noise covariance
            learnable_noise: If True, learn Q and R during training
            noise_constraint_type: Type of constraint for learnable noise covariances.
                                  Options: 'cholesky', 'diagonal_positive', 'none'
            learning_rate: Learning rate for optimizer
            num_encoding_measurements: Number of recent measurements to use for encoding
            nn_features: Features fed to the NN correction term. When provided, overrides
                        disturbance_features as input to the correction NN; disturbance_features
                        still governs the E @ d term. When None, falls back to disturbance_features.
            nn_hidden_dims: Hidden layer dimensions for MLP correction. 
                           Default: [2*latent_dim, 2*latent_dim]
            nn_activation: Activation function ('relu', 'tanh', 'elu')
            ss_warmup_epochs: Number of epochs to train only state-space matrices
            nn_warmup_epochs: Number of epochs to train NN (after ss warmup)
            freeze_matrices_after_warmup: If True, freeze A,B,C,E after ss_warmup_epochs
            freeze_noise_parameters: If True, also freeze Q,R,P0 with matrices (default: False)
            joint_finetune: If True, stage 3 unfreezes both matrices and NN for joint fine-tuning.
                            Only meaningful when freeze_matrices_after_warmup=True.
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent class
        super().__init__(
            control_features=control_features,
            disturbance_features=disturbance_features,
            latent_dim=latent_dim,
            observation_features=observation_features,
            target_features=target_features,
            output_dim=output_dim,
            process_noise_cov=process_noise_cov,
            measurement_noise_cov=measurement_noise_cov,
            learnable_noise=learnable_noise,
            noise_constraint_type=noise_constraint_type,
            learning_rate=learning_rate,
            num_encoding_measurements=num_encoding_measurements,
            **kwargs
        )
        
        # Check that disturbances are provided
        if not self.has_disturbances:
            raise ValueError(
                "LKFWithNNCorrection requires disturbance_features for NN correction input. "
                "Provide disturbance_features argument."
            )
        
        # Determine which features feed the NN correction term.
        # When nn_features is explicitly provided it takes precedence; otherwise fall back
        # to disturbance_features so existing code continues to work unchanged.
        self.nn_features = nn_features if nn_features is not None else list(disturbance_features or [])
        self.nn_dim = len(self.nn_features)
        
        # Store NN configuration
        self.ss_warmup_epochs = ss_warmup_epochs
        self.nn_warmup_epochs = nn_warmup_epochs
        self.freeze_matrices_after_warmup = freeze_matrices_after_warmup
        self.freeze_noise_parameters = freeze_noise_parameters
        self.joint_finetune = joint_finetune
        
        # Default hidden dimensions: 2x latent_dim for each layer
        if nn_hidden_dims is None:
            nn_hidden_dims = [2 * latent_dim, 2 * latent_dim]
        
        # Build MLP for NN correction: disturbance_dim -> latent_dim
        # Input: d[t] (disturbances at time t)
        # Output: correction term added to state (same dimension as state)
        activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
        }.get(nn_activation.lower(), nn.ReLU)
        
        layers = []
        input_dim = self.nn_dim
        
        for hidden_dim in nn_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim
        
        # Final layer: hidden -> latent_dim (no activation)
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.correction_nn = nn.Sequential(*layers)
        
        # Initialize NN weights with small values
        for layer in self.correction_nn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)
        
        # Freeze NN during state-space warmup
        self._freeze_nn(True)
        
        # Track training stage
        self._matrices_frozen = False
        
        # Update hyperparameters
        self.save_hyperparameters()
    
    def _freeze_nn(self, freeze: bool):
        """Freeze or unfreeze the NN correction parameters."""
        for param in self.correction_nn.parameters():
            param.requires_grad = not freeze
    
    def _freeze_matrices(self, freeze: bool):
        """Freeze or unfreeze state-space matrix parameters."""
        # A, B, C, E may be properties computed from underlying parameters
        # (e.g. BuildingDynamics uses log_A_offdiag, log_B, etc.), so we must
        # operate on the actual nn.Parameter leaf variables, not the property results.

        # Collect correction-NN parameter ids to skip (managed by _freeze_nn)
        nn_param_ids = {id(p) for p in self.correction_nn.parameters()}

        # When freezing matrices but NOT noise parameters, collect noise param ids
        # so we can leave them trainable.
        noise_param_ids = set()
        if freeze and not self.freeze_noise_parameters and hasattr(self, 'kf'):
            for attr_name in ('log_Q_diag', 'log_R_diag', 'L_Q', 'L_R', 'log_P0_diag', 'Q_param', 'R_param'):
                attr = getattr(self.kf, attr_name, None)
                if isinstance(attr, nn.Parameter):
                    noise_param_ids.add(id(attr))

        for param in self.parameters():
            if id(param) in nn_param_ids:
                continue  # managed by _freeze_nn
            if id(param) in noise_param_ids:
                continue  # keep noise params trainable
            param.requires_grad_(not freeze)

        self._matrices_frozen = freeze
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch to manage warmup stages."""
        current_epoch = self.current_epoch
        
        # Stage 1: State-space warmup (train only matrices, NN frozen)
        if current_epoch < self.ss_warmup_epochs:
            if current_epoch == 0:
                print(f"\n{'='*70}")
                print(f"TRAINING STAGE 1: State-Space Warmup (Epochs 0-{self.ss_warmup_epochs-1})")
                print(f"  - Training: A, B, C, E, Q, R matrices")
                print(f"  - Frozen: NN correction")
                print(f"{'='*70}\n")
            self._freeze_nn(True)
            self._freeze_matrices(False)
        
        # Stage 2: NN warmup (train NN, optionally freeze matrices)
        elif current_epoch < self.ss_warmup_epochs + self.nn_warmup_epochs:
            if current_epoch == self.ss_warmup_epochs:
                print(f"\n{'='*70}")
                print(f"TRAINING STAGE 2: NN Warmup (Epochs {self.ss_warmup_epochs}-{self.ss_warmup_epochs + self.nn_warmup_epochs - 1})")
                print(f"  - Training: NN correction")
                if self.freeze_matrices_after_warmup:
                    print(f"  - Frozen: A, B, C, E, Q, R matrices")
                else:
                    print(f"  - Training: A, B, C, E, Q, R matrices (not frozen)")
                print(f"{'='*70}\n")
            self._freeze_nn(False)
            if self.freeze_matrices_after_warmup:
                self._freeze_matrices(True)
            else:
                self._freeze_matrices(False)
        
        # Stage 3: Joint fine-tuning or frozen-matrix continuation
        else:
            if current_epoch == self.ss_warmup_epochs + self.nn_warmup_epochs:
                print(f"\n{'='*70}")
                print(f"TRAINING STAGE 3: {'Joint Fine-Tuning' if self.joint_finetune else 'Normal Training'} (Epoch {current_epoch}+)")
                print(f"  - All stopping conditions enabled")
                if self.joint_finetune:
                    print(f"  - Training: All parameters (NN + matrices)")
                elif self.freeze_matrices_after_warmup:
                    print(f"  - Training: NN correction")
                    print(f"  - Frozen: A, B, C, E, Q, R matrices")
                else:
                    print(f"  - Training: All parameters")
                print(f"{'='*70}\n")
            if self.joint_finetune:
                self._freeze_nn(False)
                self._freeze_matrices(False)
    
    def _compute_state_mean_with_nn(self, x_mean, u_t, A, B, d_t, E, nn_input_t=None):
        """
        Compute predicted state mean with NN correction.
        
        x[t+1] = A @ x[t] + B @ u[t] + E @ d[t] + NN(nn_input_t)
        
        Args:
            x_mean: Current state mean (batch, latent_dim)
            u_t: Control input at time t (batch, control_dim)
            A: State transition matrix (latent_dim, latent_dim)
            B: Control matrix (latent_dim, control_dim)
            d_t: Disturbance input at time t (batch, disturbance_dim)
            E: Disturbance matrix (latent_dim, disturbance_dim)
            nn_input_t: Optional explicit NN input at time t (batch, nn_dim).
                        Falls back to d_t when None.
            
        Returns:
            x_mean_pred: Predicted state mean (batch, latent_dim)
        """
        # Standard state transition
        x_mean_pred = self._compute_state_mean(x_mean, u_t, A, B, d_t, E)
        
        # Add NN correction; use nn_input_t if provided, else fall back to d_t
        nn_in = nn_input_t if nn_input_t is not None else d_t
        nn_correction = self.correction_nn(nn_in)  # (batch, latent_dim)
        x_mean_pred = x_mean_pred + nn_correction
        
        return x_mean_pred
    
    def encode_with_kf(self, control_inputs, measurements, disturbance_inputs=None,
                       nn_inputs=None, **kwargs):
        """
        Encode initial state using Kalman filter with measurements and NN correction.
        
        Override parent method to use NN-corrected state transition.
        
        Args:
            control_inputs: Historical control inputs (batch, control_dim, window)
            measurements: Historical observations (batch, output_dim, window)
            disturbance_inputs: Historical disturbances (batch, disturbance_dim, window) - required
            nn_inputs: Explicit NN inputs for the correction term (batch, nn_dim, window).
                       Falls back to disturbance_inputs when None (backward compatible).
            **kwargs: Additional model-specific parameters (for interface compatibility)
        """
        if disturbance_inputs is None:
            raise ValueError("LKFWithNNCorrection requires disturbance_inputs for NN correction")

        # Follow BuildingPWLKF pattern: explicit param with disturbance fallback
        nn_in = nn_inputs if nn_inputs is not None else disturbance_inputs
        
        import time
        start_time = time.time()
        
        batch_size = control_inputs.shape[0]
        window_size = control_inputs.shape[-1]
        device = control_inputs.device
        
        # Initialize state with zero mean and learnable covariance
        x_mean = torch.zeros(batch_size, self.latent_dim, device=device)
        x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        x_mean_traj = []
        x_cov_traj = []
        
        # Pre-expand matrices for batch operations
        A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)
        R_expanded = self.R.unsqueeze(0).expand(batch_size, -1, -1)
        I_expanded = torch.eye(self.latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Determine how many measurements to use
        if self.num_encoding_measurements == 1:
            # Use only the LAST measurement
            y_last = measurements[:, :, -1]
            x_mean, x_cov = self.kf.update(x_mean, x_cov, y_last, self.C,
                                          C_expanded=C_expanded, R_expanded=R_expanded, I_expanded=I_expanded)
            x_mean_traj.append(x_mean)
            x_cov_traj.append(x_cov)
        
        elif self.num_encoding_measurements == -1:
            # Use ALL measurements with sequential filtering
            for t in range(window_size):
                u_t = control_inputs[:, :, t]
                y_t = measurements[:, :, t]
                d_t = disturbance_inputs[:, :, t]
                nn_t = nn_in[:, :, t]
                
                # Predict state WITH NN CORRECTION
                x_mean_pred = self._compute_state_mean_with_nn(x_mean, u_t, self.A, self.B, d_t, self.E, nn_input_t=nn_t)
                x_cov_pred = self._compute_state_covariance(x_cov, A_expanded)
                
                # Update with measurement
                x_mean, x_cov = self.kf.update(x_mean_pred, x_cov_pred, y_t, self.C,
                                              C_expanded=C_expanded, R_expanded=R_expanded, I_expanded=I_expanded)
                x_mean_traj.append(x_mean)
                x_cov_traj.append(x_cov)
        
        else:
            # Use last N measurements
            num_steps = min(self.num_encoding_measurements, window_size)
            start_idx = window_size - num_steps
            
            for t in range(start_idx, window_size):
                u_t = control_inputs[:, :, t]
                y_t = measurements[:, :, t]
                d_t = disturbance_inputs[:, :, t]
                nn_t = nn_in[:, :, t]
                
                # Predict state WITH NN CORRECTION
                x_mean_pred = self._compute_state_mean_with_nn(x_mean, u_t, self.A, self.B, d_t, self.E, nn_input_t=nn_t)
                x_cov_pred = self._compute_state_covariance(x_cov, A_expanded)
                
                # Update with measurement
                x_mean, x_cov = self.kf.update(x_mean_pred, x_cov_pred, y_t, self.C,
                                              C_expanded=C_expanded, R_expanded=R_expanded, I_expanded=I_expanded)
                x_mean_traj.append(x_mean)
                x_cov_traj.append(x_cov)
        
        # Record timing
        elapsed = time.time() - start_time
        self.timing_stats['encode_with_kf']['total'] += elapsed
        self.timing_stats['encode_with_kf']['count'] += 1
        
        return torch.stack(x_mean_traj, dim=-1), torch.stack(x_cov_traj, dim=-1)
    
    def predict_with_kf(self, x_mean, x_cov, control_inputs, disturbance_inputs=None,
                        nn_inputs=None, **kwargs):
        """
        Predict forward using Kalman filter with NN correction.
        
        Override parent method to use NN-corrected state transition.
        
        Args:
            x_mean: Initial state mean (batch, latent_dim)
            x_cov: Initial state covariance (batch, latent_dim, latent_dim)
            control_inputs: Control inputs (batch, control_dim, time_steps)
            disturbance_inputs: Disturbance inputs (batch, disturbance_dim, time_steps) - required
            nn_inputs: Explicit NN inputs for the correction term (batch, nn_dim, time_steps).
                       Falls back to disturbance_inputs when None (backward compatible).
            **kwargs: Additional model-specific parameters (for interface compatibility)
        """
        if disturbance_inputs is None:
            raise ValueError("LKFWithNNCorrection requires disturbance_inputs for NN correction")

        # Follow BuildingPWLKF pattern: explicit param with disturbance fallback
        nn_in = nn_inputs if nn_inputs is not None else disturbance_inputs
        
        import time
        start_time = time.time()
        
        batch_size = control_inputs.shape[0]
        time_steps = control_inputs.shape[-1]
        
        # Pre-expand matrices
        A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)
        C_expanded = self.C.unsqueeze(0).expand(batch_size, -1, -1)
        R_diag = torch.diagonal(self.R, dim1=0, dim2=1).unsqueeze(0)
        
        predictions = []
        uncertainties = []
        
        for t in range(time_steps):
            u_t = control_inputs[:, :, t]
            d_t = disturbance_inputs[:, :, t]
            nn_t = nn_in[:, :, t]
            
            # Predict next state WITH NN CORRECTION
            x_mean = self._compute_state_mean_with_nn(x_mean, u_t, self.A, self.B, d_t, self.E, nn_input_t=nn_t)
            x_cov = self._compute_state_covariance(x_cov, A_expanded)
            
            # Predict observation (project to targets for output)
            y_pred = self._compute_observation(x_mean, project_to_targets=True)
            predictions.append(y_pred)
            
            # Compute uncertainty (project to targets for output)
            y_std = self._compute_observation_uncertainty(x_cov, C_expanded, R_diag, project_to_targets=True)
            uncertainties.append(y_std)
        
        # Stack results
        predictions = torch.stack(predictions, dim=-1)
        uncertainties = torch.stack(uncertainties, dim=-1)
        
        # Record timing
        elapsed = time.time() - start_time
        self.timing_stats['predict_with_kf']['total'] += elapsed
        self.timing_stats['predict_with_kf']['count'] += 1
        
        return predictions, uncertainties
    
    def predict_full_trajectory(self, control_inputs, historical_controls=None, historical_observations=None,
                disturbance_inputs=None, historical_disturbances=None,
                nn_inputs=None, historical_nn_inputs=None, **kwargs):
        """
        Predict full trajectory including historical reconstruction with NN correction.
        
        Override parent method to use NN-corrected state transition.
        
        Args:
            control_inputs: Control inputs for prediction (batch, control_dim, time_steps)
            historical_controls: Historical controls (batch, control_dim, window)
            historical_observations: Historical observations (batch, output_dim, window)
            disturbance_inputs: Disturbance inputs (batch, disturbance_dim, time_steps) - required
            historical_disturbances: Historical disturbances (batch, disturbance_dim, window) - required if historical data provided
            nn_inputs: Explicit NN inputs for forecast window (batch, nn_dim, time_steps).
                       Falls back to disturbance_inputs when None (backward compatible).
            historical_nn_inputs: Explicit NN inputs for history window (batch, nn_dim, window).
                                  Falls back to historical_disturbances when None (backward compatible).
            **kwargs: Additional model-specific parameters (for interface compatibility)
        """
        # Follow BuildingPWLKF pattern: explicit params with disturbance fallback
        nn_in = nn_inputs if nn_inputs is not None else disturbance_inputs
        hist_nn_in = historical_nn_inputs if historical_nn_inputs is not None else historical_disturbances

        historical_inputs = historical_controls
        historical_measurements = historical_observations
        if disturbance_inputs is None or (historical_inputs is not None and historical_disturbances is None):
            raise ValueError("LKFWithNNCorrection requires disturbance_inputs and historical_disturbances")
        
        batch_size = control_inputs.shape[0]
        device = control_inputs.device
        window_size = historical_inputs.shape[-1] if historical_inputs is not None else 0
        
        hist_predictions = []
        hist_uncertainties = []
        
        # Capture historical predictions
        if historical_inputs is not None and historical_measurements is not None:
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
                u_t = historical_inputs[:, :, t]
                y_t = historical_measurements[:, :, t]
                d_t = historical_disturbances[:, :, t]
                nn_t = hist_nn_in[:, :, t]
                
                # Predict state WITH NN CORRECTION
                x_mean = self._compute_state_mean_with_nn(x_mean, u_t, self.A, self.B, d_t, self.E, nn_input_t=nn_t)
                x_cov = self._compute_state_covariance(x_cov, A_expanded)
                
                # Compute predicted observation
                y_pred = self._compute_observation(x_mean, project_to_targets=True)
                y_std = self._compute_observation_uncertainty(x_cov, C_expanded, R_diag, project_to_targets=True)
                
                hist_predictions.append(y_pred)
                hist_uncertainties.append(y_std)
                
                # Update with measurement
                x_mean, x_cov = self.kf.update(x_mean, x_cov, y_t, self.C,
                                              C_expanded=C_expanded, R_expanded=R_expanded, I_expanded=I_expanded)
        else:
            x_mean = torch.zeros(batch_size, self.latent_dim, device=device)
            x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Predict forward
        predictions, uncertainties = self.predict_with_kf(
            x_mean, x_cov, control_inputs, disturbance_inputs, nn_inputs=nn_in, **kwargs
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
    

    def forward(self, control_inputs, historical_controls=None, historical_observations=None,
                disturbance_inputs=None, historical_disturbances=None,
                nn_inputs=None, historical_nn_inputs=None, **kwargs):
        """
        Forward pass: encode with measurements (if provided) then predict.

        Overrides parent to split nn_inputs (forecast window) from historical_nn_inputs
        (history window) before routing to encode_with_kf / predict_with_kf — the same
        separation the parent performs for disturbance_inputs vs historical_disturbances
        via positional args.  The parent passes **kwargs unchanged to both methods, so
        without this override encode_with_kf would receive the forecast-window nn_inputs
        tensor instead of the history-window one.

        Args:
            nn_inputs: NN correction inputs for the forecast window (batch, nn_dim, time_steps).
                       Falls back to disturbance_inputs inside predict_with_kf when None.
            historical_nn_inputs: NN correction inputs for the history window (batch, nn_dim, window).
                       Falls back to historical_disturbances inside encode_with_kf when None.
        """
        if historical_controls is not None and historical_observations is not None:
            x_mean_traj, x_cov_traj = self.encode_with_kf(
                historical_controls, historical_observations, historical_disturbances,
                nn_inputs=historical_nn_inputs, **kwargs
            )
            x_mean, x_cov = x_mean_traj[:, :, -1], x_cov_traj[:, :, :, -1]
        else:
            batch_size = control_inputs.shape[0]
            device = control_inputs.device
            x_mean = torch.zeros(batch_size, self.latent_dim, device=device)
            x_cov = self.kf.P0.unsqueeze(0).expand(batch_size, -1, -1).clone()

        predictions, uncertainties = self.predict_with_kf(
            x_mean, x_cov, control_inputs, disturbance_inputs,
            nn_inputs=nn_inputs, **kwargs
        )
        return predictions, uncertainties

    def training_step(self, batch, batch_idx):
        """Training step with nn_inputs support."""
        control_inputs = batch.get('control_inputs')
        disturbance_inputs = batch.get('disturbance_inputs')
        historical_controls = batch.get('historical_controls')
        historical_disturbances = batch.get('historical_disturbances')
        historical_observations = batch.get('historical_observations')
        nn_inputs = batch.get('nn_inputs')
        historical_nn_inputs = batch.get('historical_nn_inputs')
        targets = batch['targets']

        # Apply curriculum learning
        control_inputs, disturbance_inputs, targets = self.apply_curriculum_to_inputs(
            control_inputs, disturbance_inputs, targets
        )
        if nn_inputs is not None:
            curriculum_steps = targets.size(-1)
            nn_inputs = nn_inputs[..., :curriculum_steps]

        predictions, uncertainties = self(
            control_inputs, historical_controls, historical_observations,
            disturbance_inputs, historical_disturbances,
            nn_inputs=nn_inputs, historical_nn_inputs=historical_nn_inputs
        )

        losses = self._compute_loss(predictions, targets, uncertainties)
        self.log('train_loss', losses['nll'], prog_bar=True)
        self.log('train_nll', losses['nll'], prog_bar=True)
        self.log('train_mse', losses['mse'], prog_bar=True)
        if len(self.horizon_loss_windows) > 1:
            for key, value in losses.items():
                if key.startswith(('nll_horizon', 'mse_horizon', 'weighted_nll')):
                    self.log(f'train_{key}', value)
        return {'loss': losses['nll'], 'predictions': predictions,
                'targets': targets, 'uncertainties': uncertainties}

    def validation_step(self, batch, batch_idx):
        """Validation step with nn_inputs support."""
        control_inputs = batch.get('control_inputs')
        disturbance_inputs = batch.get('disturbance_inputs')
        historical_controls = batch.get('historical_controls')
        historical_disturbances = batch.get('historical_disturbances')
        historical_observations = batch.get('historical_observations')
        nn_inputs = batch.get('nn_inputs')
        historical_nn_inputs = batch.get('historical_nn_inputs')
        targets = batch['targets']

        control_inputs, disturbance_inputs, targets = self.apply_curriculum_to_inputs(
            control_inputs, disturbance_inputs, targets
        )
        if nn_inputs is not None:
            curriculum_steps = targets.size(-1)
            nn_inputs = nn_inputs[..., :curriculum_steps]

        predictions, uncertainties = self(
            control_inputs, historical_controls, historical_observations,
            disturbance_inputs, historical_disturbances,
            nn_inputs=nn_inputs, historical_nn_inputs=historical_nn_inputs
        )

        losses = self._compute_loss(predictions, targets, uncertainties)
        self.log('val_loss', losses['nll'], prog_bar=True)
        self.log('val_nll', losses['nll'], prog_bar=True)
        self.log('val_mse', losses['mse'], prog_bar=True)
        self.log('val_mean_uncertainty', uncertainties.mean())
        if len(self.horizon_loss_windows) > 1:
            for key, value in losses.items():
                if key.startswith(('nll_horizon', 'mse_horizon', 'weighted_nll')):
                    self.log(f'val_{key}', value)
        return {'loss': losses['nll'], 'predictions': predictions,
                'targets': targets, 'uncertainties': uncertainties}

    def test_step(self, batch, batch_idx):
        """Test step with nn_inputs support."""
        control_inputs = batch.get('control_inputs')
        disturbance_inputs = batch.get('disturbance_inputs')
        historical_controls = batch.get('historical_controls')
        historical_disturbances = batch.get('historical_disturbances')
        historical_observations = batch.get('historical_observations')
        nn_inputs = batch.get('nn_inputs')
        historical_nn_inputs = batch.get('historical_nn_inputs')
        targets = batch['targets']

        predictions, uncertainties = self(
            control_inputs, historical_controls, historical_observations,
            disturbance_inputs, historical_disturbances,
            nn_inputs=nn_inputs, historical_nn_inputs=historical_nn_inputs
        )

        losses = self._compute_loss(predictions, targets, uncertainties)
        self.log('test_loss', losses['nll'])
        self.log('test_nll', losses['nll'])
        self.log('test_mse', losses['mse'])
        self.log('test_mean_uncertainty', uncertainties.mean())
        if len(self.horizon_loss_windows) > 1:
            for key, value in losses.items():
                if key.startswith(('nll_horizon', 'mse_horizon', 'weighted_nll')):
                    self.log(f'test_{key}', value)
        return {'loss': losses['nll'], 'predictions': predictions,
                'targets': targets, 'uncertainties': uncertainties}


class GeneralLKFWithNNCorrection(LKFWithNNCorrection, TorchLinearDynamicsWithKF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BuildingLKFWithNNCorrection(LKFWithNNCorrection, BuildingDynamics):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)