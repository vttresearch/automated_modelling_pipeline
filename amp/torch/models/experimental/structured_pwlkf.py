"""
Structured PWLKF - Piecewise Linear Kalman Filter with physics-informed matrices.

This extends PWLKF to use physics-informed matrix generators instead of
unconstrained neural networks. Matrices are generated from physical parameters
that follow known equations or constraints.

⚠️  CRITICAL: StructuredPWLKF is FEATURE-ORDER DEPENDENT!
Unlike base PWLKF, the order of features matters because matrix indices
correspond to feature positions in the physical equations.

Example usage:
    ```python
    from amp.torch.models.structured_pwlkf import StructuredPWLKF
    from amp.torch.models.matrix_generator import BaseMatrixGenerator
    
    # Define your custom matrix generator (see template_matrix_generator.py)
    class MyRCGenerator(BaseMatrixGenerator):
        def _physics_to_matrices(self, params, batch_size):
            # Convert R, C to A, B, C, E matrices
            ...
    
    # Define physical parameter configuration
    param_config = {
        'learnable_params': ['R1', 'R2', 'C1', 'C2'],
        'constraints': {
            'R1': {'type': 'positive'},
            'R2': {'type': 'positive'},
            'C1': {'type': 'positive'},
            'C2': {'type': 'positive'},
        },
        'constants': {}
    }
    
    # Create model
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_out'],
        target_features=['t_indoor'],
        latent_dim=2,
        matrix_generator_class=MyRCGenerator,
        param_config=param_config,
    )
    ```
"""

import torch
import torch.nn as nn
from typing import Type, Dict, Optional, List
from amp.torch.models.experimental.pwlkf import PWLKF
from amp.torch.models.experimental.matrix_generator import BaseMatrixGenerator
import logging

logger = logging.getLogger(__name__)


def validate_feature_order_compatibility(
    control_features: List[str],
    disturbance_features: List[str],
    observation_features: Optional[List[str]],
    equation_config: Optional[Dict] = None,
) -> None:
    """
    Validate feature order consistency before model creation.
    
    Use this function to check feature ordering BEFORE instantiating StructuredPWLKF.
    Helps catch ordering errors early.
    
    Args:
        control_features: Control feature names
        disturbance_features: Disturbance feature names  
        observation_features: Observation/state feature names (optional)
        equation_config: Equation configuration dict (optional)
        
    Raises:
        ValueError: If feature orders are inconsistent
        
    Example:
        ```python
        # Check ordering before creating model
        validate_feature_order_compatibility(
            control_features=['Q_heat'],
            disturbance_features=['T_out'],
            observation_features=['T_in', 'T_env'],
            equation_config=my_equations
        )
        
        # If no error, proceed with model creation
        model = StructuredPWLKF(...)
        ```
    """
    if equation_config is None:
        logger.warning(
            "No equation_config provided - cannot validate feature ordering. "
            "Ensure your custom matrix generator uses features in the order provided."
        )
        return
    
    # Check control variables
    if 'control_vars' in equation_config:
        control_vars = equation_config['control_vars']
        if control_features != control_vars:
            raise ValueError(
                f"Control feature order mismatch!\n"
                f"  control_features: {control_features}\n"
                f"  equation control_vars: {control_vars}\n\n"
                f"StructuredPWLKF requires exact order matching."
            )
    
    # Check disturbance variables
    if 'disturbance_vars' in equation_config:
        dist_vars = equation_config['disturbance_vars']
        if disturbance_features != dist_vars:
            raise ValueError(
                f"Disturbance feature order mismatch!\n"
                f"  disturbance_features: {disturbance_features}\n"
                f"  equation disturbance_vars: {dist_vars}\n\n"
                f"StructuredPWLKF requires exact order matching."
            )
    
    # Check state variables
    if observation_features is not None and 'equations' in equation_config:
        state_vars = list(equation_config['equations'].keys())
        if observation_features != state_vars:
            raise ValueError(
                f"Observation feature order mismatch!\n"
                f"  observation_features: {observation_features}\n"
                f"  equation state_vars: {state_vars}\n\n"
                f"StructuredPWLKF requires exact order matching."
            )
    
    logger.info("✓ Feature ordering validated successfully")


class StructuredPWLKF(PWLKF):
    """
    PWLKF with physics-informed structured matrix generation.
    
    Instead of a generic MLP generating matrix elements, this uses
    a physics-informed generator that:
    1. Learns physical parameters (R, C, etc.)
    2. Applies physical constraints
    3. Converts parameters to matrices using known equations
    
    This provides:
    - Better interpretability (can inspect physical parameters)
    - Stronger inductive bias (physics-informed structure)
    - Fewer learnable parameters
    - Better generalization
    """
    
    def __init__(
        self,
        control_features: list,
        disturbance_features: list,
        param_config: Dict,
        matrix_generator_class: Type[BaseMatrixGenerator],
        latent_dim: int = 2,
        observation_features: list = None,
        target_features: list = None,
        output_dim: int = 1,
        process_noise_cov: float = 1e-3,
        measurement_noise_cov: float = 1e-2,
        learnable_noise: bool = True,
        noise_constraint_type: str = 'diagonal_positive',
        learning_rate: float = 1e-3,
        num_encoding_measurements: int = 1,
        # Matrix generator parameters
        nn_hidden_dims: list = None,
        nn_activation: str = 'relu',
        equation_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize Structured PWLKF.
        
        Args:
            control_features: List of control input feature names
            disturbance_features: List of disturbance feature names (REQUIRED)
            param_config: Configuration for physical parameters:
                - 'learnable_params': List of parameter names to learn
                - 'constraints': Dict mapping param names to constraint configs
                - 'constants': Dict of fixed constant values
            matrix_generator_class: Class for matrix generator 
                (subclass of BaseMatrixGenerator)
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
            num_encoding_measurements: Number of recent measurements for encoding
            nn_hidden_dims: Hidden layer dimensions for parameter network.
                           Default: [2*latent_dim, 2*latent_dim]
            nn_activation: Activation function ('relu', 'tanh', 'elu')
            equation_config: Optional dict for string-based equation approach
            **kwargs: Additional arguments passed to PWLKF parent class
        """
        # Store config before parent init
        self.param_config = param_config
        self.matrix_generator_class = matrix_generator_class
        self.equation_config = equation_config
        
        # Store NN config for matrix generator
        self._mg_nn_hidden_dims = nn_hidden_dims
        self._mg_nn_activation = nn_activation
        
        # Initialize parent PWLKF
        # Note: All matrices must be dynamic in StructuredPWLKF (no static parameters)
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
            nn_hidden_dims=nn_hidden_dims,
            nn_activation=nn_activation,
            dynamic_matrices=['A', 'B', 'C', 'E'],  # All matrices generated by physics model
            **kwargs
        )
        
        # Replace parent's generic matrix_generator with structured one
        self.matrix_generator = self._create_structured_generator()
        
        # Clear unused matrix generators from parent PWLKF
        # (StructuredPWLKF uses a single physics-informed generator instead)
        self.matrix_generators.clear()
        self.static_matrices.clear()
        
        # Delete unused static parameters from grandparent LinearKalmanFilter
        # These are created in LKF.__init__ but never used in PWLKF/StructuredPWLKF
        if hasattr(self, 'A'):
            delattr(self, 'A')
        if hasattr(self, '_B'):
            delattr(self, '_B')
        if hasattr(self, '_C'):
            delattr(self, '_C')
        if hasattr(self, '_E'):
            delattr(self, '_E')
        
        # Validate feature ordering (CRITICAL for structured models!)
        self._validate_feature_order()
        
        # Update hyperparameters
        self.save_hyperparameters()
        
        logger.info(
            f"StructuredPWLKF initialized with {self.matrix_generator.__class__.__name__}, "
            f"learning {len(param_config['learnable_params'])} physical parameters"
        )
    
    def _create_structured_generator(self) -> BaseMatrixGenerator:
        """
        Create physics-informed matrix generator.
        
        Matrix dimensions are automatically inferred from self (the structured model).
        
        Returns:
            Instance of matrix_generator_class
        """
        generator = self.matrix_generator_class(
            structured_model=self,
            param_config=self.param_config,
            nn_hidden_dims=self._mg_nn_hidden_dims,
            nn_activation=self._mg_nn_activation,
            equation_config=self.equation_config,
        )
        
        return generator
    
    def _validate_feature_order(self):
        """
        Validate feature ordering consistency between model and matrix generator.
        
        CRITICAL: StructuredPWLKF is feature-order dependent!
        Matrix indices directly correspond to feature positions.
        
        Raises:
            ValueError: If feature orders don't match between model and equations
            UserWarning: If using observation_features without explicit validation
        """
        mg = self.matrix_generator
        
        # Check if matrix generator has equation-based variable lists
        if hasattr(mg, 'control_vars') and mg.control_vars:
            # Validate control features match equation control_vars
            if self.control_features != mg.control_vars:
                raise ValueError(
                    f"Control feature order mismatch!\n"
                    f"  Model control_features: {self.control_features}\n"
                    f"  Equation control_vars:  {mg.control_vars}\n\n"
                    f"StructuredPWLKF requires exact feature order matching."
                )
        
        if hasattr(mg, 'disturbance_vars') and mg.disturbance_vars:
            # Validate disturbance features match equation disturbance_vars
            if self.disturbance_features != mg.disturbance_vars:
                raise ValueError(
                    f"Disturbance feature order mismatch!\n"
                    f"  Model disturbance_features: {self.disturbance_features}\n"
                    f"  Equation disturbance_vars:  {mg.disturbance_vars}\n\n"
                    f"StructuredPWLKF requires exact feature order matching."
                )
        
        if hasattr(mg, 'state_vars') and mg.state_vars:
            # Check observation features if provided
            if self.observation_features is not None:
                if self.observation_features != mg.state_vars:
                    raise ValueError(
                        f"Observation feature order mismatch!\n"
                        f"  Model observation_features: {self.observation_features}\n"
                        f"  Equation state_vars:        {mg.state_vars}\n\n"
                        f"StructuredPWLKF requires exact feature order matching."
                    )
        
        # Log successful validation
        logger.info("✓ Feature ordering validated successfully")
        logger.debug(f"  States: {getattr(mg, 'state_vars', 'N/A')}")
        logger.debug(f"  Controls: {getattr(mg, 'control_vars', self.control_features)}")
        logger.debug(f"  Disturbances: {getattr(mg, 'disturbance_vars', self.disturbance_features)}")
    
    def get_physical_parameters(self, d_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get physical parameters for given disturbances.
        
        Useful for model inspection and validation.
        
        Args:
            d_t: Disturbances (batch, disturbance_dim)
            
        Returns:
            Dict of parameter names to constrained values
            
        Example:
            ```python
            d_t = torch.randn(10, 1)  # 10 samples, 1 disturbance
            params = model.get_physical_parameters(d_t)
            print(f"R1 values: {params['R1']}")  # (10,) tensor
            ```
        """
        return self.matrix_generator.get_parameter_dict(d_t)
    
    def _generate_matrices(self, d_t: torch.Tensor) -> tuple:
        """
        Generate matrices using physics-informed generator.
        
        Override parent's method because our matrix_generator returns
        matrices directly as a tuple (A, B, C, E), not as a flattened tensor
        that needs splitting.
        
        Args:
            d_t: Disturbances at time t (batch_size, disturbance_dim)
            
        Returns:
            Tuple of (A, B, C, E) matrices:
                A: State transition matrix (batch, latent_dim, latent_dim)
                B: Control matrix (batch, latent_dim, control_dim)
                C: Observation matrix (batch, measurement_dim, latent_dim)
                E: Disturbance matrix (batch, latent_dim, disturbance_dim)
        """
        return self.matrix_generator(d_t)
    
    def inspect_matrices(self, d_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate and return all matrices for inspection.
        
        Args:
            d_t: Disturbances (batch, disturbance_dim)
            
        Returns:
            Dict with keys 'A', 'B', 'C', 'E', 'params'
            
        Example:
            ```python
            d_t = torch.tensor([[15.0]])  # Outdoor temp = 15°C
            matrices = model.inspect_matrices(d_t)
            print("A matrix:", matrices['A'])
            print("Physical params:", matrices['params'])
            ```
        """
        A, B, C, E = self.matrix_generator(d_t)
        params = self.get_physical_parameters(d_t)
        
        return {
            'A': A,
            'B': B,
            'C': C,
            'E': E,
            'params': params
        }
