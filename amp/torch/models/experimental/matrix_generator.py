"""
Base classes for physics-informed matrix generators in PyTorch.

This module provides flexible matrix generation for state-space models where
matrices (A, B, C, E) can be:
1. Explicitly coded via override methods (for known physics)
2. Defined via string equations (inspired by legacy SymPy approach)
3. Generated from neural network parameters with physical constraints

Key Features:
- Parameter constraints (positive, bounded, simplex)
- Batch support for time-varying matrices
- Integration with PWLKF and other Kalman filter models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
import re
import logging

logger = logging.getLogger(__name__)


class ParameterConstraint:
    """
    Defines constraints for physical parameters.
    
    Supports:
    - 'positive': Ensures parameter > 0 using softplus
    - 'bounded': Constrains parameter to [lower, upper] using sigmoid
    - 'simplex': Ensures set of parameters sum to constant using softmax
    - 'fixed': Parameter is constant (not learnable)
    """
    
    CONSTRAINT_TYPES = ['positive', 'bounded', 'simplex', 'fixed']
    
    def __init__(self, constraint_type: str, **kwargs):
        """
        Initialize parameter constraint.
        
        Args:
            constraint_type: One of CONSTRAINT_TYPES
            **kwargs: 
                For 'bounded': lower (float), upper (float)
                For 'simplex': target_sum (float), group_params (list of param names)
                For 'fixed': value (float)
        """
        if constraint_type not in self.CONSTRAINT_TYPES:
            raise ValueError(f"constraint_type must be one of {self.CONSTRAINT_TYPES}")
        
        self.type = constraint_type
        self.kwargs = kwargs
        
        # Validate required kwargs
        if constraint_type == 'bounded':
            if 'lower' not in kwargs or 'upper' not in kwargs:
                raise ValueError("'bounded' constraint requires 'lower' and 'upper'")
        elif constraint_type == 'simplex':
            if 'target_sum' not in kwargs or 'group_params' not in kwargs:
                raise ValueError("'simplex' constraint requires 'target_sum' and 'group_params'")
        elif constraint_type == 'fixed':
            if 'value' not in kwargs:
                raise ValueError("'fixed' constraint requires 'value'")
    
    def apply(self, raw_value: torch.Tensor) -> torch.Tensor:
        """
        Apply constraint transformation to raw parameter value.
        
        Args:
            raw_value: Unconstrained parameter value from NN
            
        Returns:
            Constrained parameter value
        """
        if self.type == 'positive':
            return torch.nn.functional.softplus(raw_value)
        
        elif self.type == 'bounded':
            lb = self.kwargs['lower']
            ub = self.kwargs['upper']
            return torch.sigmoid(raw_value) * (ub - lb) + lb
        
        elif self.type == 'fixed':
            # Return constant value, detach from computation graph
            return torch.full_like(raw_value, self.kwargs['value']).detach()
        
        elif self.type == 'simplex':
            # Note: Simplex constraint is applied to groups, not individual values
            # This will be handled in the parent class
            return raw_value
        
        else:
            return raw_value


class BaseMatrixGenerator(nn.Module, ABC):
    """
    Abstract base class for generating state-space matrices.
    
    Supports two modes:
    1. Override `_physics_to_matrices()` method in child class
    2. Provide equation strings via `equation_config` (parsed dynamically)
    
    The generator:
    - Takes disturbances as input
    - Outputs physical parameters via neural network
    - Converts parameters to matrices A, B, C, E
    
    Matrices have shape (batch, rows, cols) for time-varying dynamics.
    
    Matrix dimensions are automatically inferred from the structured_model instance,
    not passed as explicit arguments.
    """
    
    def __init__(
        self,
        structured_model,
        param_config: Dict,
        nn_hidden_dims: Optional[List[int]] = None,
        nn_activation: str = 'relu',
        equation_config: Optional[Dict] = None,
    ):
        """
        Initialize base matrix generator.
        
        Args:
            structured_model: The StructuredPWLKF model instance to get dimensions from
            param_config: Configuration dict with:
                - 'learnable_params': List of parameter names to learn
                - 'constraints': Dict mapping param names to ParameterConstraint
                - 'constants': Dict of fixed constant values (not learned)
            nn_hidden_dims: Hidden layer dimensions for parameter network
                           Default: [2*latent_dim, 2*latent_dim]
            nn_activation: Activation function ('relu', 'tanh', 'elu')
            equation_config: Optional dict for string-based equation approach:
                - 'equations': Dict mapping state names to equation strings
                - 'control_vars': List of control variable names
                - 'disturbance_vars': List of disturbance variable names
        """
        super().__init__()
        
        # Extract dimensions from structured model
        self.disturbance_dim = structured_model.disturbance_dim
        self.latent_dim = structured_model.latent_dim
        self.control_dim = structured_model.control_dim
        self.observation_dim = structured_model.measurement_dim
        
        # Parameter configuration
        self.learnable_params = param_config['learnable_params']
        self.n_params = len(self.learnable_params)
        self.param_constraints = param_config.get('constraints', {})
        self.constants = param_config.get('constants', {})
        
        # Neural network configuration
        if nn_hidden_dims is None:
            nn_hidden_dims = [2 * self.latent_dim, 2 * self.latent_dim]
        self.nn_hidden_dims = nn_hidden_dims
        
        # Get activation function
        activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'gelu': nn.GELU,
        }.get(nn_activation.lower(), nn.ReLU)
        
        # Build neural network for parameter generation
        self.param_nn = self._build_mlp(
            input_dim=self.disturbance_dim,
            output_dim=self.n_params,
            hidden_dims=nn_hidden_dims,
            activation_fn=activation_fn
        )
        
        # Equation-based approach (optional)
        self.equation_config = equation_config
        if equation_config is not None:
            self._setup_equation_parser(equation_config)
    
    def _build_mlp(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int], 
        activation_fn
    ) -> nn.Sequential:
        """
        Build MLP for parameter generation.
        
        Args:
            input_dim: Input dimension (disturbance_dim)
            output_dim: Output dimension (n_params)
            hidden_dims: List of hidden layer dimensions
            activation_fn: Activation function class
            
        Returns:
            MLP module
        """
        layers = []
        current_dim = input_dim
        
        # Hidden layers with activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn())
            current_dim = hidden_dim
        
        # Output layer (no activation - raw parameters)
        layers.append(nn.Linear(current_dim, output_dim))
        
        mlp = nn.Sequential(*layers)
        
        # Initialize with small weights for stability
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        
        return mlp
    
    def _setup_equation_parser(self, equation_config: Dict):
        """
        Setup equation parsing for string-based matrix generation.
        
        This method prepares structures for parsing equation strings
        similar to the legacy SymPy approach, but using PyTorch operations.
        
        Args:
            equation_config: Dict with 'equations', 'control_vars', 'disturbance_vars'
        """
        self.state_vars = list(equation_config['equations'].keys())
        self.control_vars = equation_config.get('control_vars', [])
        self.disturbance_vars = equation_config.get('disturbance_vars', [])
        self.equations = equation_config['equations']
        
        # Validate dimensions
        if len(self.state_vars) != self.latent_dim:
            raise ValueError(
                f"Number of state equations ({len(self.state_vars)}) "
                f"must match latent_dim ({self.latent_dim})"
            )
        
        logger.info(f"Equation-based matrix generator initialized with states: {self.state_vars}")
    
    def forward(self, d_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate state-space matrices from disturbances.
        
        Args:
            d_t: Disturbances at time t (batch_size, disturbance_dim)
            
        Returns:
            Tuple of (A, B, C, E) matrices:
                A: State transition (batch, latent_dim, latent_dim)
                B: Control matrix (batch, latent_dim, control_dim)
                C: Observation matrix (batch, observation_dim, latent_dim)
                E: Disturbance matrix (batch, latent_dim, disturbance_dim)
        """
        batch_size = d_t.shape[0]
        
        # Step 1: Generate raw parameters from neural network
        raw_params = self.param_nn(d_t)  # (batch, n_params)
        
        # Step 2: Apply constraints to get physical parameters
        physical_params = self._apply_constraints(raw_params, batch_size)
        
        # Step 3: Convert physical parameters to matrices
        if self.equation_config is not None:
            # Use equation-based approach
            A, B, C, E = self._equations_to_matrices(physical_params, batch_size)
        else:
            # Use override method approach
            A, B, C, E = self._physics_to_matrices(physical_params, batch_size)
        
        return A, B, C, E
    
    def _apply_constraints(
        self, 
        raw_params: torch.Tensor, 
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Apply physical constraints to raw NN parameters.
        
        Handles individual constraints (positive, bounded, fixed) and
        group constraints (simplex).
        
        Args:
            raw_params: Raw parameters from NN (batch, n_params)
            batch_size: Batch size for fixed parameters
            
        Returns:
            Dict mapping parameter names to constrained values (batch,)
        """
        constrained = {}
        simplex_groups = {}  # Track simplex groups
        
        # Apply individual constraints
        for i, param_name in enumerate(self.learnable_params):
            raw_val = raw_params[:, i]  # (batch,)
            
            if param_name in self.param_constraints:
                constraint = self.param_constraints[param_name]
                
                if isinstance(constraint, dict):
                    # Convert dict to ParameterConstraint object
                    # Use .get() to avoid modifying the original dict
                    constraint_dict = constraint.copy()
                    constraint = ParameterConstraint(
                        constraint_dict.pop('type'), 
                        **constraint_dict
                    )
                
                if constraint.type == 'simplex':
                    # Store for group processing
                    group_name = tuple(constraint.kwargs['group_params'])
                    if group_name not in simplex_groups:
                        simplex_groups[group_name] = []
                    simplex_groups[group_name].append((param_name, raw_val))
                else:
                    # Apply individual constraint
                    constrained[param_name] = constraint.apply(raw_val)
            else:
                # No constraint - use raw value
                constrained[param_name] = raw_val
        
        # Apply simplex constraints to groups
        for group_params, param_list in simplex_groups.items():
            # Get constraint config (same for all in group)
            first_param = param_list[0][0]
            constraint = self.param_constraints[first_param]
            target_sum = constraint.kwargs['target_sum']
            
            # Stack raw values
            raw_vals = torch.stack([val for _, val in param_list], dim=1)  # (batch, group_size)
            
            # Apply softmax to ensure they sum to target_sum
            constrained_vals = torch.softmax(raw_vals, dim=1) * target_sum
            
            # Assign back to dict
            for i, (param_name, _) in enumerate(param_list):
                constrained[param_name] = constrained_vals[:, i]
        
        # Add constants (not learned, just expanded to batch)
        for const_name, const_value in self.constants.items():
            constrained[const_name] = torch.full(
                (batch_size,), 
                float(const_value), 
                dtype=raw_params.dtype, 
                device=raw_params.device
            )
        
        return constrained
    
    def _physics_to_matrices(
        self, 
        params: Dict[str, torch.Tensor], 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert physical parameters to state-space matrices.
        
        This method should be overridden in child classes to implement
        specific physics-based matrix construction. If not overridden,
        equation_config must be provided to use equation-based approach.
        
        Args:
            params: Dict of physical parameters (each is (batch,) tensor)
            batch_size: Batch size
            
        Returns:
            Tuple of (A, B, C, E) matrices
        
        Example for 2R2C thermal model:
            ```python
            R1 = params['R1']
            R2 = params['R2']
            C1 = params['C1']
            C2 = params['C2']
            
            A = torch.zeros(batch_size, self.latent_dim, self.latent_dim)
            A[:, 0, 0] = -1/(R1*C1) - 1/(R2*C1)
            A[:, 0, 1] = 1/(R1*C1)
            # ... etc
            
            return A, B, C, E
            ```
        """
        raise NotImplementedError(
            "Must implement _physics_to_matrices() or provide equation_config. "
            "If using equation-based approach, pass equation_config during initialization."
        )
    
    def _equations_to_matrices(
        self, 
        params: Dict[str, torch.Tensor], 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert equation strings to state-space matrices.
        
        This implements a simplified version of the legacy SymPy approach,
        using PyTorch operations for differentiability.
        
        Equation format: 'dx/dt = (x2 - x1) / R1 + u1 / C1'
        
        The method:
        1. Parses each equation to identify coefficients
        2. Builds matrices row by row based on coefficient extraction
        
        Args:
            params: Dict of physical parameters
            batch_size: Batch size
            
        Returns:
            Tuple of (A, B, C, E) matrices
        """
        device = next(self.parameters()).device
        
        # Initialize matrices
        A = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=device)
        B = torch.zeros(batch_size, self.latent_dim, self.control_dim, device=device)
        C = torch.eye(self.observation_dim, self.latent_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        E = torch.zeros(batch_size, self.latent_dim, self.disturbance_dim, device=device)
        
        # Parse each equation
        for state_idx, state_name in enumerate(self.state_vars):
            equation_str = self.equations[state_name]
            
            # Extract coefficients for this row
            row_coeffs = self._parse_equation(equation_str, params)
            
            # Fill in A matrix (state terms)
            for i, state_var in enumerate(self.state_vars):
                if state_var in row_coeffs:
                    A[:, state_idx, i] = row_coeffs[state_var]
            
            # Fill in B matrix (control terms)
            for i, control_var in enumerate(self.control_vars):
                if control_var in row_coeffs:
                    B[:, state_idx, i] = row_coeffs[control_var]
            
            # Fill in E matrix (disturbance terms)
            for i, dist_var in enumerate(self.disturbance_vars):
                if dist_var in row_coeffs:
                    E[:, state_idx, i] = row_coeffs[dist_var]
        
        return A, B, C, E
    
    def _parse_equation(
        self, 
        equation_str: str, 
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Parse equation string to extract coefficients for each variable using SymPy.
        
        This implements the legacy SymPy-based approach, converting symbolic
        coefficients to PyTorch tensors for differentiability.
        
        Example: '(T2 - T1) / (R1 * C1) + Q_heat / C1'
        -> {'T1': -1/(R1*C1), 'T2': 1/(R1*C1), 'Q_heat': 1/C1}
        
        Args:
            equation_str: Equation as string (right-hand side of dx/dt = ...)
            params: Physical parameters for evaluation (PyTorch tensors)
            
        Returns:
            Dict mapping variable names to their coefficients (PyTorch tensors)
        """
        try:
            import sympy
        except ImportError:
            logger.error(
                "SymPy is required for equation parsing. "
                "Install with: pip install sympy"
            )
            return {}
        
        # Collect all variable names from the equation
        all_vars = self.state_vars + self.control_vars + self.disturbance_vars
        
        # Parse equation string to identify all symbols
        split_chars = ['\\+', '\\-', '\\*', '\\/', '\\(', '\\)']
        potential_symbols = [s.strip() for s in re.split('|'.join(split_chars), equation_str)]
        potential_symbols = [s for s in potential_symbols if s and not s.replace('.', '').isdigit()]
        
        # Create SymPy symbols for all variables and parameters
        symbol_dict = {}
        for name in set(potential_symbols):
            symbol_dict[name] = sympy.symbols(name)
        
        # Evaluate the equation string to create SymPy expression
        try:
            expr = sympy.sympify(equation_str, locals=symbol_dict)
            expr = sympy.expand(expr)
        except Exception as e:
            logger.error(f"Failed to parse equation '{equation_str}': {e}")
            return {}
        
        # Extract coefficients for each variable
        coeffs = {}
        for var_name in all_vars:
            if var_name not in symbol_dict:
                continue
            
            var_symbol = symbol_dict[var_name]
            
            # Get all terms containing this variable
            coeff_expr = 0
            for term in expr.as_ordered_terms():
                if var_symbol in term.free_symbols:
                    # Extract the coefficient by substituting var=1
                    coeff_expr += term.subs(var_symbol, 1)
            
            if coeff_expr != 0:
                # Simplify the coefficient expression
                coeff_expr = sympy.simplify(coeff_expr)
                
                # Convert SymPy expression to PyTorch tensor
                # Replace parameter symbols with their PyTorch values
                coeff_tensor = self._sympy_to_torch(coeff_expr, params, symbol_dict)
                
                if coeff_tensor is not None:
                    coeffs[var_name] = coeff_tensor
        
        return coeffs
    
    def _sympy_to_torch(
        self, 
        sympy_expr, 
        params: Dict[str, torch.Tensor],
        symbol_dict: Dict
    ) -> Optional[torch.Tensor]:
        """
        Convert SymPy expression to PyTorch tensor while preserving gradients.
        
        Evaluates the symbolic expression by substituting parameter values
        using PyTorch operations to maintain differentiability.
        
        Args:
            sympy_expr: SymPy expression
            params: Dict of parameter tensors (must maintain gradients)
            symbol_dict: Dict mapping names to SymPy symbols
            
        Returns:
            PyTorch tensor with evaluated expression, or None if conversion fails
        """
        try:
            import sympy
        except ImportError:
            return None
        
        # Get all symbols in the expression
        symbols_in_expr = sympy_expr.free_symbols
        
        if not symbols_in_expr:
            # Constant expression
            return torch.tensor(float(sympy_expr), device=next(self.parameters()).device)
        
        # Build substitution dict: symbol -> tensor value
        subs_dict = {}
        for symbol in symbols_in_expr:
            symbol_name = str(symbol)
            if symbol_name in params:
                subs_dict[symbol] = params[symbol_name]
        
        # Check if all symbols can be substituted
        if len(subs_dict) != len(symbols_in_expr):
            missing = [str(s) for s in symbols_in_expr if str(s) not in params]
            logger.warning(
                f"Cannot evaluate expression - missing parameters: {missing}"
            )
            return None
        
        # Convert to numerical function using 'torch' module for differentiability
        # Get ordered list of symbols
        symbols_list = list(subs_dict.keys())
        
        # Create lambdified function that uses PyTorch operations
        try:
            # Use 'torch' module to maintain gradients
            func = sympy.lambdify(symbols_list, sympy_expr, modules=['torch', 'numpy'])
        except Exception as e:
            logger.warning(f"Failed to lambdify expression with torch: {e}")
            return None
        
        # Get parameter values in correct order (keep as tensors!)
        param_values = [subs_dict[sym] for sym in symbols_list]
        
        # Evaluate function with tensor inputs (preserves gradients)
        try:
            result = func(*param_values)
            # Ensure result is a tensor
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, device=param_values[0].device)
            return result
        except Exception as e:
            logger.warning(f"Failed to evaluate expression with tensors: {e}")
            return None
    
    def get_parameter_dict(self, d_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get constrained physical parameters for given disturbances.
        
        Useful for inspection and debugging.
        
        Args:
            d_t: Disturbances (batch, disturbance_dim)
            
        Returns:
            Dict of parameter names to values (batch,)
        """
        batch_size = d_t.shape[0]
        raw_params = self.param_nn(d_t)
        return self._apply_constraints(raw_params, batch_size)