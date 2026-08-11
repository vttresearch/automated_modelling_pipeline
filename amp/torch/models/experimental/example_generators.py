"""
Example matrix generators for StructuredPWLKF models.

This module provides reference implementations of physics-informed matrix generators
that can be used with StructuredPWLKF. These serve as examples for creating custom
generators for specific applications.

Examples:
    Using a pre-built generator:
        ```python
        from amp.torch.models.structured_pwlkf import StructuredPWLKF
        from amp.torch.models.example_generators import SimpleRCGenerator, get_simple_rc_config
        
        model = StructuredPWLKF(
            control_features=['heat_power'],
            disturbance_features=['t_outdoor'],
            target_features=['t_indoor'],
            latent_dim=1,
            matrix_generator_class=SimpleRCGenerator,
            param_config=get_simple_rc_config(),
        )
        ```
"""

import torch
from amp.torch.models.experimental.matrix_generator import BaseMatrixGenerator


# ============================================================================
# Simple RC Network (1 State)
# ============================================================================

class SimpleRCGenerator(BaseMatrixGenerator):
    """
    Simple RC thermal network generator.
    
    Physics Model:
        Single-zone building with:
        - One thermal mass (indoor air)
        - Thermal resistance to outdoor (R)
        - Heat capacity (C)
    
    State Variables:
        T_indoor: Indoor temperature [°C]
    
    Control Inputs:
        Q_heat: Heating power [W]
    
    Disturbances:
        T_outdoor: Outdoor temperature [°C]
    
    Governing Equation:
        C * dT_indoor/dt = (T_outdoor - T_indoor) / R + Q_heat
    
    Discretized State-Space:
        x[k+1] = A*x[k] + B*u[k] + E*d[k]
        y[k] = C*x[k]
    
    Where (for timestep dt):
        A = exp(-dt/(R*C))
        B = (1 - exp(-dt/(R*C))) / C
        C = 1 (observe temperature directly)
        E = (1 - exp(-dt/(R*C))) / C
    
    Parameters:
        R: Thermal resistance [K/W]
        C: Heat capacity [J/K]
    
    Usage:
        ```python
        from amp.torch.models.example_generators import SimpleRCGenerator, get_simple_rc_config
        from amp.torch.models.structured_pwlkf import StructuredPWLKF
        
        model = StructuredPWLKF(
            control_features=['heat_power'],
            disturbance_features=['t_outdoor'],
            target_features=['t_indoor'],
            observation_features=['t_indoor'],
            latent_dim=1,
            output_dim=1,
            matrix_generator_class=SimpleRCGenerator,
            param_config=get_simple_rc_config(),
        )
        ```
    """
    
    def _physics_to_matrices(self, params, batch_size):
        """Convert R, C parameters to discretized state-space matrices."""
        device = next(self.parameters()).device
        
        R = params['R']
        C = params['C']
        
        # Time constant
        tau = R * C
        
        # Discretization timestep (assuming 1 hour)
        dt = 1.0
        
        # A matrix: exp(-dt/tau)
        A = torch.zeros(batch_size, 1, 1, device=device)
        A[:, 0, 0] = torch.exp(-dt / tau)
        
        # B matrix: (1 - exp(-dt/tau)) / C
        # This converts heat power to temperature change
        B = torch.zeros(batch_size, 1, 1, device=device)
        B[:, 0, 0] = (1 - torch.exp(-dt / tau)) / C
        
        # C matrix: observe temperature directly
        C_mat = torch.ones(batch_size, 1, 1, device=device)
        
        # E matrix: (1 - exp(-dt/tau)) / C
        # Effect of outdoor temperature
        E = torch.zeros(batch_size, 1, 1, device=device)
        E[:, 0, 0] = (1 - torch.exp(-dt / tau)) / C
        
        return A, B, C_mat, E


def get_simple_rc_config():
    """
    Get parameter configuration for SimpleRCGenerator.
    
    Returns:
        dict: Parameter configuration with learnable parameters and constraints
        
    Example:
        ```python
        config = get_simple_rc_config()
        model = StructuredPWLKF(
            ...,
            matrix_generator_class=SimpleRCGenerator,
            param_config=config,
        )
        ```
    """
    return {
        'learnable_params': ['R', 'C'],
        'constraints': {
            'R': {'type': 'bounded', 'lower': 0.5, 'upper': 10.0},
            'C': {'type': 'bounded', 'lower': 10.0, 'upper': 1000.0},
        },
        'constants': {}
    }


# ============================================================================
# Two-State RC Network (2R2C)
# ============================================================================

class TwoStateRCGenerator(BaseMatrixGenerator):
    """
    Two-state RC network (2R2C) generator.
    
    Physics Model:
        Two-zone building with:
        - Indoor zone with thermal mass C1
        - Building envelope with thermal mass C2
        - Thermal resistance between zones (R1)
        - Thermal resistance to outdoor (R2)
    
    State Variables:
        T_indoor: Indoor temperature [°C]
        T_envelope: Envelope temperature [°C]
    
    Control Inputs:
        Q_heat: Heating power [W]
    
    Disturbances:
        T_outdoor: Outdoor temperature [°C]
    
    Governing Equations:
        C1 * dT_indoor/dt = (T_envelope - T_indoor) / R1 + Q_heat
        C2 * dT_envelope/dt = (T_outdoor - T_envelope) / R2 + (T_indoor - T_envelope) / R1
    
    Continuous-time A matrix:
        A_cont = [[-1/(R1*C1),        1/(R1*C1)    ],
                  [ 1/(R1*C2),   -(1/R1 + 1/R2)/C2 ]]
    
    Parameters:
        R1: Indoor-envelope thermal resistance [K/W]
        R2: Envelope-outdoor thermal resistance [K/W]
        C1: Indoor heat capacity [J/K]
        C2: Envelope heat capacity [J/K]
    
    Usage:
        ```python
        from amp.torch.models.example_generators import TwoStateRCGenerator, get_two_state_rc_config
        from amp.torch.models.structured_pwlkf import StructuredPWLKF
        
        model = StructuredPWLKF(
            control_features=['heat_power'],
            disturbance_features=['t_outdoor'],
            target_features=['t_indoor', 't_envelope'],
            observation_features=['t_indoor', 't_envelope'],
            latent_dim=2,
            output_dim=2,
            matrix_generator_class=TwoStateRCGenerator,
            param_config=get_two_state_rc_config(),
        )
        ```
    """
    
    def _physics_to_matrices(self, params, batch_size):
        """Convert 2R2C parameters to state-space matrices."""
        device = next(self.parameters()).device
        
        R1 = params['R1']  # Indoor-envelope resistance
        R2 = params['R2']  # Envelope-outdoor resistance
        C1 = params['C1']  # Indoor capacitance
        C2 = params['C2']  # Envelope capacitance
        
        dt = 1.0  # 1 hour timestep
        
        # Continuous-time A matrix
        A_cont = torch.zeros(batch_size, 2, 2, device=device)
        A_cont[:, 0, 0] = -1 / (R1 * C1)
        A_cont[:, 0, 1] = 1 / (R1 * C1)
        A_cont[:, 1, 0] = 1 / (R1 * C2)
        A_cont[:, 1, 1] = -(1/R1 + 1/R2) / C2
        
        # Discretize using matrix exponential (approximate with Taylor series)
        # For small dt: exp(A*dt) ≈ I + A*dt + (A*dt)^2/2
        I = torch.eye(2, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        A_dt = A_cont * dt
        A = I + A_dt + torch.bmm(A_dt, A_dt) / 2.0
        
        # B matrix: effect of heating on indoor temperature
        B = torch.zeros(batch_size, 2, 1, device=device)
        B[:, 0, 0] = dt / C1
        
        # C matrix: observe both states
        C_mat = torch.eye(2, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # E matrix: effect of outdoor temperature
        E = torch.zeros(batch_size, 2, 1, device=device)
        E[:, 1, 0] = dt / (R2 * C2)
        
        return A, B, C_mat, E


def get_two_state_rc_config():
    """
    Get parameter configuration for TwoStateRCGenerator.
    
    Returns:
        dict: Parameter configuration with learnable parameters and constraints
        
    Example:
        ```python
        config = get_two_state_rc_config()
        model = StructuredPWLKF(
            ...,
            matrix_generator_class=TwoStateRCGenerator,
            param_config=config,
        )
        ```
    """
    return {
        'learnable_params': ['R1', 'R2', 'C1', 'C2'],
        'constraints': {
            'R1': {'type': 'bounded', 'lower': 0.1, 'upper': 5.0},
            'R2': {'type': 'bounded', 'lower': 0.1, 'upper': 5.0},
            'C1': {'type': 'bounded', 'lower': 50.0, 'upper': 500.0},
            'C2': {'type': 'bounded', 'lower': 50.0, 'upper': 500.0},
        },
        'constants': {}
    }


# ============================================================================
# Equation-Based RC Network (2R2C) - String Equations
# ============================================================================

class EquationBasedRCGenerator(BaseMatrixGenerator):
    """
    2R2C generator using equation strings instead of explicit physics implementation.
    
    This demonstrates the equation-based approach where physics is defined via string
    equations that are automatically parsed and converted to state-space matrices.
    
    Physics Model:
        Same as TwoStateRCGenerator, but defined using equation strings
    
    State Variables:
        T_in: Indoor temperature [°C]
        T_env: Envelope temperature [°C]
    
    Control Inputs:
        Q_heat: Heating power [W]
    
    Disturbances:
        T_out: Outdoor temperature [°C]
    
    Governing Equations (as strings):
        dT_in/dt = (T_env - T_in) / (R1 * C1) + Q_heat / C1
        dT_env/dt = (T_out - T_env) / (R2 * C2) + (T_in - T_env) / (R1 * C2)
    
    Note:
        This class doesn't override _physics_to_matrices(). Instead, it relies on
        the equation_config passed during initialization. The base class will use
        _equations_to_matrices() method to parse the strings and generate matrices.
    
    Usage:
        ```python
        from amp.torch.models.example_generators import (
            EquationBasedRCGenerator, 
            get_equation_based_rc_config
        )
        from amp.torch.models.structured_pwlkf import StructuredPWLKF
        
        config = get_equation_based_rc_config()
        param_config = config.copy()
        equation_config = param_config.pop('equation_config')
        
        model = StructuredPWLKF(
            control_features=['Q_heat'],
            disturbance_features=['T_out'],
            target_features=['T_in', 'T_env'],
            observation_features=['T_in', 'T_env'],
            latent_dim=2,
            output_dim=2,
            matrix_generator_class=EquationBasedRCGenerator,
            param_config=param_config,
            equation_config=equation_config,
        )
        ```
    """
    
    # Note: This class doesn't override _physics_to_matrices
    # The base class will use _equations_to_matrices() based on equation_config
    pass


def get_equation_based_rc_config():
    """
    Get parameter and equation configuration for EquationBasedRCGenerator.
    
    Returns:
        dict: Configuration with learnable parameters, constraints, and equation strings
        
    Example:
        ```python
        config = get_equation_based_rc_config()
        param_config = config.copy()
        equation_config = param_config.pop('equation_config')
        
        model = StructuredPWLKF(
            ...,
            matrix_generator_class=EquationBasedRCGenerator,
            param_config=param_config,
            equation_config=equation_config,
        )
        ```
    """
    return {
        'learnable_params': ['R1', 'R2', 'C1', 'C2'],
        'constraints': {
            'R1': {'type': 'bounded', 'lower': 0.1, 'upper': 5.0},
            'R2': {'type': 'bounded', 'lower': 0.1, 'upper': 5.0},
            'C1': {'type': 'bounded', 'lower': 50.0, 'upper': 500.0},
            'C2': {'type': 'bounded', 'lower': 50.0, 'upper': 500.0},
        },
        'constants': {},
        'equation_config': {
            'equations': {
                'T_in': '(T_env - T_in) / (R1 * C1) + Q_heat / C1',
                'T_env': '(T_out - T_env) / (R2 * C2) + (T_in - T_env) / (R1 * C2)'
            },
            'control_vars': ['Q_heat'],
            'disturbance_vars': ['T_out'],
        }
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def list_example_generators():
    """
    List all available example generators with descriptions.
    
    Returns:
        dict: Mapping of generator names to their classes and descriptions
        
    Example:
        ```python
        from amp.torch.models.example_generators import list_example_generators
        
        generators = list_example_generators()
        for name, info in generators.items():
            print(f"{name}: {info['description']}")
        ```
    """
    return {
        'SimpleRCGenerator': {
            'class': SimpleRCGenerator,
            'config_func': get_simple_rc_config,
            'description': 'Single-zone RC thermal network (1 state)',
            'states': 1,
            'parameters': ['R', 'C'],
        },
        'TwoStateRCGenerator': {
            'class': TwoStateRCGenerator,
            'config_func': get_two_state_rc_config,
            'description': '2R2C thermal network with envelope (2 states)',
            'states': 2,
            'parameters': ['R1', 'R2', 'C1', 'C2'],
        },
        'EquationBasedRCGenerator': {
            'class': EquationBasedRCGenerator,
            'config_func': get_equation_based_rc_config,
            'description': '2R2C network using string equations (2 states)',
            'states': 2,
            'parameters': ['R1', 'R2', 'C1', 'C2'],
            'note': 'Uses equation strings instead of explicit matrix generation',
        },
    }
