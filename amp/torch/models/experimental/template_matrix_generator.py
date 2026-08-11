"""
TEMPLATE: Custom Matrix Generator

Copy this file and fill in your physics equations to create a custom
structured PWLKF model.

Steps:
1. Copy this file to your project directory
2. Rename the class (MyCustomGenerator -> YourNameGenerator)
3. Fill in the TODO sections with your physics
4. Set up param_config with your parameters
5. Use it with StructuredPWLKF

Example usage after customization:
    from your_module import YourCustomGenerator, get_your_config
    from amp.torch.models.structured_pwlkf import StructuredPWLKF
    
    model = StructuredPWLKF(
        control_features=['u1', 'u2'],
        disturbance_features=['d1'],
        target_features=['y1'],
        latent_dim=YOUR_STATE_DIM,
        matrix_generator_class=YourCustomGenerator,
        param_config=get_your_config(),
    )
"""

import torch
from amp.torch.models.experimental.matrix_generator import BaseMatrixGenerator
from typing import Dict, Tuple


# ============================================================================
# STEP 1: Define your custom generator class
# ============================================================================

class MyCustomGenerator(BaseMatrixGenerator):
    """
    TODO: Add docstring describing your physical system.
    
    Example:
        Custom generator for [YOUR SYSTEM NAME].
        
        Physical system:
        - States: [describe your state variables]
        - Control: [describe control inputs]
        - Disturbance: [describe disturbances]
        
        Equations:
            [Write your differential equations here]
            dx1/dt = ...
            dx2/dt = ...
    """
    
    def _physics_to_matrices(
        self, 
        params: Dict[str, torch.Tensor], 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert physical parameters to state-space matrices.
        
        TODO: Implement your physics equations here.
        
        State-space form:
            dx/dt = A @ x + B @ u + E @ d
            y = C @ x
        
        Args:
            params: Dict of physical parameters (each is (batch,) tensor)
                   Keys match your param_config['learnable_params']
            batch_size: Batch size for matrix generation
            
        Returns:
            Tuple of (A, B, C, E) where:
                A: State transition matrix (batch, latent_dim, latent_dim)
                B: Control matrix (batch, latent_dim, control_dim)
                C: Observation matrix (batch, measurement_dim, latent_dim)
                E: Disturbance matrix (batch, latent_dim, disturbance_dim)
        """
        device = next(self.parameters()).device
        
        # TODO: Extract your physical parameters from params dict
        # Example:
        # R1 = params['R1']  # (batch,) tensor
        # R2 = params['R2']
        # C1 = params['C1']
        # C2 = params['C2']
        
        # TODO: Initialize matrices with correct dimensions
        # Replace these dimensions with your actual values:
        latent_dim = self.latent_dim
        control_dim = self.control_dim
        measurement_dim = self.measurement_dim
        disturbance_dim = self.disturbance_dim
        
        A = torch.zeros(batch_size, latent_dim, latent_dim, device=device)
        B = torch.zeros(batch_size, latent_dim, control_dim, device=device)
        C = torch.zeros(batch_size, measurement_dim, latent_dim, device=device)
        E = torch.zeros(batch_size, latent_dim, disturbance_dim, device=device)
        
        # TODO: Fill in matrix elements using your physics equations
        # Example for 2x2 A matrix:
        # A[:, 0, 0] = -1/(R1*C1) - 1/(R2*C1)
        # A[:, 0, 1] = 1/(R1*C1)
        # A[:, 1, 0] = 1/(R1*C2)
        # A[:, 1, 1] = -1/(R1*C2)
        
        # TODO: Fill B matrix (control inputs)
        # Example:
        # B[:, 0, 0] = 1/C1  # First control affects first state
        
        # TODO: Fill C matrix (observations)
        # Common case: observe all states directly
        # C = torch.eye(measurement_dim, latent_dim, device=device)
        # C = C.unsqueeze(0).expand(batch_size, -1, -1)
        
        # TODO: Fill E matrix (disturbances)
        # Example:
        # E[:, 0, 0] = 1/(R2*C1)  # First disturbance affects first state
        
        return A, B, C, E


# ============================================================================
# STEP 2: Define your parameter configuration
# ============================================================================

def get_my_custom_config() -> Dict:
    """
    TODO: Define configuration for your physical parameters.
    
    Returns:
        param_config dict with:
            - 'learnable_params': List of parameter names
            - 'constraints': Dict of parameter constraints
            - 'constants': Dict of fixed constants
    """
    return {
        # TODO: List all parameters your model should learn
        'learnable_params': [
            'param1',  # TODO: Replace with your parameter names
            'param2',
            'param3',
        ],
        
        # TODO: Define constraints for each parameter
        'constraints': {
            # Positive constraint: param > 0
            'param1': {'type': 'positive'},
            
            # Bounded constraint: lower <= param <= upper
            'param2': {'type': 'bounded', 'lower': 0.01, 'upper': 10.0},
            
            # Simplex constraint: params sum to target_sum
            # 'frac1': {
            #     'type': 'simplex',
            #     'group_params': ['frac1', 'frac2'],
            #     'target_sum': 1.0
            # },
            
            # No constraint (uncomment if needed):
            # 'param3': {},
        },
        
        # TODO: Define any fixed constants (not learned)
        'constants': {
            # 'air_density': 1.225,
            # 'building_area': 100.0,
        }
    }


# ============================================================================
# STEP 3: Create convenience function (optional)
# ============================================================================

def create_my_custom_model(**kwargs):
    """
    TODO: Convenience function to create your model.
    
    Args:
        **kwargs: Arguments for StructuredPWLKF
        
    Returns:
        StructuredPWLKF instance with your custom generator
        
    Example usage:
        model = create_my_custom_model(
            control_features=['u1'],
            disturbance_features=['d1'],
            target_features=['y1'],
            learning_rate=1e-3,
        )
    """
    from amp.torch.models.experimental.structured_pwlkf import StructuredPWLKF
    
    param_config = get_my_custom_config()
    
    # TODO: Set default values for your system
    defaults = {
        'latent_dim': 2,  # TODO: Your state dimension
        'matrix_generator_class': MyCustomGenerator,
        'param_config': param_config,
        'nn_hidden_dims': [8, 8],  # Small network is usually enough
    }
    
    # Merge with user-provided kwargs
    defaults.update(kwargs)
    
    return StructuredPWLKF(**defaults)


# ============================================================================
# STEP 4: Test your implementation (optional)
# ============================================================================

def test_my_custom_generator():
    """
    TODO: Test that your generator works correctly.
    
    Run this after implementing to verify everything works.
    """
    print("Testing MyCustomGenerator...")
    
    # Create model
    model = create_my_custom_model(
        control_features=['u1'],
        disturbance_features=['d1'],
        target_features=['y1'],
        observation_features=['y1'],
        output_dim=1,
    )
    
    print(f"✓ Model created successfully")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"  Control dim: {model.control_dim}")
    print(f"  Disturbance dim: {model.disturbance_dim}")
    
    # Test matrix generation
    d_t = torch.randn(5, 1)  # 5 samples, 1 disturbance
    matrices = model.inspect_matrices(d_t)
    
    print(f"✓ Matrix generation works")
    print(f"  A shape: {matrices['A'].shape}")
    print(f"  B shape: {matrices['B'].shape}")
    print(f"  C shape: {matrices['C'].shape}")
    print(f"  E shape: {matrices['E'].shape}")
    
    # Check parameters
    params = matrices['params']
    print(f"✓ Physical parameters:")
    for name, value in params.items():
        if name in get_my_custom_config()['learnable_params']:
            print(f"  {name}: {value[0].item():.4f}")
    
    # Test forward pass
    batch_size = 2
    history_len = 6
    forecast_len = 24
    
    batch = {
        'control_inputs': torch.randn(batch_size, model.control_dim, forecast_len),
        'disturbance_inputs': torch.randn(batch_size, model.disturbance_dim, forecast_len),
        'historical_inputs': torch.randn(batch_size, model.control_dim, history_len),
        'historical_disturbances': torch.randn(batch_size, model.disturbance_dim, history_len),
        'historical_measurements': torch.randn(batch_size, model.measurement_dim, history_len),
        'targets': torch.randn(batch_size, model.output_dim, forecast_len),
    }
    
    predictions, uncertainties = model(batch)
    print(f"✓ Forward pass works")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Uncertainties shape: {uncertainties.shape}")
    
    print("\n✓ All tests passed! Your generator is ready to use.")


# ============================================================================
# Example: Complete implementation for reference
# ============================================================================

class ExampleRCGenerator(BaseMatrixGenerator):
    """
    Example: Simple RC thermal network.
    
    States: [T_indoor]
    Control: [Q_heat]
    Disturbance: [T_outdoor]
    
    Physics:
        C * dT_indoor/dt = (T_outdoor - T_indoor) / R + Q_heat
    """
    
    def _physics_to_matrices(self, params, batch_size):
        device = next(self.parameters()).device
        
        R = params['R']
        C = params['C']
        
        # A matrix: -1/(R*C)
        A = torch.zeros(batch_size, 1, 1, device=device)
        A[:, 0, 0] = -1/(R*C)
        
        # B matrix: 1/C
        B = torch.zeros(batch_size, 1, 1, device=device)
        B[:, 0, 0] = 1/C
        
        # C matrix: observe temperature directly
        C_mat = torch.ones(batch_size, 1, 1, device=device)
        
        # E matrix: 1/(R*C)
        E = torch.zeros(batch_size, 1, 1, device=device)
        E[:, 0, 0] = 1/(R*C)
        
        return A, B, C_mat, E


def get_example_rc_config():
    """Example configuration for simple RC network."""
    return {
        'learnable_params': ['R', 'C'],
        'constraints': {
            'R': {'type': 'bounded', 'lower': 0.1, 'upper': 10.0},
            'C': {'type': 'bounded', 'lower': 10.0, 'upper': 1000.0},
        },
        'constants': {}
    }


# ============================================================================
# Main: Run tests if executed directly
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TEMPLATE: Custom Matrix Generator")
    print("="*70)
    print("\nThis is a template file. To use it:")
    print("1. Copy this file to your project")
    print("2. Fill in the TODO sections")
    print("3. Run test_my_custom_generator() to verify")
    print("4. Use with StructuredPWLKF in your training code")
    print("\nRunning example implementation test...\n")
    
    # Test the example implementation
    from amp.torch.models.experimental.structured_pwlkf import StructuredPWLKF
    
    model = StructuredPWLKF(
        control_features=['heat'],
        disturbance_features=['t_out'],
        target_features=['t_in'],
        observation_features=['t_in'],
        latent_dim=1,
        output_dim=1,
        matrix_generator_class=ExampleRCGenerator,
        param_config=get_example_rc_config(),
    )
    
    d_t = torch.tensor([[15.0]])
    inspection = model.inspect_matrices(d_t)
    
    print("Example RC Network Generator:")
    print(f"  R: {inspection['params']['R'].item():.4f}")
    print(f"  C: {inspection['params']['C'].item():.4f}")
    print(f"  A matrix: {inspection['A'].squeeze()}")
    print("\n✓ Example works! Now customize for your system.")
