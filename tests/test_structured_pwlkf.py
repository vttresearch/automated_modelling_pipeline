"""
Tests for StructuredPWLKF model with physics-informed matrix generators.

This test suite verifies that StructuredPWLKF can be instantiated with custom
matrix generators, properly validates feature ordering, and generates physically
meaningful matrices.
"""

import pytest
import torch
import numpy as np

# `lightning` is an optional extra (see extras_requirements.txt) required by
# amp.torch.models.lkf, a transitive dependency of structured_pwlkf. Skip
# this module gracefully instead of erroring pytest collection for the
# whole test suite when it isn't installed.
pytest.importorskip("lightning")

from amp.torch.models.experimental.structured_pwlkf import StructuredPWLKF, validate_feature_order_compatibility
from amp.torch.models.experimental.example_generators import (
    SimpleRCGenerator,
    TwoStateRCGenerator,
    EquationBasedRCGenerator,
    get_simple_rc_config,
    get_two_state_rc_config,
    get_equation_based_rc_config,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_rc_config():
    """Configuration for simple RC network."""
    return get_simple_rc_config()


@pytest.fixture
def two_state_rc_config():
    """Configuration for 2R2C network."""
    return get_two_state_rc_config()


@pytest.fixture
def simple_rc_equation_config():
    """Configuration for 2R2C network using equation strings."""
    return get_equation_based_rc_config()


# ============================================================================
# Basic Instantiation Tests
# ============================================================================

def test_create_simple_structured_pwlkf(simple_rc_config):
    """Test creating a basic StructuredPWLKF with simple RC generator."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor'],
        observation_features=['t_indoor'],
        latent_dim=1,
        output_dim=1,
        matrix_generator_class=SimpleRCGenerator,
        param_config=simple_rc_config,
    )
    
    # Verify model properties
    assert model.latent_dim == 1
    assert model.control_dim == 1
    assert model.disturbance_dim == 1
    assert model.output_dim == 1
    assert isinstance(model.matrix_generator, SimpleRCGenerator)
    
    # Verify feature names
    assert model.control_features == ['heat_power']
    assert model.disturbance_features == ['t_outdoor']
    assert model.target_features == ['t_indoor']
    assert model.observation_features == ['t_indoor']


def test_create_two_state_structured_pwlkf(two_state_rc_config):
    """Test creating a StructuredPWLKF with 2-state RC generator."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor', 't_envelope'],
        observation_features=['t_indoor', 't_envelope'],
        latent_dim=2,
        output_dim=2,
        matrix_generator_class=TwoStateRCGenerator,
        param_config=two_state_rc_config,
    )
    
    # Verify model properties
    assert model.latent_dim == 2
    assert model.control_dim == 1
    assert model.disturbance_dim == 1
    assert model.output_dim == 2
    assert isinstance(model.matrix_generator, TwoStateRCGenerator)


# ============================================================================
# Feature Order Validation Tests
# ============================================================================

def test_feature_order_validation_valid(simple_rc_config):
    """Test that valid feature ordering passes validation."""
    # This should NOT raise an error
    validate_feature_order_compatibility(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        observation_features=['t_indoor'],
        equation_config=None  # No equation config = warning but no error
    )


def test_feature_order_validation_with_equation_config():
    """Test feature order validation with equation config."""
    equation_config = {
        'control_vars': ['Q_heat'],
        'disturbance_vars': ['T_out'],
        'equations': {
            'T_in': 'some_equation'
        }
    }
    
    # Matching order should pass
    validate_feature_order_compatibility(
        control_features=['Q_heat'],
        disturbance_features=['T_out'],
        observation_features=['T_in'],
        equation_config=equation_config
    )
    
    # Mismatched control order should fail
    with pytest.raises(ValueError, match="Control feature order mismatch"):
        validate_feature_order_compatibility(
            control_features=['wrong_feature'],
            disturbance_features=['T_out'],
            observation_features=['T_in'],
            equation_config=equation_config
        )
    
    # Mismatched disturbance order should fail
    with pytest.raises(ValueError, match="Disturbance feature order mismatch"):
        validate_feature_order_compatibility(
            control_features=['Q_heat'],
            disturbance_features=['wrong_feature'],
            observation_features=['T_in'],
            equation_config=equation_config
        )


# ============================================================================
# Matrix Generation Tests
# ============================================================================

def test_matrix_generation_simple_rc(simple_rc_config):
    """Test that matrix generation produces correct shapes and physically valid values."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor'],
        observation_features=['t_indoor'],
        latent_dim=1,
        output_dim=1,
        matrix_generator_class=SimpleRCGenerator,
        param_config=simple_rc_config,
    )
    
    # Test matrix generation
    batch_size = 5
    d_t = torch.randn(batch_size, 1)  # Random outdoor temperatures
    
    inspection = model.inspect_matrices(d_t)
    
    # Check matrix shapes
    assert inspection['A'].shape == (batch_size, 1, 1)
    assert inspection['B'].shape == (batch_size, 1, 1)
    assert inspection['C'].shape == (batch_size, 1, 1)
    assert inspection['E'].shape == (batch_size, 1, 1)
    
    # Check physical parameter bounds
    params = inspection['params']
    assert 'R' in params
    assert 'C' in params
    
    # R should be bounded [0.5, 10.0]
    assert torch.all(params['R'] >= 0.5)
    assert torch.all(params['R'] <= 10.0)
    
    # C should be bounded [10.0, 1000.0]
    assert torch.all(params['C'] >= 10.0)
    assert torch.all(params['C'] <= 1000.0)
    
    # A matrix should be in [0, 1] for stable system (exp(-dt/tau))
    assert torch.all(inspection['A'] >= 0)
    assert torch.all(inspection['A'] <= 1)
    
    # C matrix should be identity for direct observation
    assert torch.allclose(inspection['C'], torch.ones(batch_size, 1, 1))


def test_matrix_generation_two_state(two_state_rc_config):
    """Test matrix generation for 2-state system."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor', 't_envelope'],
        observation_features=['t_indoor', 't_envelope'],
        latent_dim=2,
        output_dim=2,
        matrix_generator_class=TwoStateRCGenerator,
        param_config=two_state_rc_config,
    )
    
    # Test matrix generation
    batch_size = 3
    d_t = torch.randn(batch_size, 1)
    
    inspection = model.inspect_matrices(d_t)
    
    # Check matrix shapes
    assert inspection['A'].shape == (batch_size, 2, 2)
    assert inspection['B'].shape == (batch_size, 2, 1)
    assert inspection['C'].shape == (batch_size, 2, 2)
    assert inspection['E'].shape == (batch_size, 2, 1)
    
    # Check all 4 parameters exist
    params = inspection['params']
    for param_name in ['R1', 'R2', 'C1', 'C2']:
        assert param_name in params
        # Check bounds
        if param_name in ['R1', 'R2']:
            assert torch.all(params[param_name] >= 0.1)
            assert torch.all(params[param_name] <= 5.0)
        else:  # C1, C2
            assert torch.all(params[param_name] >= 50.0)
            assert torch.all(params[param_name] <= 500.0)


# ============================================================================
# Forward Pass Tests
# ============================================================================

def test_forward_pass_simple_rc(simple_rc_config):
    """Test that forward pass works and produces correct output shapes."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor'],
        observation_features=['t_indoor'],
        latent_dim=1,
        output_dim=1,
        matrix_generator_class=SimpleRCGenerator,
        param_config=simple_rc_config,
    )
    
    # Create synthetic batch
    batch_size = 2
    history_len = 6
    forecast_len = 24
    
    control_inputs = torch.randn(batch_size, 1, forecast_len)
    disturbance_inputs = torch.randn(batch_size, 1, forecast_len)
    historical_inputs = torch.randn(batch_size, 1, history_len)
    historical_disturbances = torch.randn(batch_size, 1, history_len)
    historical_measurements = torch.randn(batch_size, 1, history_len) + 20.0  # ~20°C
    
    # Forward pass
    predictions, uncertainties = model(
        control_inputs=control_inputs,
        disturbance_inputs=disturbance_inputs,
        historical_inputs=historical_inputs,
        historical_disturbances=historical_disturbances,
        historical_measurements=historical_measurements
    )
    
    # Check output shapes
    assert predictions.shape == (batch_size, 1, forecast_len)
    assert uncertainties.shape == (batch_size, 1, forecast_len)
    
    # Check that predictions are reasonable (not NaN or Inf)
    assert torch.all(torch.isfinite(predictions))
    assert torch.all(torch.isfinite(uncertainties))
    
    # Uncertainties should be positive
    assert torch.all(uncertainties >= 0)


def test_forward_pass_two_state(two_state_rc_config):
    """Test forward pass with 2-state system."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor', 't_envelope'],
        observation_features=['t_indoor', 't_envelope'],
        latent_dim=2,
        output_dim=2,
        matrix_generator_class=TwoStateRCGenerator,
        param_config=two_state_rc_config,
    )
    
    # Create synthetic batch
    batch_size = 2
    history_len = 6
    forecast_len = 12
    
    control_inputs = torch.randn(batch_size, 1, forecast_len)
    disturbance_inputs = torch.randn(batch_size, 1, forecast_len)
    historical_inputs = torch.randn(batch_size, 1, history_len)
    historical_disturbances = torch.randn(batch_size, 1, history_len)
    historical_measurements = torch.randn(batch_size, 2, history_len) + 20.0
    
    # Forward pass
    predictions, uncertainties = model(
        control_inputs=control_inputs,
        disturbance_inputs=disturbance_inputs,
        historical_inputs=historical_inputs,
        historical_disturbances=historical_disturbances,
        historical_measurements=historical_measurements
    )
    
    # Check output shapes
    assert predictions.shape == (batch_size, 2, forecast_len)
    assert uncertainties.shape == (batch_size, 2, forecast_len)
    
    # Check validity
    assert torch.all(torch.isfinite(predictions))
    assert torch.all(torch.isfinite(uncertainties))
    assert torch.all(uncertainties >= 0)


# ============================================================================
# Parameter Learning Tests
# ============================================================================

def test_parameter_learning(simple_rc_config):
    """Test that physical parameters can be learned through gradient descent."""
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor'],
        observation_features=['t_indoor'],
        latent_dim=1,
        output_dim=1,
        matrix_generator_class=SimpleRCGenerator,
        param_config=simple_rc_config,
        learning_rate=1e-2,
    )
    
    # Get initial parameters
    d_t = torch.randn(1, 1)
    initial_params = model.get_physical_parameters(d_t)
    initial_R = initial_params['R'].clone()
    initial_C = initial_params['C'].clone()
    
    # Create simple training batch
    control_inputs = torch.ones(1, 1, 10) * 1000.0  # Constant heating
    disturbance_inputs = torch.ones(1, 1, 10) * 5.0  # Constant outdoor temp
    historical_inputs = torch.zeros(1, 1, 6)
    historical_disturbances = torch.ones(1, 1, 6) * 5.0
    historical_measurements = torch.ones(1, 1, 6) * 15.0
    targets = torch.ones(1, 1, 10) * 20.0  # Target: warm up to 20°C
    
    # Run a few optimization steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    initial_loss = None
    for i in range(10):
        optimizer.zero_grad()
        predictions, _ = model(
            control_inputs=control_inputs,
            disturbance_inputs=disturbance_inputs,
            historical_inputs=historical_inputs,
            historical_disturbances=historical_disturbances,
            historical_measurements=historical_measurements
        )
        loss = torch.nn.functional.mse_loss(predictions, targets)
        
        if i == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    
    # Loss should decrease
    assert final_loss < initial_loss, "Loss should decrease after training"
    
    # Parameters should have changed
    final_params = model.get_physical_parameters(d_t)
    final_R = final_params['R']
    final_C = final_params['C']
    
    # At least one parameter should change significantly
    param_changed = (
        not torch.allclose(initial_R, final_R, rtol=0.01) or
        not torch.allclose(initial_C, final_C, rtol=0.01)
    )
    assert param_changed, "At least one parameter should change during training"


# ============================================================================
# Integration Test
# ============================================================================

def test_structured_pwlkf_end_to_end(simple_rc_config):
    """
    End-to-end test: create model, generate matrices, run forward pass,
    compute loss, and verify all components work together.
    """
    # Create model
    model = StructuredPWLKF(
        control_features=['heat_power'],
        disturbance_features=['t_outdoor'],
        target_features=['t_indoor'],
        observation_features=['t_indoor'],
        latent_dim=1,
        output_dim=1,
        matrix_generator_class=SimpleRCGenerator,
        param_config=simple_rc_config,
    )
    
    # Inspect matrices
    d_t = torch.tensor([[10.0]])  # 10°C outdoor
    inspection = model.inspect_matrices(d_t)
    
    assert 'A' in inspection
    assert 'B' in inspection
    assert 'C' in inspection
    assert 'E' in inspection
    assert 'params' in inspection
    
    # Run forward pass
    control_inputs = torch.randn(2, 1, 24)
    disturbance_inputs = torch.randn(2, 1, 24)
    historical_inputs = torch.randn(2, 1, 6)
    historical_disturbances = torch.randn(2, 1, 6)
    historical_measurements = torch.randn(2, 1, 6) + 20.0
    targets = torch.randn(2, 1, 24) + 20.0
    
    predictions, uncertainties = model(
        control_inputs=control_inputs,
        disturbance_inputs=disturbance_inputs,
        historical_inputs=historical_inputs,
        historical_disturbances=historical_disturbances,
        historical_measurements=historical_measurements
    )
    
    # Compute loss
    mse_loss = torch.nn.functional.mse_loss(predictions, targets)
    
    # Verify everything is finite and reasonable
    assert torch.isfinite(mse_loss)
    assert mse_loss.item() >= 0
    
    # Verify gradients can be computed
    mse_loss.backward()
    
    # Check that generator parameters have gradients
    for name, param in model.matrix_generator.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} has non-finite gradient"


# ============================================================================
# Equation-Based Tests (String Equations)
# ============================================================================

def test_create_equation_based_rc_model(simple_rc_equation_config):
    """Test creating a StructuredPWLKF using equation strings instead of explicit physics."""
    param_config = simple_rc_equation_config.copy()
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
    
    # Verify model properties
    assert model.latent_dim == 2
    assert model.control_dim == 1
    assert model.disturbance_dim == 1
    assert isinstance(model.matrix_generator, EquationBasedRCGenerator)
    
    # Verify equation config was passed correctly
    assert model.matrix_generator.equation_config is not None
    assert 'T_in' in model.matrix_generator.equations
    assert 'T_env' in model.matrix_generator.equations


def test_equation_based_matrix_generation(simple_rc_equation_config):
    """Test that equation-based approach generates correct matrices."""
    param_config = simple_rc_equation_config.copy()
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
    
    # Test matrix generation
    batch_size = 3
    d_t = torch.randn(batch_size, 1)
    
    inspection = model.inspect_matrices(d_t)
    
    # Check matrix shapes
    assert inspection['A'].shape == (batch_size, 2, 2)
    assert inspection['B'].shape == (batch_size, 2, 1)
    assert inspection['C'].shape == (batch_size, 2, 2)
    assert inspection['E'].shape == (batch_size, 2, 1)
    
    # Check parameters exist and are within bounds
    params = inspection['params']
    for param_name in ['R1', 'R2', 'C1', 'C2']:
        assert param_name in params
        if param_name in ['R1', 'R2']:
            assert torch.all(params[param_name] >= 0.1)
            assert torch.all(params[param_name] <= 5.0)
        else:  # C1, C2
            assert torch.all(params[param_name] >= 50.0)
            assert torch.all(params[param_name] <= 500.0)
    
    # A matrix diagonal should be negative (state decay)
    assert torch.all(inspection['A'][:, 0, 0] < 0), "A[0,0] should be negative"
    assert torch.all(inspection['A'][:, 1, 1] < 0), "A[1,1] should be negative"
    
    # B matrix first element should be positive (heating affects indoor temp)
    assert torch.all(inspection['B'][:, 0, 0] > 0), "B[0,0] should be positive"
    
    # E matrix second element should be positive (outdoor temp affects envelope)
    assert torch.all(inspection['E'][:, 1, 0] > 0), "E[1,0] should be positive"


def test_equation_based_forward_pass(simple_rc_equation_config):
    """Test forward pass with equation-based model."""
    param_config = simple_rc_equation_config.copy()
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
    
    # Create synthetic batch
    batch_size = 2
    history_len = 6
    forecast_len = 12
    
    control_inputs = torch.randn(batch_size, 1, forecast_len) * 1000  # Heating power
    disturbance_inputs = torch.randn(batch_size, 1, forecast_len) * 10 + 5  # Outdoor temp
    historical_inputs = torch.randn(batch_size, 1, history_len) * 1000
    historical_disturbances = torch.randn(batch_size, 1, history_len) * 10 + 5
    historical_measurements = torch.randn(batch_size, 2, history_len) + 20.0
    
    # Forward pass
    predictions, uncertainties = model(
        control_inputs=control_inputs,
        disturbance_inputs=disturbance_inputs,
        historical_inputs=historical_inputs,
        historical_disturbances=historical_disturbances,
        historical_measurements=historical_measurements
    )
    
    # Check output shapes
    assert predictions.shape == (batch_size, 2, forecast_len)
    assert uncertainties.shape == (batch_size, 2, forecast_len)
    
    # Check validity
    assert torch.all(torch.isfinite(predictions))
    assert torch.all(torch.isfinite(uncertainties))
    assert torch.all(uncertainties >= 0)


def test_equation_based_parameter_learning(simple_rc_equation_config):
    """Test that parameters can be learned with equation-based models."""
    param_config = simple_rc_equation_config.copy()
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
        learning_rate=1e-2,
    )
    
    # Get initial parameters
    d_t = torch.randn(1, 1)
    initial_params = model.get_physical_parameters(d_t)
    initial_R1 = initial_params['R1'].clone()
    
    # Create training batch
    control_inputs = torch.ones(1, 1, 10) * 500.0
    disturbance_inputs = torch.ones(1, 1, 10) * 0.0
    historical_inputs = torch.zeros(1, 1, 6)
    historical_disturbances = torch.zeros(1, 1, 6)
    historical_measurements = torch.ones(1, 2, 6) * 18.0
    targets = torch.ones(1, 2, 10) * 22.0
    
    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for i in range(5):
        optimizer.zero_grad()
        predictions, _ = model(
            control_inputs=control_inputs,
            disturbance_inputs=disturbance_inputs,
            historical_inputs=historical_inputs,
            historical_disturbances=historical_disturbances,
            historical_measurements=historical_measurements
        )
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
    
    # Parameters should have changed
    final_params = model.get_physical_parameters(d_t)
    final_R1 = final_params['R1']
    
    assert not torch.allclose(initial_R1, final_R1, rtol=0.01), \
        "Parameters should change during training"


def test_equation_based_gradient_computation(simple_rc_equation_config):
    """Test that gradients flow through equation-based matrix generation."""
    param_config = simple_rc_equation_config.copy()
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
    
    # Single forward-backward pass
    control_inputs = torch.randn(1, 1, 10)
    disturbance_inputs = torch.randn(1, 1, 10)
    historical_inputs = torch.randn(1, 1, 6)
    historical_disturbances = torch.randn(1, 1, 6)
    historical_measurements = torch.randn(1, 2, 6) + 20.0
    targets = torch.randn(1, 2, 10) + 20.0
    
    predictions, _ = model(
        control_inputs=control_inputs,
        disturbance_inputs=disturbance_inputs,
        historical_inputs=historical_inputs,
        historical_disturbances=historical_disturbances,
        historical_measurements=historical_measurements
    )
    
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    
    # Verify matrix generator parameters have gradients
    # (Note: KF noise parameters like log_Q_diag may not always have gradients
    # depending on the specific data, which is expected behavior)
    generator_params_have_grads = False
    
    print("\n=== Gradient Status for All Model Parameters ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None
            grad_info = "✓ HAS GRADIENT" if has_grad else "✗ NO GRADIENT"
            print(f"{name:40s} {grad_info}")
            
            # Track if generator params have grads
            if 'matrix_generator' in name and has_grad:
                generator_params_have_grads = True
    
    # Check specifically matrix generator parameters
    print("\n=== Matrix Generator Parameters ===")
    for name, param in model.matrix_generator.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Matrix generator parameter {name} has no gradient"
            assert torch.all(torch.isfinite(param.grad)), f"Matrix generator parameter {name} has non-finite gradient"
            print(f"{name:40s} ✓ gradient OK")
            generator_params_have_grads = True
    
    assert generator_params_have_grads, "Matrix generator should have at least some parameters with gradients"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
