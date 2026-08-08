"""Tests for the physics modules (aerodynamics, drivetrain, thermal)."""

from __future__ import annotations

import torch

from src.physics.aerodynamics import (
    BETZ_LIMIT,
    aerodynamic_physics_loss,
    jensen_wake_deficit,
    power_coefficient,
    rotor_mechanical_power,
    tip_speed_ratio,
)
from src.physics.drivetrain import (
    bearing_l10_life_hours,
    drivetrain_physics_loss,
    gearbox_torque_transfer,
    vibration_stress_energy,
)
from src.physics.thermal import (
    generator_heat_dissipation,
    simulate_winding_temperature,
    steady_state_winding_temperature,
    thermal_physics_loss,
)


class TestAerodynamics:
    def test_cp_within_betz_limit(self):
        """Cp must always lie in [0, 16/27] over a broad operating grid."""
        lam = torch.linspace(0.1, 20.0, 100).repeat(50, 1)
        beta = torch.linspace(0.0, 45.0, 50).unsqueeze(1).expand(-1, 100)
        cp = power_coefficient(lam, beta)
        assert torch.all(cp >= 0.0)
        assert torch.all(cp <= BETZ_LIMIT + 1e-6)

    def test_rotor_power_cubic_in_wind(self):
        """Doubling wind speed at fixed Cp multiplies power by 8."""
        cp = torch.tensor([0.4])
        p1 = rotor_mechanical_power(torch.tensor([8.0]), cp, rotor_radius_m=60.0)
        p2 = rotor_mechanical_power(torch.tensor([16.0]), cp, rotor_radius_m=60.0)
        assert torch.allclose(p2 / p1, torch.tensor(8.0), rtol=1e-5)

    def test_wake_deficit_decays_downstream(self):
        """Jensen deficit decreases monotonically with distance and stays in [0,1]."""
        ct = torch.tensor(0.8)
        d_near = jensen_wake_deficit(torch.tensor(100.0), 60.0, ct)
        d_far = jensen_wake_deficit(torch.tensor(1000.0), 60.0, ct)
        assert 0.0 <= float(d_far) < float(d_near) <= 1.0

    def test_aero_loss_gradients(self):
        """The aero physics loss must be differentiable w.r.t. predictions."""
        pred = (torch.rand(16) * 1e6).requires_grad_(True)
        loss = aerodynamic_physics_loss(
            pred, torch.rand(16) * 12 + 3, torch.rand(16) * 2, torch.zeros(16), 60.0
        )
        loss.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_tip_speed_ratio_positive(self):
        lam = tip_speed_ratio(torch.tensor([1.5]), torch.tensor([10.0]), 60.0)
        assert float(lam) > 0


class TestDrivetrain:
    def test_torque_conservation(self):
        """HSS power (T*omega) never exceeds rotor power (efficiency <= 1)."""
        rotor_torque = torch.tensor([2.0e6])
        rotor_speed = torch.tensor([1.5])
        ratio, eta = 97.0, 0.97
        hss_torque = gearbox_torque_transfer(rotor_torque, ratio, eta)
        p_in = rotor_torque * rotor_speed
        p_out = hss_torque * rotor_speed * ratio
        assert float(p_out) <= float(p_in)
        assert torch.allclose(p_out / p_in, torch.tensor(eta), rtol=1e-6)

    def test_l10_life_decreases_with_load(self):
        speed = torch.tensor([1500.0])
        life_low = bearing_l10_life_hours(5.0e5, torch.tensor([1.0e5]), speed)
        life_high = bearing_l10_life_hours(5.0e5, torch.tensor([3.0e5]), speed)
        assert float(life_high) < float(life_low)

    def test_vibration_energy_nonnegative_and_quadratic(self):
        e1 = vibration_stress_energy(torch.tensor([2.0]))
        e2 = vibration_stress_energy(torch.tensor([4.0]))
        assert float(e1) >= 0
        assert torch.allclose(e2 / e1, torch.tensor(4.0), rtol=1e-6)

    def test_drivetrain_loss_gradients(self):
        pred = (torch.rand(8) * 2.0e4).requires_grad_(True)
        loss = drivetrain_physics_loss(pred, torch.rand(8) * 2.0e6)
        loss.backward()
        assert pred.grad is not None and torch.all(torch.isfinite(pred.grad))


class TestThermal:
    def test_heat_dissipation_positive(self):
        q = generator_heat_dissipation(torch.tensor([500.0]), torch.tensor([150.0]))
        assert float(q) > 0

    def test_energy_conservation_steady_state(self):
        """At steady state, temperature settles to T_coolant + R_th * Q."""
        q = torch.full((5000,), 1.0e4)
        coolant = torch.full((5000,), 40.0)
        traj = simulate_winding_temperature(q, coolant, initial_temp_c=40.0, dt_s=10.0)
        expected = steady_state_winding_temperature(q[-1], coolant[-1])
        assert abs(float(traj[-1]) - float(expected)) < 1.0

    def test_winding_never_below_coolant_at_steady_state(self):
        q = torch.tensor([5.0e3])
        coolant = torch.tensor([35.0])
        t_ss = steady_state_winding_temperature(q, coolant)
        assert float(t_ss) >= float(coolant)

    def test_thermal_loss_gradients(self):
        pred = (torch.rand(8) * 100.0).requires_grad_(True)
        loss = thermal_physics_loss(
            pred, torch.rand(8) * 500, torch.rand(8) * 150, torch.full((8,), 40.0)
        )
        loss.backward()
        assert pred.grad is not None and torch.all(torch.isfinite(pred.grad))
