import torch

from src.physics.constraints import (
    check_violations,
    iso_281_l10_hours,
    physics_loss,
)


def test_check_violations_flags_each_breach():
    bad = {
        "vibration_mms": 5.0,
        "temperature_c": 85.0,
        "rpm": 1850.0,
        "oil_viscosity_cst": 8.0,
        "load_pct": 95.0,
    }
    v = check_violations(bad)
    assert len(v) == 4


def test_physics_loss_monotonic_in_rul_when_limits_breached():
    tel = {
        "vibration_mms": torch.tensor([5.0]),
        "temperature_c": torch.tensor([85.0]),
        "rpm": torch.tensor([1850.0]),
        "oil_viscosity_cst": torch.tensor([8.0]),
    }
    rul_small = torch.tensor([5.0], requires_grad=True)
    rul_large = torch.tensor([500.0], requires_grad=True)
    l_small, _ = physics_loss(tel, rul_small)
    l_large, _ = physics_loss(tel, rul_large)
    # large RUL should be penalized more when physics is broken
    assert l_large.item() > l_small.item()


def test_iso_281_reasonable_life():
    # Typical main/planetary bearing: C ~ 3000 kN, P ~ 200 kN @ 1500 RPM.
    # L10 should come out in the tens-to-hundreds-of-thousands of hours.
    l10 = iso_281_l10_hours(C=3000.0, P=200.0, p=10.0/3.0, rpm=1500.0)
    assert 50_000 < l10 < 5_000_000
    # Higher load should shorten life
    l10_heavy = iso_281_l10_hours(C=3000.0, P=400.0, p=10.0/3.0, rpm=1500.0)
    assert l10_heavy < l10
