"""Physics-grounded explainability (SHAP → physics residual mapping)."""

from src.explainability.physics_shap import (
    CATEGORY_RESIDUALS,
    DEFAULT_FEATURE_PHYSICS_MAP,
    PhysicsSHAP,
)

__all__ = ["CATEGORY_RESIDUALS", "DEFAULT_FEATURE_PHYSICS_MAP", "PhysicsSHAP"]
