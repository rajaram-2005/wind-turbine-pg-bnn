"""Aerovigil AI: Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction."""

from .inference import MonteCarloVI
from .model import PhysicsGuidedBNN

__version__ = "0.1.0"
__all__ = ["PhysicsGuidedBNN", "MonteCarloVI"]
