"""Aerovigil AI: Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction."""

from .model import PhysicsGuidedBNN
from .inference import MonteCarloVI

__version__ = "0.1.0"
__all__ = ["PhysicsGuidedBNN", "MonteCarloVI"]
