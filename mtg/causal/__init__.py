"""Causal modelling module for MTG-Causal-RL.

This module provides:

* The structural causal model (:class:`StructuralCausalModel`) that
  encodes the MTG strategic DAG and its structural equations.
* Identification-aware estimators (:class:`CausalInference`) for
  average / conditional treatment effects and back-door / front-door
  adjustment.
* Off-policy value estimators (:class:`IPSEstimator`,
  :class:`DRPolicyEvaluator`) used for offline evaluation of
  evaluation policies against data collected under a different
  behaviour policy.
"""

from mtg.causal.inference import (
    AverageTreatmentEffect,
    CausalInference,
    DRPolicyEvaluator,
    IPSEstimator,
    OPEResult,
)
from mtg.causal.scm import (
    CausalLayer,
    CausalVariable,
    CausalVariableSet,
    StructuralCausalModel,
)

CausalSCM = StructuralCausalModel

__all__ = [
    "AverageTreatmentEffect",
    "CausalInference",
    "CausalLayer",
    "CausalSCM",
    "CausalVariable",
    "CausalVariableSet",
    "DRPolicyEvaluator",
    "IPSEstimator",
    "OPEResult",
    "StructuralCausalModel",
]
