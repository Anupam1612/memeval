from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult
from memeval.metrics.consistency import ConsistencyMetric
from memeval.metrics.cost import CostMetric
from memeval.metrics.forgetting import ForgettingQualityMetric
from memeval.metrics.latency import LatencyCostMetric
from memeval.metrics.privacy import PrivacyIsolationMetric
from memeval.metrics.recall import RecallAccuracyMetric
from memeval.metrics.relevance import RelevanceMetric
from memeval.metrics.scalability import ScalabilityMetric
from memeval.metrics.update import UpdatePropagationMetric

ALL_METRICS = [
    RecallAccuracyMetric,
    RelevanceMetric,
    ConsistencyMetric,
    ForgettingQualityMetric,
    UpdatePropagationMetric,
    LatencyCostMetric,
    ScalabilityMetric,
    PrivacyIsolationMetric,
    CostMetric,
]

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {m.name: m for m in ALL_METRICS}  # type: ignore[attr-defined,misc]

__all__ = [
    "BaseMetric",
    "MetricCategory",
    "MetricResult",
    "RecallAccuracyMetric",
    "RelevanceMetric",
    "ConsistencyMetric",
    "CostMetric",
    "ForgettingQualityMetric",
    "UpdatePropagationMetric",
    "LatencyCostMetric",
    "ScalabilityMetric",
    "PrivacyIsolationMetric",
    "ALL_METRICS",
    "METRIC_REGISTRY",
]
