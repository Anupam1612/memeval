from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult
from memeval.metrics.recall import RecallAccuracyMetric
from memeval.metrics.relevance import RelevanceMetric
from memeval.metrics.consistency import ConsistencyMetric
from memeval.metrics.forgetting import ForgettingQualityMetric
from memeval.metrics.update import UpdatePropagationMetric
from memeval.metrics.latency import LatencyCostMetric
from memeval.metrics.scalability import ScalabilityMetric
from memeval.metrics.privacy import PrivacyIsolationMetric

ALL_METRICS = [
    RecallAccuracyMetric,
    RelevanceMetric,
    ConsistencyMetric,
    ForgettingQualityMetric,
    UpdatePropagationMetric,
    LatencyCostMetric,
    ScalabilityMetric,
    PrivacyIsolationMetric,
]

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {m.name: m for m in ALL_METRICS}

__all__ = [
    "BaseMetric",
    "MetricCategory",
    "MetricResult",
    "RecallAccuracyMetric",
    "RelevanceMetric",
    "ConsistencyMetric",
    "ForgettingQualityMetric",
    "UpdatePropagationMetric",
    "LatencyCostMetric",
    "ScalabilityMetric",
    "PrivacyIsolationMetric",
    "ALL_METRICS",
    "METRIC_REGISTRY",
]
