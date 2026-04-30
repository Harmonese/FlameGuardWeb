from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from domain.types import FeedstockObservation


class FeedPreviewProvider:
    def get(self, time_s: float, *, horizon_s: float, dt_s: float) -> list[FeedstockObservation]:
        raise NotImplementedError


@dataclass
class ConstantFeedPreview(FeedPreviewProvider):
    feedstock: FeedstockObservation

    def get(self, time_s: float, *, horizon_s: float, dt_s: float) -> list[FeedstockObservation]:
        n = max(1, int(round(horizon_s / max(dt_s, 1e-9))))
        return [
            FeedstockObservation(
                time_s=time_s + (k + 1) * dt_s,
                moisture_wb=self.feedstock.moisture_wb,
                drying_time_ref_min=self.feedstock.drying_time_ref_min,
                drying_sensitivity_min_per_C=self.feedstock.drying_sensitivity_min_per_C,
                bulk_density_kg_m3=self.feedstock.bulk_density_kg_m3,
                wet_mass_flow_kgps=self.feedstock.wet_mass_flow_kgps,
                source=self.feedstock.source,
                confidence=self.feedstock.confidence,
                raw=self.feedstock.raw,
            )
            for k in range(n)
        ]


@dataclass
class KnownScheduleFeedPreview(FeedPreviewProvider):
    feedstock_schedule: Callable[[float], FeedstockObservation]

    def get(self, time_s: float, *, horizon_s: float, dt_s: float) -> list[FeedstockObservation]:
        n = max(1, int(round(horizon_s / max(dt_s, 1e-9))))
        return [self.feedstock_schedule(time_s + (k + 1) * dt_s) for k in range(n)]
