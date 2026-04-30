from __future__ import annotations

"""Composition adapter for FlameGuardWeb phase 1.

The web UI still uses the six-category waste composition from FlameGuard-0.1.0.
This module converts it to the property-oriented feedstock payload used by
FlameGuard-main: moisture, reference drying time, drying sensitivity, flow,
source, confidence, and raw metadata.
"""

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

NAMES = ["菜叶", "西瓜皮", "橙子皮", "肉", "杂项混合", "米饭"]
# Values copied from FlameGuard-0.1.0 cleanser/cleanser.py.
OMEGA_WB = [0.948, 0.948, 0.817, 0.442, 0.773, 0.611]
TREF_TIME_MIN = [12.1, 17.7, 15.3, 11.5, 16.3, 15.8]
SLOPE_MIN_PER_C = [-0.132, -0.251, -0.189, -0.216, -0.210, -0.243]
DEFAULT_WET_MASS_FLOW_KGPS = 20000.0 / 86400.0
DEFAULT_BULK_DENSITY_KG_M3 = 420.0


@dataclass(frozen=True)
class FeedstockObservationPayload:
    time_s: float
    moisture_wb: float
    drying_time_ref_min: float
    drying_sensitivity_min_per_C: float
    bulk_density_kg_m3: float | None
    wet_mass_flow_kgps: float | None
    source: str
    confidence: float
    raw: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def validate_composition(values: Iterable[float], *, normalize: bool = False) -> list[float]:
    arr = [float(v) for v in values]
    if len(arr) != 6:
        raise ValueError("组分输入必须正好包含 6 项，对应 [菜叶, 西瓜皮, 橙子皮, 肉, 杂项混合, 米饭]。")
    if any(v < 0.0 for v in arr):
        raise ValueError("组分比例不能为负数。")
    total = sum(arr)
    if normalize:
        if total <= 0.0:
            raise ValueError("组分比例和必须为正数。")
        arr = [v / total for v in arr]
    elif abs(total - 1.0) > 1e-6:
        raise ValueError(f"组分比例和为 {total:.6f}，不等于 1。")
    return arr


def composition_to_feedstock(
    values: Iterable[float],
    *,
    time_s: float = 0.0,
    source: str = "manual",
    confidence: float = 1.0,
    wet_mass_flow_kgps: float | None = None,
    bulk_density_kg_m3: float | None = DEFAULT_BULK_DENSITY_KG_M3,
    normalize: bool = False,
) -> FeedstockObservationPayload:
    x = validate_composition(values, normalize=normalize)
    confidence = max(0.0, min(1.0, float(confidence)))
    flow = DEFAULT_WET_MASS_FLOW_KGPS if wet_mass_flow_kgps is None else float(wet_mass_flow_kgps)
    payload = FeedstockObservationPayload(
        time_s=float(time_s),
        moisture_wb=_dot(x, OMEGA_WB),
        drying_time_ref_min=_dot(x, TREF_TIME_MIN),
        drying_sensitivity_min_per_C=_dot(x, SLOPE_MIN_PER_C),
        bulk_density_kg_m3=bulk_density_kg_m3,
        wet_mass_flow_kgps=flow,
        source=str(source or "manual"),
        confidence=confidence,
        raw={
            "composition": x,
            "composition_named": {name: value for name, value in zip(NAMES, x)},
            "adapter": "FlameGuardWeb phase1 composition_adapter",
        },
    )
    return payload
