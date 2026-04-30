from __future__ import annotations

"""
cleanser.py

将 6 类垃圾组分输入清洗并翻译为优化器所需的等效性质。

输入顺序固定为：
[菜叶, 西瓜皮, 橙子皮, 肉, 杂项混合, 米饭]

输出：
- 等效入口初始含水率 omega0
- 参考干燥时间 tref_min
- 温度敏感性 slope_min_per_c

依赖
----
pip install numpy
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import numpy as np

from .config import Config
from domain.types import EquivalentProperties, FeedstockObservation

NAMES = ["菜叶", "西瓜皮", "橙子皮", "肉", "杂项混合", "米饭"]
OMEGA = np.array([0.948, 0.948, 0.817, 0.442, 0.773, 0.611], dtype=float)
TREF_TIME = np.array([12.1, 17.7, 15.3, 11.5, 16.3, 15.8], dtype=float)
SLOPE = np.array([-0.132, -0.251, -0.189, -0.216, -0.210, -0.243], dtype=float)


@dataclass(frozen=True)
class CompositionInput:
    values: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(NAMES, self.values)}


@dataclass(frozen=True)
class CleansedComposition:
    composition: CompositionInput
    equivalent: EquivalentProperties
    ceq_kJ_per_kgK: float



def validate_composition(values: Sequence[float], *, normalize: bool = False) -> CompositionInput:
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError("组分输入必须正好包含 6 项，对应 [菜叶, 西瓜皮, 橙子皮, 肉, 杂项混合, 米饭]。")
    if np.any(arr < 0.0):
        raise ValueError("组分比例不能为负数。")
    total = float(arr.sum())
    if normalize:
        if total <= 0.0:
            raise ValueError("当 normalize=True 时，组分和必须为正数。")
        arr = arr / total
    else:
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"组分比例和为 {total:.10f}，不等于 1。")
    return CompositionInput(values=arr)



def composition_to_equivalent_properties(
    values: Sequence[float],
    *,
    normalize: bool = False,
    cfg: Config | None = None,
) -> CleansedComposition:
    cfg = cfg or Config()
    comp = validate_composition(values, normalize=normalize)
    x = comp.values
    omega0 = float(np.dot(x, OMEGA))
    tref = float(np.dot(x, TREF_TIME))
    slope = float(np.dot(x, SLOPE))
    ceq = (1.0 - omega0) * cfg.CS + omega0 * cfg.CW
    eq = EquivalentProperties(
        omega0=omega0,
        tref_min=tref,
        slope_min_per_c=slope,
        bulk_density_kg_m3=None,
        wet_mass_flow_kgps=None,
    )
    return CleansedComposition(
        composition=comp,
        equivalent=eq,
        ceq_kJ_per_kgK=float(ceq),
    )



def feedstock_from_composition(
    time_s: float,
    values: Sequence[float],
    *,
    normalize: bool = False,
    bulk_density_kg_m3: float | None = None,
    dry_basis_ratio: float | None = None,
    wet_mass_flow_kgps: float | None = None,
    source: str = "composition_adapter",
    confidence: float = 1.0,
    cfg: Config | None = None,
) -> FeedstockObservation:
    """Convert legacy/test composition vectors into the stable feedstock protocol."""
    result = composition_to_equivalent_properties(values, normalize=normalize, cfg=cfg)
    raw = {"composition": tuple(float(x) for x in result.composition.values)}
    if dry_basis_ratio is not None:
        raw["dry_basis_ratio"] = float(dry_basis_ratio)
    if wet_mass_flow_kgps is not None:
        raw["wet_mass_flow_kgps"] = float(wet_mass_flow_kgps)
    return FeedstockObservation(
        time_s=float(time_s),
        moisture_wb=float(result.equivalent.omega0),
        drying_time_ref_min=float(result.equivalent.tref_min),
        drying_sensitivity_min_per_C=float(result.equivalent.slope_min_per_c),
        bulk_density_kg_m3=bulk_density_kg_m3,
        wet_mass_flow_kgps=wet_mass_flow_kgps,
        source=source,
        confidence=float(confidence),
        raw=raw,
    )


def properties_from_feedstock(feedstock: FeedstockObservation) -> EquivalentProperties:
    """Translate the stable feedstock protocol into model-facing properties."""
    omega = float(feedstock.moisture_wb)
    if not (0.0 <= omega < 1.0):
        raise ValueError(f"moisture_wb must be in [0, 1), got {omega!r}")
    tref = float(feedstock.drying_time_ref_min)
    if tref <= 0.0:
        raise ValueError(f"drying_time_ref_min must be positive, got {tref!r}")
    confidence = float(feedstock.confidence)
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
    density = feedstock.bulk_density_kg_m3
    if density is not None and float(density) <= 0.0:
        raise ValueError(f"bulk_density_kg_m3 must be positive when provided, got {density!r}")
    wet_flow = feedstock.wet_mass_flow_kgps
    if wet_flow is not None and float(wet_flow) <= 0.0:
        raise ValueError(f"wet_mass_flow_kgps must be positive when provided, got {wet_flow!r}")
    return EquivalentProperties(
        omega0=omega,
        tref_min=tref,
        slope_min_per_c=float(feedstock.drying_sensitivity_min_per_C),
        bulk_density_kg_m3=None if density is None else float(density),
        wet_mass_flow_kgps=None if wet_flow is None else float(wet_flow),
    )


def batch_compositions_to_equivalent_properties(
    rows: Iterable[Sequence[float]],
    *,
    normalize: bool = False,
    cfg: Config | None = None,
) -> List[CleansedComposition]:
    cfg = cfg or Config()
    return [
        composition_to_equivalent_properties(row, normalize=normalize, cfg=cfg)
        for row in rows
    ]



def print_cleansed_result(result: CleansedComposition) -> None:
    print("=" * 72)
    print("一、清洗后的组分比例")
    for name, value in result.composition.as_dict().items():
        print(f"{name:<8s}: {value:.6f}")
    print(f"比例和: {result.composition.values.sum():.6f}")

    print("\n" + "=" * 72)
    print("二、等效性质")
    print(f"omega0            = {result.equivalent.omega0:.6f} ({result.equivalent.omega0*100:.2f}%)")
    print(f"tref_min          = {result.equivalent.tref_min:.6f} min")
    print(f"slope_min_per_c   = {result.equivalent.slope_min_per_c:.6f} min/°C")
    print(f"ceq_kJ_per_kgK    = {result.ceq_kJ_per_kgK:.6f}")


__all__ = [
    "NAMES",
    "CompositionInput",
    "CleansedComposition",
    "validate_composition",
    "composition_to_equivalent_properties",
    "feedstock_from_composition",
    "properties_from_feedstock",
    "batch_compositions_to_equivalent_properties",
    "print_cleansed_result",
]
