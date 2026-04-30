from __future__ import annotations

from dataclasses import dataclass

from domain.types import FurnaceObservation


@dataclass
class FurnaceDisturbanceEstimate:
    d_Tavg_C: float = 0.0
    d_Tstack_C: float = 0.0
    d_vstack_mps: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.d_Tavg_C), float(self.d_Tstack_C), float(self.d_vstack_mps))


@dataclass(frozen=True)
class FurnaceDisturbanceObserverConfig:
    alpha: float = 0.05
    max_abs_T_C: float = 300.0
    max_abs_v_mps: float = 10.0


class FurnaceDisturbanceObserver:
    """Low-pass residual observer for additive furnace output disturbances."""

    def __init__(self, cfg: FurnaceDisturbanceObserverConfig | None = None):
        self.cfg = cfg or FurnaceDisturbanceObserverConfig()
        self.estimate = FurnaceDisturbanceEstimate()

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return min(max(float(x), lo), hi)

    def update(self, measured: FurnaceObservation, nominal: FurnaceObservation) -> FurnaceDisturbanceEstimate:
        a = min(max(self.cfg.alpha, 0.0), 1.0)
        r_avg = measured.T_avg_C - nominal.T_avg_C
        r_stack = measured.T_stack_C - nominal.T_stack_C
        r_v = measured.v_stack_mps - nominal.v_stack_mps
        self.estimate = FurnaceDisturbanceEstimate(
            d_Tavg_C=self._clip((1.0 - a) * self.estimate.d_Tavg_C + a * r_avg, -self.cfg.max_abs_T_C, self.cfg.max_abs_T_C),
            d_Tstack_C=self._clip((1.0 - a) * self.estimate.d_Tstack_C + a * r_stack, -self.cfg.max_abs_T_C, self.cfg.max_abs_T_C),
            d_vstack_mps=self._clip((1.0 - a) * self.estimate.d_vstack_mps + a * r_v, -self.cfg.max_abs_v_mps, self.cfg.max_abs_v_mps),
        )
        return self.estimate
