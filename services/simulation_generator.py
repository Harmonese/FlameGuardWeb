from __future__ import annotations

"""Phase-1 pseudo-realtime telemetry generator.

This is not the final FlameGuard-main NMPC runtime. It intentionally implements
lightweight first-order dynamics that move only when the dashboard is polled.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .composition_adapter import DEFAULT_WET_MASS_FLOW_KGPS, composition_to_feedstock


@dataclass
class GeneratorState:
    time_s: float = 0.0
    running: bool = True
    source: str = "simulation"
    composition: list[float] = field(default_factory=lambda: [0.20, 0.20, 0.10, 0.20, 0.20, 0.10])
    confidence: float = 1.0
    wet_mass_flow_kgps: float = DEFAULT_WET_MASS_FLOW_KGPS
    T_avg_C: float = 872.0
    T_stack_C: float = 910.0
    v_stack_mps: float = 17.0
    T_solid_out_C: float = 166.0
    omega_out: float = 0.335
    Tg_ref_C: float = 790.0
    vg_ref_mps: float = 11.5
    Tg_cmd_C: float = 785.0
    vg_cmd_mps: float = 11.2
    Q_aux_heat_kW: float = 0.0
    recovery_guard_active: bool = False
    operator_feasible: bool = True
    _last_wall_time: float = field(default_factory=time.monotonic)


class Phase1SimulationGenerator:
    def __init__(self) -> None:
        self.state = GeneratorState()

    def start(self) -> None:
        self.state.running = True
        self.state._last_wall_time = time.monotonic()

    def stop(self) -> None:
        self._advance_to_now()
        self.state.running = False

    def reset(self) -> None:
        self.state = GeneratorState()

    def update_feedstock(self, composition: list[float], *, source: str = "manual", confidence: float = 1.0, wet_mass_flow_kgps: float | None = None) -> dict:
        self._advance_to_now()
        self.state.composition = composition_to_feedstock(composition, normalize=False).raw["composition"]
        self.state.source = str(source or "manual")
        self.state.confidence = max(0.0, min(1.0, float(confidence)))
        if wet_mass_flow_kgps is not None:
            self.state.wet_mass_flow_kgps = max(0.01, float(wet_mass_flow_kgps))
        return self.snapshot()

    def snapshot(self) -> dict:
        self._advance_to_now()
        return self._build_dashboard_payload()

    def _advance_to_now(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self.state._last_wall_time)
        self.state._last_wall_time = now
        if not self.state.running:
            return
        simulated_elapsed = min(elapsed, 5.0)
        while simulated_elapsed > 1e-9:
            dt = min(1.0, simulated_elapsed)
            self._step(dt)
            simulated_elapsed -= dt

    def _step(self, dt_s: float) -> None:
        s = self.state
        s.time_s += dt_s
        feed = composition_to_feedstock(s.composition, time_s=s.time_s, source=s.source, confidence=s.confidence, wet_mass_flow_kgps=s.wet_mass_flow_kgps)
        moisture = feed.moisture_wb
        flow = feed.wet_mass_flow_kgps or DEFAULT_WET_MASS_FLOW_KGPS
        flow_factor = (flow - DEFAULT_WET_MASS_FLOW_KGPS) / DEFAULT_WET_MASS_FLOW_KGPS
        wet_load = (moisture - 0.74) * 140.0 + flow_factor * 22.0
        wave = math.sin(s.time_s / 45.0) * 4.0 + math.sin(s.time_s / 13.0) * 1.3

        T_set_C = 873.0
        desired_T_avg = T_set_C - wet_load + wave + 0.018 * (s.Tg_cmd_C - 780.0) + 0.55 * (s.vg_cmd_mps - 11.0)
        s.T_avg_C += (desired_T_avg - s.T_avg_C) * min(1.0, dt_s / 20.0)

        temp_error = T_set_C - s.T_avg_C
        s.Tg_ref_C = _clamp(780.0 + temp_error * 3.0 + (moisture - 0.72) * 65.0, 650.0, 950.0)
        s.vg_ref_mps = _clamp(11.5 + temp_error * 0.035 + flow_factor * 1.2, 6.0, 18.0)
        s.Tg_cmd_C += (s.Tg_ref_C - s.Tg_cmd_C) * min(1.0, dt_s / 10.0)
        s.vg_cmd_mps += (s.vg_ref_mps - s.vg_cmd_mps) * min(1.0, dt_s / 8.0)

        target_omega_out = _clamp(moisture - 0.39 - (s.Tg_cmd_C - 780.0) * 0.00038 - (s.vg_cmd_mps - 11.0) * 0.006, 0.18, 0.58)
        target_T_solid = _clamp(145.0 + (s.Tg_cmd_C - 700.0) * 0.08 + s.vg_cmd_mps * 1.6 - moisture * 18.0, 95.0, 230.0)
        s.omega_out += (target_omega_out - s.omega_out) * min(1.0, dt_s / 28.0)
        s.T_solid_out_C += (target_T_solid - s.T_solid_out_C) * min(1.0, dt_s / 22.0)

        s.T_stack_C = _clamp(900.0 + 0.22 * (s.T_avg_C - T_set_C) - 0.08 * (s.Tg_cmd_C - 780.0) + wave * 0.8, 760.0, 1120.0)
        s.v_stack_mps = _clamp(17.0 + 0.12 * math.sin(s.time_s / 30.0) + flow_factor * 0.5, 10.0, 24.0)

        safety_margin_C = s.T_avg_C - 850.0
        s.recovery_guard_active = safety_margin_C < 10.0
        s.operator_feasible = safety_margin_C > -5.0 and s.Tg_ref_C < 948.0 and s.vg_ref_mps < 17.8
        s.Q_aux_heat_kW = max(0.0, (850.0 - s.T_avg_C) * 5.0 + (s.Tg_ref_C - 900.0) * 0.8)

    def _build_dashboard_payload(self) -> dict[str, Any]:
        s = self.state
        feed = composition_to_feedstock(s.composition, time_s=s.time_s, source=s.source, confidence=s.confidence, wet_mass_flow_kgps=s.wet_mass_flow_kgps)
        T_set_C = 873.0
        T_compliance_min_C = 850.0
        omega_target = 0.3218
        safety_margin_C = s.T_avg_C - T_compliance_min_C
        wet_flow = feed.wet_mass_flow_kgps or DEFAULT_WET_MASS_FLOW_KGPS
        dry_out = wet_flow * max(0.05, 1.0 - s.omega_out)
        water_out = wet_flow * s.omega_out
        mdot_stack_available = max(0.0, 0.042 * s.v_stack_mps)
        health_ok = bool(s.operator_feasible and safety_margin_C > 0.0)
        return {
            "success": True,
            "schema": "FlameGuardWeb.dashboard.v1",
            "time_s": round(s.time_s, 3),
            "running": s.running,
            "feedstock": {
                **feed.to_dict(),
                "composition": list(s.composition),
                "composition_names": ["菜叶", "西瓜皮", "橙子皮", "肉", "杂项混合", "米饭"],
            },
            "furnace": {
                "T_avg_C": s.T_avg_C,
                "T_stack_C": s.T_stack_C,
                "v_stack_mps": s.v_stack_mps,
                "T_set_C": T_set_C,
                "T_compliance_min_C": T_compliance_min_C,
                "temperature_error_C": s.T_avg_C - T_set_C,
            },
            "preheater": {
                "omega_out": s.omega_out,
                "T_solid_out_C": s.T_solid_out_C,
                "Tg_in_C": s.Tg_cmd_C,
                "Tg_out_C": max(120.0, s.Tg_cmd_C - 160.0 - s.vg_cmd_mps * 3.5),
                "wet_out_kgps": wet_flow,
                "dry_out_kgps": dry_out,
                "water_out_kgps": water_out,
            },
            "control": {
                "Tg_ref_C": s.Tg_ref_C,
                "Tg_cmd_C": s.Tg_cmd_C,
                "vg_ref_mps": s.vg_ref_mps,
                "vg_cmd_mps": s.vg_cmd_mps,
                "omega_target": omega_target,
                "Q_aux_heat_kW": s.Q_aux_heat_kW,
                "operator_feasible": s.operator_feasible,
                "recovery_guard_active": s.recovery_guard_active,
                "safety_margin_C": safety_margin_C,
                "mdot_stack_available_kgps": mdot_stack_available,
                "mdot_preheater_kgps": 0.037 * s.vg_cmd_mps,
                "fan_circulation_power_kW": 0.08 * s.vg_cmd_mps * s.vg_cmd_mps,
            },
            "health": {
                "ok": health_ok,
                "status": "running" if s.running else "paused",
                "stale": not s.running,
                "comms_ok": True,
                "sensors_ok": True,
                "actuators_ok": True,
                "message": "伪实时仿真数据" if s.running else "监控已暂停，数据保持最近状态",
            },
        }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
