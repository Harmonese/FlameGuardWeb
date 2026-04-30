from __future__ import annotations

from domain.types import ControlSetpoint


def safe_nominal_setpoint(time_s: float, *, Tg_ref_C: float = 800.0, vg_ref_mps: float = 6.0, omega_ref: float = 0.3218) -> ControlSetpoint:
    return ControlSetpoint(
        time_s=float(time_s),
        Tg_ref_C=float(Tg_ref_C),
        vg_ref_mps=float(vg_ref_mps),
        source="safe_nominal_fallback",
        omega_target=float(omega_ref),
        omega_reachable=float(omega_ref),
    )
