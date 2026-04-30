from __future__ import annotations

"""Shared thermal/proxy equations used by model and control layers.

This module is intentionally independent from ``optimizer``.  It is the single
Python location for low-order heat-transfer, flue-gas density, mass-flow, and
furnace steady-proxy equations.  Optimizers and forward models should import
from here rather than from a solver-specific module.
"""

from .config import Config
from domain.types import EquivalentProperties, ResourceBoundary


def validate_equivalent_properties(props: EquivalentProperties) -> EquivalentProperties:
    if not (0.0 < props.omega0 < 1.0):
        raise ValueError("omega0 must be in (0, 1).")
    if props.tref_min <= 0.0:
        raise ValueError("tref_min must be positive.")
    return props


def validate_resource_boundary(resource: ResourceBoundary, cfg: Config) -> ResourceBoundary:
    if resource.T_stack_cap_C < cfg.TG_MIN:
        raise ValueError(
            f"Available stack temperature {resource.T_stack_cap_C:.3f}°C is below "
            f"minimum operating temperature {cfg.TG_MIN:.1f}°C."
        )
    if resource.v_stack_cap_mps <= 0.0:
        raise ValueError("Stack gas velocity must be positive.")
    return resource


def ceq_from_props(props: EquivalentProperties, cfg: Config) -> float:
    return (1.0 - props.omega0) * cfg.CS + props.omega0 * cfg.CW


def rho_g(T_degC: float, cfg: Config) -> float:
    return cfg.RHO_G_REF * cfg.T_G_REF_K / (T_degC + 273.15)


def q_sup_kW(Tg: float, vg: float, Tm: float, cfg: Config) -> float:
    U = cfg.U0 + cfg.K_U * (vg ** cfg.N_U)
    return U * cfg.A * (Tg - Tm) / 1000.0


def evap_water_per_kg_wet(omega0: float, omega_target: float) -> float:
    if omega_target >= 1.0:
        raise ValueError("Target wet-basis moisture must be below 1.")
    if omega0 <= omega_target:
        return 0.0
    return (omega0 - omega_target) / (1.0 - omega_target)


def q_req_kW(Tm: float, props: EquivalentProperties, omega_target: float, cfg: Config) -> float:
    ceq = ceq_from_props(props, cfg)
    m_evap = evap_water_per_kg_wet(props.omega0, omega_target)
    return cfg.MDOT_W * (ceq * (Tm - cfg.T0) + cfg.LAMBDA * m_evap)


def tau20(Tm: float, props: EquivalentProperties, cfg: Config) -> float:
    return props.tref_min + props.slope_min_per_c * (Tm - cfg.T_REF)


def tau_target(Tm: float, props: EquivalentProperties, omega_target: float, cfg: Config) -> float:
    denom = props.omega0 - cfg.OMEGA_MODEL_MIN
    if denom < 0:
        raise ValueError("omega0 < OMEGA_MODEL_MIN, current tau-scaling model is not applicable.")
    if abs(denom) <= 1e-12:
        if abs(props.omega0 - omega_target) <= 1e-12:
            return 0.0
        raise ValueError("omega0 == OMEGA_MODEL_MIN only allows omega_target == OMEGA_MODEL_MIN.")
    return tau20(Tm, props, cfg) * (props.omega0 - omega_target) / denom


def power_kW(Tg: float, vg: float, cfg: Config) -> float:
    flow_area = getattr(cfg, "A_FLOW_EQ", cfg.A_D)
    return rho_g(Tg, cfg) * flow_area * vg * cfg.CPG * (Tg - cfg.T_AMB)


def mdot_stack_cap(resource: ResourceBoundary, cfg: Config) -> float:
    return rho_g(resource.T_stack_cap_C, cfg) * cfg.A_S * resource.v_stack_cap_mps


def mdot_preheater(Tg: float, vg: float, cfg: Config) -> float:
    flow_area = getattr(cfg, "A_FLOW_EQ", cfg.A_D)
    return rho_g(Tg, cfg) * flow_area * vg


def T_avg_proxy(w_percent: float, cfg: Config) -> float:
    return cfg.TAVG_A * w_percent + cfg.TAVG_B


def T_min_proxy(w_percent: float, cfg: Config) -> float:
    return cfg.TMIN_A * w_percent + cfg.TMIN_B


def T_max_proxy(w_percent: float, cfg: Config) -> float:
    return cfg.TMAX_A * w_percent + cfg.TMAX_B


def sigma_proxy(w_percent: float, cfg: Config) -> float:
    return cfg.SIGMA_A * w_percent + cfg.SIGMA_B


def strict_burn_feasible(omega: float, cfg: Config) -> bool:
    w = 100.0 * omega
    return bool(
        T_min_proxy(w, cfg) >= cfg.TMIN_BURN_MIN - 1e-6
        and T_avg_proxy(w, cfg) >= cfg.TAVG_BURN_MIN - 1e-6
        and T_avg_proxy(w, cfg) <= cfg.TAVG_BURN_MAX + 1e-6
        and T_max_proxy(w, cfg) <= cfg.TMAX_BURN_MAX + 1e-6
    )


def steady_band_violation_percent(omega: float, cfg: Config) -> float:
    w = 100.0 * omega
    lo = 100.0 * cfg.OMEGA_STEADY_MIN
    hi = 100.0 * cfg.OMEGA_STEADY_MAX
    if w < lo:
        return lo - w
    if w > hi:
        return w - hi
    return 0.0


__all__ = [
    "Config",
    "EquivalentProperties",
    "ResourceBoundary",
    "validate_equivalent_properties",
    "validate_resource_boundary",
    "ceq_from_props",
    "rho_g",
    "q_sup_kW",
    "q_req_kW",
    "evap_water_per_kg_wet",
    "tau20",
    "tau_target",
    "power_kW",
    "mdot_stack_cap",
    "mdot_preheater",
    "T_avg_proxy",
    "T_min_proxy",
    "T_max_proxy",
    "sigma_proxy",
    "strict_burn_feasible",
    "steady_band_violation_percent",
]
