from __future__ import annotations

from dataclasses import dataclass

from domain.types import FurnaceObservation
from .config import Config
from domain.types import ResourceBoundary
from .physics import rho_g, mdot_stack_cap


@dataclass(frozen=True)
class ResourceModelConfig:
    stack_to_preheater_loss_C: float = 0.0
    extractable_velocity_fraction: float = 1.0
    min_stack_T_C: float = 100.0
    min_stack_v_mps: float = 0.1
    aux_Tg_max_C: float = 1100.0
    vg_cmd_max_mps: float = 12.0


@dataclass(frozen=True)
class ResourceState:
    time_s: float
    T_stack_available_C: float
    v_stack_available_mps: float
    mdot_stack_available_kgps: float
    effective_Tg_cap_C: float
    effective_vg_cap_mps: float
    natural_resource: ResourceBoundary
    effective_resource: ResourceBoundary


class ResourceModel:
    """Convert furnace stack outputs into preheater resource limits.

    Natural stack temperature/flow are dynamic and come from the furnace model.
    Auxiliary heating can raise gas temperature up to aux_Tg_max_C. Preheater
    v_g is interpreted as heat-transfer-side circulation velocity, so natural
    stack mass flow is tracked as a resource diagnostic / circulation demand,
    not as a hard v_g clamp.
    """

    def __init__(self, cfg: ResourceModelConfig | None = None, opt_cfg: Config | None = None):
        self.cfg = cfg or ResourceModelConfig()
        self.opt_cfg = opt_cfg or Config()

    def from_observation(self, obs: FurnaceObservation) -> ResourceState:
        T_avail = max(self.cfg.min_stack_T_C, float(obs.T_stack_C) - self.cfg.stack_to_preheater_loss_C)
        v_avail = max(self.cfg.min_stack_v_mps, float(obs.v_stack_mps) * self.cfg.extractable_velocity_fraction)
        natural = ResourceBoundary(T_avail, v_avail)
        try:
            mdot_cap = mdot_stack_cap(natural, self.opt_cfg)
        except Exception:
            mdot_cap = rho_g(T_avail, self.opt_cfg) * self.opt_cfg.A_S * v_avail
        eff_vg = float(self.cfg.vg_cmd_max_mps)
        effective = ResourceBoundary(float(self.cfg.aux_Tg_max_C), float(max(eff_vg, 0.1)))
        return ResourceState(
            time_s=float(obs.time_s),
            T_stack_available_C=float(T_avail),
            v_stack_available_mps=float(v_avail),
            mdot_stack_available_kgps=float(max(mdot_cap, 1e-12)),
            effective_Tg_cap_C=float(self.cfg.aux_Tg_max_C),
            effective_vg_cap_mps=float(eff_vg),
            natural_resource=natural,
            effective_resource=effective,
        )
