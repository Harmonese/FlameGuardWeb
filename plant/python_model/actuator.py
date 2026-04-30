from __future__ import annotations

from dataclasses import dataclass
import math

from domain.types import ActuatorCommand
from .config import Config
from .physics import mdot_preheater


@dataclass(frozen=True)
class ActuatorDynamicConfig:
    """Rate-limited first-order actuator model.

    README described the gas-temperature channel as roughly 1/(3s+1) and the
    gas-flow channel as roughly 1/(1s+1).  The model below keeps the engineering
    rate limits from ControlExecutor and adds those first-order lags.  It is used
    both in the plant path and inside NMPC rollout so prediction and execution
    share the same actuator dynamics.
    """

    tau_Tg_s: float = 3.0
    tau_vg_s: float = 1.0
    max_dTg_C_per_s: float = 20.0
    max_dvg_mps_per_s: float = 0.4
    Tg_min_C: float = 100.0
    Tg_max_C: float = 2000.0
    vg_min_mps: float = 3.0
    vg_max_mps: float = 12.0
    cp_g_kJ_per_kgK: float = 1.05
    # Diagnostic fan/circulation cost used by NMPC. This is an economic
    # penalty model rather than a detailed fan design calculation.
    fan_power_ref_kW: float = 18.0


@dataclass(frozen=True)
class ResourceApplicationResult:
    Tg_cmd_C: float
    vg_cmd_mps: float
    heater_enable: bool
    Q_aux_heat_kW: float
    aux_resource_required: bool
    aux_heat_required: bool
    aux_circulation_required: bool
    mdot_preheater_kgps: float
    mdot_stack_cap_kgps: float
    T_stack_available_C: float
    v_stack_available_mps: float
    mdot_aux_flow_kgps: float = 0.0
    recirculation_required: bool = False
    fan_circulation_power_kW: float = 0.0


class ActuatorDynamic:
    def __init__(self, cfg: ActuatorDynamicConfig | None = None, opt_cfg: Config | None = None):
        self.cfg = cfg or ActuatorDynamicConfig()
        self.opt_cfg = opt_cfg or Config()
        self.Tg_actual_C = 175.0
        self.vg_actual_mps = 6.0
        self.Tg_rate_state_C = 175.0
        self.vg_rate_state_mps = 6.0

    def clone(self) -> "ActuatorDynamic":
        other = ActuatorDynamic(self.cfg, self.opt_cfg)
        other.Tg_actual_C = self.Tg_actual_C
        other.vg_actual_mps = self.vg_actual_mps
        other.Tg_rate_state_C = self.Tg_rate_state_C
        other.vg_rate_state_mps = self.vg_rate_state_mps
        return other

    def initialize(self, Tg_C: float, vg_mps: float) -> None:
        Tg = float(min(max(Tg_C, self.cfg.Tg_min_C), self.cfg.Tg_max_C))
        vg = float(min(max(vg_mps, self.cfg.vg_min_mps), self.cfg.vg_max_mps))
        self.Tg_actual_C = Tg
        self.vg_actual_mps = vg
        self.Tg_rate_state_C = Tg
        self.vg_rate_state_mps = vg

    @staticmethod
    def _rate_limit(target: float, prev: float, max_delta: float) -> float:
        return min(max(target, prev - max_delta), prev + max_delta)

    @staticmethod
    def _first_order(prev: float, target: float, tau_s: float, dt_s: float) -> float:
        if tau_s <= 1e-12:
            return target
        alpha = 1.0 - math.exp(-max(dt_s, 0.0) / tau_s)
        return prev + alpha * (target - prev)

    def apply_resource(
        self,
        Tg_lag_C: float,
        vg_lag_mps: float,
        *,
        T_stack_available_C: float,
        v_stack_available_mps: float,
        mdot_stack_cap_kgps: float,
    ) -> ResourceApplicationResult:
        """Apply natural resource diagnostics and auxiliary heat accounting.

        Temperature shortfall can be supplied by auxiliary heat.  The preheater
        gas velocity v_g is treated as heat-transfer-side circulation velocity,
        not as a hard one-pass stack extraction velocity.  Therefore natural
        stack mass-flow shortage is recorded as auxiliary circulation /
        recirculation demand instead of clamping v_g.
        """
        Tg_cmd = float(min(max(Tg_lag_C, self.cfg.Tg_min_C), self.cfg.Tg_max_C))
        vg_cmd = float(min(max(vg_lag_mps, self.cfg.vg_min_mps), self.cfg.vg_max_mps))

        mdot_cap = max(float(mdot_stack_cap_kgps), 1e-12)
        mdot_need = mdot_preheater(Tg_cmd, vg_cmd, self.opt_cfg)
        mdot_aux_flow = max(mdot_need - mdot_cap, 0.0)
        recirc_required = mdot_aux_flow > 1e-9
        aux_resource_required = bool(recirc_required)
        natural_T = float(T_stack_available_C)
        aux_delta_T = max(Tg_cmd - natural_T, 0.0)
        Q_aux = mdot_need * self.cfg.cp_g_kJ_per_kgK * aux_delta_T
        Tg_aux = aux_delta_T > 1e-9
        if Tg_aux:
            aux_resource_required = True

        # Simple cubic fan/circulation diagnostic: high v_g remains available
        # for emergency recovery, but it now carries an explicit economic cost.
        fan_power = self.cfg.fan_power_ref_kW * (vg_cmd / max(self.cfg.vg_max_mps, 1e-12)) ** 3

        return ResourceApplicationResult(
            Tg_cmd_C=Tg_cmd,
            vg_cmd_mps=vg_cmd,
            heater_enable=Q_aux > 1e-9,
            Q_aux_heat_kW=float(Q_aux),
            aux_resource_required=bool(aux_resource_required),
            aux_heat_required=bool(Tg_aux),
            aux_circulation_required=bool(recirc_required),
            mdot_preheater_kgps=float(mdot_need),
            mdot_stack_cap_kgps=float(mdot_cap),
            T_stack_available_C=float(T_stack_available_C),
            v_stack_available_mps=float(v_stack_available_mps),
            mdot_aux_flow_kgps=float(mdot_aux_flow),
            recirculation_required=bool(recirc_required),
            fan_circulation_power_kW=float(fan_power),
        )

    def step(
        self,
        Tg_ref_C: float,
        vg_ref_mps: float,
        dt_s: float,
        *,
        T_stack_available_C: float,
        v_stack_available_mps: float,
        mdot_stack_cap_kgps: float,
    ) -> tuple[float, float, ResourceApplicationResult]:
        max_dT = self.cfg.max_dTg_C_per_s * max(dt_s, 0.0)
        max_dv = self.cfg.max_dvg_mps_per_s * max(dt_s, 0.0)
        self.Tg_rate_state_C = self._rate_limit(float(Tg_ref_C), self.Tg_rate_state_C, max_dT)
        self.vg_rate_state_mps = self._rate_limit(float(vg_ref_mps), self.vg_rate_state_mps, max_dv)

        Tg_lag = self._first_order(self.Tg_actual_C, self.Tg_rate_state_C, self.cfg.tau_Tg_s, dt_s)
        vg_lag = self._first_order(self.vg_actual_mps, self.vg_rate_state_mps, self.cfg.tau_vg_s, dt_s)
        res = self.apply_resource(
            Tg_lag,
            vg_lag,
            T_stack_available_C=T_stack_available_C,
            v_stack_available_mps=v_stack_available_mps,
            mdot_stack_cap_kgps=mdot_stack_cap_kgps,
        )
        self.Tg_actual_C = res.Tg_cmd_C
        self.vg_actual_mps = res.vg_cmd_mps
        return self.Tg_actual_C, self.vg_actual_mps, res

    def to_command(self, time_s: float, res: ResourceApplicationResult) -> ActuatorCommand:
        return ActuatorCommand(
            time_s=float(time_s),
            Tg_cmd_C=res.Tg_cmd_C,
            vg_cmd_mps=res.vg_cmd_mps,
            heater_enable=res.heater_enable,
            Q_aux_heat_kW=res.Q_aux_heat_kW,
            aux_resource_required=res.aux_resource_required,
            aux_heat_required=res.aux_heat_required,
            aux_circulation_required=res.aux_circulation_required,
            mdot_preheater_kgps=res.mdot_preheater_kgps,
            mdot_stack_cap_kgps=res.mdot_stack_cap_kgps,
            mdot_aux_flow_kgps=res.mdot_aux_flow_kgps,
            fan_circulation_power_kW=res.fan_circulation_power_kW,
        )
