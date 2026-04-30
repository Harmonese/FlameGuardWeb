from __future__ import annotations

from dataclasses import dataclass, replace

from domain.types import ActuatorCommand, ControlSetpoint
from controller.predictor.config import Config
from controller.predictor.physics import mdot_preheater, rho_g
from controller.predictor.actuator import ActuatorDynamic, ActuatorDynamicConfig


@dataclass
class ControlExecutorConfig:
    Tg_min_C: float = 100.0
    vg_min_mps: float = 3.0
    Tg_max_C: float = 2000.0
    vg_max_mps: float = 12.0
    cp_g_kJ_per_kgK: float = 1.05
    T_amb_C: float = 20.0
    max_delta_Tg_cmd_C: float = 40.0
    max_delta_vg_cmd_mps: float = 0.8
    # README actuator dynamics: Tg channel approx 1/(3s+1), vg channel approx 1/(s+1)
    tau_Tg_s: float = 3.0
    tau_vg_s: float = 1.0
    recovery_guard_enabled: bool = True
    recovery_Tg_C: float = 1100.0
    recovery_vg_mps: float = 12.0
    recovery_bypass_actuator_lag: bool = True


class ControlExecutor:
    def __init__(self, cfg: ControlExecutorConfig | None = None, opt_cfg: Config | None = None):
        self.cfg = cfg or ControlExecutorConfig()
        self.opt_cfg = opt_cfg or Config()
        self._prev_Tg_cmd = 175.0
        self._prev_vg_cmd = 6.0
        self._last_time_s: float | None = None
        self.actuator = ActuatorDynamic(
            ActuatorDynamicConfig(
                tau_Tg_s=self.cfg.tau_Tg_s,
                tau_vg_s=self.cfg.tau_vg_s,
                max_dTg_C_per_s=self.cfg.max_delta_Tg_cmd_C / 2.0,
                max_dvg_mps_per_s=self.cfg.max_delta_vg_cmd_mps / 2.0,
                Tg_min_C=self.cfg.Tg_min_C,
                Tg_max_C=self.cfg.Tg_max_C,
                vg_min_mps=self.cfg.vg_min_mps,
                vg_max_mps=self.cfg.vg_max_mps,
                cp_g_kJ_per_kgK=self.cfg.cp_g_kJ_per_kgK,
            ),
            opt_cfg=self.opt_cfg,
        )

    def initialize_previous(self, Tg_cmd_C: float, vg_cmd_mps: float) -> None:
        """Synchronize internal actuator state with a known command."""
        self._prev_Tg_cmd = float(Tg_cmd_C)
        self._prev_vg_cmd = float(vg_cmd_mps)
        self._last_time_s = None
        self.actuator.initialize(Tg_cmd_C, vg_cmd_mps)

    @staticmethod
    def _rate_limit(target: float, prev: float, max_step: float) -> float:
        return min(max(target, prev - max_step), prev + max_step)

    def _dt_from_time(self, time_s: float) -> float:
        if self._last_time_s is None:
            dt = 2.0
        else:
            dt = max(float(time_s) - self._last_time_s, 1e-9)
        self._last_time_s = float(time_s)
        return dt

    def step(self, setpoint: ControlSetpoint, snapshot=None, estimate=None, dt_s: float | None = None) -> ActuatorCommand:
        """Protocol-compatible executor entrypoint.

        snapshot/estimate/dt_s are accepted for interface compatibility; this
        executor only needs the setpoint because safety gating is encoded by
        controller/runtime into setpoint.recovery_guard_requested.
        """
        return self.translate_setpoint(setpoint)

    def translate_setpoint(self, setpoint: ControlSetpoint) -> ActuatorCommand:
        """Translate an NMPC setpoint into actuator commands.

        This path now uses rate-limit + first-order actuator dynamics and dynamic
        natural-resource accounting.  If gas temperature requested exceeds the
        natural stack temperature, auxiliary heat supplies the gap and is logged
        as Q_aux_heat_kW.  If natural stack mass flow is insufficient, v_g is
        not clamped; the shortage is logged as auxiliary circulation / recirculation demand.
        """
        dt_s = self._dt_from_time(setpoint.time_s)
        recovery_guard_active = bool(self.cfg.recovery_guard_enabled and setpoint.recovery_guard_requested)
        Tg_ref = float(setpoint.Tg_ref_C)
        vg_ref = float(setpoint.vg_ref_mps)
        if recovery_guard_active:
            Tg_ref = max(Tg_ref, float(self.cfg.recovery_Tg_C))
            vg_ref = max(vg_ref, float(self.cfg.recovery_vg_mps))
            # Safety recovery is intentionally outside the economic NMPC layer.
            # In an unsafe state, keep the actuator model internally synchronized
            # with the emergency command so the plant receives a prompt recovery
            # action rather than waiting for the nominal first-order lag to wind up.
            if self.cfg.recovery_bypass_actuator_lag:
                self.actuator.initialize(Tg_ref, vg_ref)

        Tg_cmd, vg_cmd, applied = self.actuator.step(
            Tg_ref,
            vg_ref,
            dt_s,
            T_stack_available_C=setpoint.T_stack_available_C,
            v_stack_available_mps=setpoint.v_stack_available_mps,
            mdot_stack_cap_kgps=setpoint.mdot_stack_cap_kgps,
        )
        self._prev_Tg_cmd = Tg_cmd
        self._prev_vg_cmd = vg_cmd
        cmd = self.actuator.to_command(setpoint.time_s, applied)
        if recovery_guard_active:
            cmd = replace(cmd, recovery_guard_active=True)
        return cmd


