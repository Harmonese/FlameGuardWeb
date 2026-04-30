from __future__ import annotations

from typing import Callable

from domain.types import (
    ActuatorFeedback,
    FurnaceObservation,
    PlantHealth,
    PlantInitialState,
    PlantSnapshot,
    PlantStepInput,
    PreheaterOutput,
    StackResourceMeasurement,
)

from .preheater import PreheaterForwardModel
from .furnace import FurnaceDyn, dry_basis_ratio_from_feedstock, furnace_feed_from_preheater_output
from .resource import ResourceModel


def _normalize_disturbance(disturbance):
    if disturbance is None:
        return 0.0, 0.0, 0.0
    if isinstance(disturbance, (tuple, list)) and len(disturbance) == 3:
        return float(disturbance[0]), float(disturbance[1]), float(disturbance[2])
    return float(disturbance), 0.0, 0.0


class PythonPlantBackend:
    """Python low-order plant backend used by runtime tests.

    The public protocol is PlantStepInput -> PlantSnapshot.  Test-only external
    disturbances are injected through an optional disturbance_schedule owned by
    this backend, not through the stable PlantStepInput contract.
    """

    def __init__(
        self,
        preheater: PreheaterForwardModel,
        furnace: FurnaceDyn,
        resource_model: ResourceModel | None = None,
        disturbance_schedule: Callable[[float], float | tuple[float, float, float]] | None = None,
    ):
        self.preheater = preheater
        self.furnace = furnace
        self.resource_model = resource_model or ResourceModel()
        self.disturbance_schedule = disturbance_schedule
        self._last_snapshot: PlantSnapshot | None = None

    def reset(self, initial: PlantInitialState | None = None) -> PlantSnapshot:
        if initial is not None and initial.furnace_obs is not None:
            obs = initial.furnace_obs
        else:
            Tavg = self.furnace.refs["T_avg"] + self.furnace.states["T_avg"][1]
            Tstack = self.furnace.refs["T_stack"] + self.furnace.states["T_stack"][1]
            vstack = self.furnace.refs["v_stack"] + self.furnace.states["v_stack"][1]
            obs = FurnaceObservation(float(self.preheater.time_s), Tavg, Tstack, vstack)
        pre_state = self.preheater.state(time_s=obs.time_s)
        resource_state = self.resource_model.from_observation(obs)
        self._last_snapshot = self._make_snapshot(obs.time_s, obs, pre_state, resource_state, command=None)
        return self._last_snapshot

    def _make_snapshot(self, time_s, obs, pre_state, resource_state, command) -> PlantSnapshot:
        diag = getattr(self.preheater, "last_diagnostics", None)
        pre_out = PreheaterOutput(
            time_s=float(time_s),
            omega_out=float(pre_state.omega_out),
            T_solid_out_C=float(pre_state.T_solid_out_C),
            wet_out_kgps=None if diag is None else float(getattr(diag, "wet_out_kgps", float("nan"))),
            dry_out_kgps=None if diag is None else float(getattr(diag, "dry_out_kgps", float("nan"))),
            water_out_kgps=None if diag is None else float(getattr(diag, "water_out_kgps", float("nan"))),
            Tg_out_C=None if diag is None else float(getattr(diag, "Tg_out_C", float("nan"))),
        )
        stack = StackResourceMeasurement(
            time_s=float(time_s),
            T_stack_available_C=float(resource_state.T_stack_available_C),
            v_stack_available_mps=float(resource_state.v_stack_available_mps),
            mdot_stack_available_kgps=float(resource_state.mdot_stack_available_kgps),
        )
        feedback = None
        if command is not None:
            feedback = ActuatorFeedback(
                time_s=float(time_s),
                Tg_actual_C=float(command.Tg_cmd_C),
                vg_actual_mps=float(command.vg_cmd_mps),
                heater_enabled_actual=bool(command.heater_enable),
                command_applied=True,
            )
        return PlantSnapshot(
            time_s=float(time_s),
            furnace_obs=obs,
            preheater_output=pre_out,
            preheater_state=pre_state,
            stack_resource=stack,
            actuator_feedback=feedback,
            health=PlantHealth(ok=True),
            raw={"preheater_diagnostics": diag} if diag is not None else {},
        )

    def step(self, step_input: PlantStepInput) -> PlantSnapshot:
        cmd = step_input.command
        feedstock = step_input.feedstock
        pre_state = self.preheater.step(feedstock, cmd.Tg_cmd_C, cmd.vg_cmd_mps, step_input.dt_s)
        disturbance = None
        if self.disturbance_schedule is not None:
            disturbance = self.disturbance_schedule(step_input.time_s)
        diag = getattr(self.preheater, "last_diagnostics", None)
        furnace_feed = furnace_feed_from_preheater_output(
            time_s=step_input.time_s,
            omega_b=pre_state.omega_out,
            mdot_d_kgps=None if diag is None else getattr(diag, "dry_out_kgps", None),
            mdot_water_kgps=None if diag is None else getattr(diag, "water_out_kgps", None),
            mdot_wet_kgps=None if diag is None else getattr(diag, "wet_out_kgps", None),
            mdot_d_ref_kgps=getattr(self.furnace.cfg, "mdot_d_ref_kgps", 0.052),
        )
        Tavg, Tstack, vstack = self.furnace.step(
            furnace_feed.omega_b,
            mdot_d_kgps=furnace_feed.mdot_d_kgps,
            dt_s=step_input.dt_s,
            disturbance=_normalize_disturbance(disturbance),
        )
        obs = FurnaceObservation(float(step_input.time_s), Tavg, Tstack, vstack)
        resource_state = self.resource_model.from_observation(obs)
        self._last_snapshot = self._make_snapshot(step_input.time_s, obs, pre_state, resource_state, cmd)
        return self._last_snapshot

    def observe(self) -> PlantSnapshot | None:
        return self._last_snapshot
