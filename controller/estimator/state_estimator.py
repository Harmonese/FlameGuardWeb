from __future__ import annotations

"""Controller-side state estimator.

This module is the boundary between external plant observations and the
operator/predictor.  It owns controller-side predictor state and always returns a
StateEstimate with a complete PreheaterState, even when the plant snapshot only
contains limited scalar measurements.
"""

from dataclasses import replace
from typing import Tuple

from domain.types import (
    ActuatorCommand,
    FeedstockObservation,
    FurnaceObservation,
    PlantSnapshot,
    PredictorBundle,
    PreheaterCellState,
    PreheaterState,
    StackResourceMeasurement,
    StateEstimate,
)
from controller.estimator.furnace_disturbance_observer import FurnaceDisturbanceObserver
from controller.predictor.preheater import PreheaterForwardModel
from controller.predictor.furnace import FurnaceDyn, dry_basis_ratio_from_feedstock, furnace_feed_from_preheater_output


class ControllerStateEstimator:
    """Maintain controller-owned predictor states and residual estimates.

    Full preheater snapshots are synchronized directly.  When a backend cannot
    expose 20-cell state, the estimator advances its predictor-side digital twin
    using actuator feedback when available, otherwise the previous command.
    Furnace residuals are estimated here as well, so runtime does not manage a
    nominal furnace predictor or disturbance observer.
    """

    def __init__(
        self,
        preheater_predictor: PreheaterForwardModel,
        furnace_predictor: FurnaceDyn | None = None,
        disturbance_observer: FurnaceDisturbanceObserver | None = None,
        resource_model=None,
    ):
        self.preheater_predictor = preheater_predictor
        self.furnace_predictor = furnace_predictor
        self.disturbance_observer = disturbance_observer or FurnaceDisturbanceObserver()
        self.resource_model = resource_model
        self._last_estimate: StateEstimate | None = None
        self._last_snapshot_time_s: float | None = None
        self._last_nominal_furnace_obs: FurnaceObservation | None = None

    def get_predictor_bundle(self) -> PredictorBundle:
        return PredictorBundle(
            preheater=self.preheater_predictor,
            furnace=self.furnace_predictor,
            resource_model=self.resource_model,
        )

    def reset(self, snapshot: PlantSnapshot, *, feedstock: FeedstockObservation | None = None) -> StateEstimate:
        if snapshot.preheater_state is not None:
            self.preheater_predictor.load_state(snapshot.preheater_state, feedstock=feedstock)
            pre_state = self.preheater_predictor.state(time_s=snapshot.time_s)
            source = "snapshot_preheater_state"
            confidence = 1.0
        else:
            pre_state = self.preheater_predictor.state(time_s=snapshot.time_s)
            pre_state = self._apply_preheater_output_correction(pre_state, snapshot)
            self.preheater_predictor.load_state(pre_state, feedstock=feedstock)
            source = "predictor_prior_no_snapshot_state"
            confidence = 0.5
        estimate = self._make_estimate(snapshot, pre_state, source=source, confidence=confidence)
        self._last_estimate = estimate
        self._last_snapshot_time_s = float(snapshot.time_s)
        return estimate

    def update(
        self,
        snapshot: PlantSnapshot,
        previous_command: ActuatorCommand,
        feedstock: FeedstockObservation,
        *,
        dt_s: float | None = None,
        disturbance_est: Tuple[float, float, float] | None = None,
    ) -> StateEstimate:
        if snapshot.preheater_state is not None:
            self.preheater_predictor.load_state(snapshot.preheater_state, feedstock=feedstock)
            pre_state = self.preheater_predictor.state(time_s=snapshot.time_s)
            source = "snapshot_preheater_state"
            confidence = min(1.0, max(0.0, float(feedstock.confidence)))
        else:
            dt = self._infer_dt(snapshot, dt_s)
            effective_cmd = self._command_from_feedback(previous_command, snapshot)
            if self._last_estimate is None:
                self.preheater_predictor.load_state(
                    self.preheater_predictor.state(time_s=snapshot.time_s),
                    feedstock=feedstock,
                )
            pre_state = self.preheater_predictor.step(
                feedstock,
                effective_cmd.Tg_cmd_C,
                effective_cmd.vg_cmd_mps,
                dt,
            )
            pre_state = self._apply_preheater_output_correction(pre_state, snapshot)
            self.preheater_predictor.load_state(pre_state, feedstock=feedstock)
            source = "predictor_digital_twin"
            confidence = 0.65 * min(1.0, max(0.0, float(feedstock.confidence)))
            if snapshot.actuator_feedback is not None and snapshot.actuator_feedback.command_applied:
                confidence = max(confidence, 0.70)
            if snapshot.preheater_output is not None:
                if snapshot.preheater_output.omega_out is not None or snapshot.preheater_output.T_solid_out_C is not None:
                    confidence = max(confidence, 0.78)

        if disturbance_est is not None:
            d_tuple = tuple(float(x) for x in disturbance_est)
        else:
            d_tuple = self._update_furnace_disturbance(snapshot, pre_state, feedstock=feedstock, dt_s=dt_s)
        estimate = self._make_estimate(snapshot, pre_state, source=source, confidence=confidence, disturbance=d_tuple)
        self._last_estimate = estimate
        self._last_snapshot_time_s = float(snapshot.time_s)
        return estimate

    def with_disturbance(self, estimate: StateEstimate, disturbance_est: Tuple[float, float, float]) -> StateEstimate:
        dT, dS, dV = disturbance_est
        updated = replace(
            estimate,
            disturbance_est_Tavg_C=float(dT),
            disturbance_est_Tstack_C=float(dS),
            disturbance_est_vstack_mps=float(dV),
        )
        self._last_estimate = updated
        return updated

    def _make_estimate(
        self,
        snapshot: PlantSnapshot,
        pre_state: PreheaterState,
        *,
        source: str,
        confidence: float,
        disturbance: Tuple[float, float, float] | None = None,
    ) -> StateEstimate:
        dT, dS, dV = disturbance if disturbance is not None else (0.0, 0.0, 0.0)
        return StateEstimate(
            time_s=float(snapshot.time_s),
            furnace_obs=snapshot.furnace_obs,
            preheater_state_est=pre_state,
            stack_resource_est=self._resource_from_snapshot(snapshot),
            disturbance_est_Tavg_C=float(dT),
            disturbance_est_Tstack_C=float(dS),
            disturbance_est_vstack_mps=float(dV),
            confidence=float(max(0.0, min(1.0, confidence))),
            source=source,
        )


    def _furnace_feed_from_state(self, pre_state: PreheaterState, snapshot: PlantSnapshot) -> object:
        out = snapshot.preheater_output
        mdot_d = None if out is None else out.dry_out_kgps
        mdot_w = None if out is None else out.water_out_kgps
        mdot_wet = None if out is None else out.wet_out_kgps
        if mdot_d is None and pre_state.cells:
            mdot_d = pre_state.cells[-1].dry_mass_kg / max(self.preheater_predictor.tau_cell_s, 1e-12)
        if mdot_w is None and pre_state.cells:
            mdot_w = pre_state.cells[-1].water_mass_kg / max(self.preheater_predictor.tau_cell_s, 1e-12)
        if mdot_wet is None and mdot_d is not None and mdot_w is not None:
            mdot_wet = mdot_d + mdot_w
        return furnace_feed_from_preheater_output(
            time_s=snapshot.time_s,
            omega_b=pre_state.omega_out,
            mdot_d_kgps=mdot_d,
            mdot_water_kgps=mdot_w,
            mdot_wet_kgps=mdot_wet,
            mdot_d_ref_kgps=getattr(self.furnace_predictor.cfg, "mdot_d_ref_kgps", 0.052) if self.furnace_predictor is not None else 0.052,
        )

    def _update_furnace_disturbance(
        self,
        snapshot: PlantSnapshot,
        pre_state: PreheaterState,
        *,
        feedstock: FeedstockObservation,
        dt_s: float | None,
    ) -> Tuple[float, float, float]:
        if self.furnace_predictor is None:
            return (0.0, 0.0, 0.0)
        dt = self._infer_dt(snapshot, dt_s)
        furnace_feed = self._furnace_feed_from_state(pre_state, snapshot)
        nom_Tavg, nom_Tstack, nom_vstack = self.furnace_predictor.step(
            furnace_feed.omega_b,
            mdot_d_kgps=furnace_feed.mdot_d_kgps,
            dt_s=dt,
            disturbance=None,
        )
        nominal = FurnaceObservation(
            time_s=float(snapshot.time_s),
            T_avg_C=float(nom_Tavg),
            T_stack_C=float(nom_Tstack),
            v_stack_mps=float(nom_vstack),
        )
        self._last_nominal_furnace_obs = nominal
        return self.disturbance_observer.update(snapshot.furnace_obs, nominal).as_tuple()

    def _command_from_feedback(self, previous_command: ActuatorCommand, snapshot: PlantSnapshot) -> ActuatorCommand:
        feedback = snapshot.actuator_feedback
        if feedback is None or not feedback.command_applied:
            return previous_command
        Tg = previous_command.Tg_cmd_C if feedback.Tg_actual_C is None else float(feedback.Tg_actual_C)
        vg = previous_command.vg_cmd_mps if feedback.vg_actual_mps is None else float(feedback.vg_actual_mps)
        heater = previous_command.heater_enable if feedback.heater_enabled_actual is None else bool(feedback.heater_enabled_actual)
        return replace(previous_command, Tg_cmd_C=Tg, vg_cmd_mps=vg, heater_enable=heater)

    def _apply_preheater_output_correction(self, pre_state: PreheaterState, snapshot: PlantSnapshot) -> PreheaterState:
        out = snapshot.preheater_output
        if out is None:
            return pre_state
        omega_out = pre_state.omega_out if out.omega_out is None else float(out.omega_out)
        T_out = pre_state.T_solid_out_C if out.T_solid_out_C is None else float(out.T_solid_out_C)
        cells = list(pre_state.cells)
        if cells:
            last = cells[-1]
            cells[-1] = replace(
                last,
                omega=omega_out,
                T_solid_C=T_out,
                water_mass_kg=self._water_mass_from_omega(last.dry_mass_kg, omega_out),
            )
        return replace(
            pre_state,
            cells=tuple(cells),
            omega_out=omega_out,
            T_solid_out_C=T_out,
        )

    @staticmethod
    def _water_mass_from_omega(dry_mass_kg: float, omega_wb: float) -> float:
        omega = min(max(float(omega_wb), 0.0), 0.999999)
        dry = max(float(dry_mass_kg), 0.0)
        return dry * omega / max(1.0 - omega, 1e-9)

    def _infer_dt(self, snapshot: PlantSnapshot, dt_s: float | None) -> float:
        if dt_s is not None:
            return max(float(dt_s), 1e-9)
        if self._last_snapshot_time_s is not None:
            return max(float(snapshot.time_s) - float(self._last_snapshot_time_s), 1e-9)
        return 1e-9

    @staticmethod
    def _resource_from_snapshot(snapshot: PlantSnapshot) -> StackResourceMeasurement:
        if snapshot.stack_resource is not None:
            return snapshot.stack_resource
        obs = snapshot.furnace_obs
        return StackResourceMeasurement(
            time_s=float(snapshot.time_s),
            T_stack_available_C=float(obs.T_stack_C),
            v_stack_available_mps=float(obs.v_stack_mps),
            mdot_stack_available_kgps=None,
        )
