from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence
import csv
import time

import numpy as np

from domain.types import ActuatorCommand, PlantStepInput, ControlSetpoint, OperatorContext, ResourceBoundary, StackResourceMeasurement
from controller.predictor.feed_preview import KnownScheduleFeedPreview
from controller.factory import make_executor, make_operator, make_predictor_resource_model, make_state_estimator
from plant.python_model.furnace import furnace_feed_from_preheater_output, furnace_static_outputs_from_inputs
from plant.factory import make_plant_backend
from plant.python_model.material_model import feedstock_from_composition


REFERENCE_TAVG_BAND = (850.0, 895.99)
SAFE_TAVG_BAND = (850.0, 1100.0)
SUPERVISION_TAVG_BAND = (845.0, 1105.0)
REFERENCE_OMEGA_BAND = (0.3042675804697915, 0.3393469511577442)



@dataclass
class EventWindow:
    start_s: float
    end_s: float
    label: str


@dataclass
class SimConfig:
    dt_meas_s: float = 0.1
    dt_ctrl_s: float = 1.0
    dt_opt_s: float = 2.0
    total_time_s: float = 1200.0
    omega_ref: float = 0.3218
    T_target_C: float = 873.0
    T_compliance_min_C: float = 850.0
    daily_wet_feed_kg: float = 20000.0
    wet_mass_flow_override_kgps: float | None = None
    pre_dead_s: float = 5.0
    pre_tau_s: float = 985.0
    furnace_dead_s: float = 5.0
    furnace_tau1_s: float = 0.223
    furnace_tau2_s: float = 75.412
    resource_v_stack_cap_mps: float = 18.0
    plant_backend: str = "python"
    control_mode: str = "nmpc"
    mpc_dt_s: float = 20.0
    mpc_horizon_s: float = 600.0
    preheater_n_cells: int = 20

    # NMPC / initialization options. The nominal command is the full-preheater
    # steady-hold point used to synchronize preheater inventory, actuator limiter,
    # and controller warm start.
    nominal_Tg_C: float = 800.0
    nominal_vg_mps: float = 12.0
    preheater_warmup_s: float = 6.0 * 985.0
    preheater_warmup_dt_s: float = 20.0
    nmpc_reoptimize_s: float = 60.0
    # NMPC can keep its 20 s decision grid, while internally integrating
    # the distributed preheater with a smaller substep for prediction/plant
    # consistency.
    nmpc_rollout_dt_s: float = 5.0
    nmpc_maxiter: int = 20
    # Deployment-style non-blocking NMPC. Keep False for offline studies that
    # intentionally advance simulated time only after each SLSQP solve.
    nmpc_async: bool = False
    nmpc_async_stale_plan_timeout_s: float = 300.0

    # Optional real-time compute-latency simulation. The default "none" keeps
    # historical ideal-simulation behavior: simulated plant time does not advance
    # while the controller spends wall-clock time in SLSQP. "profile" advances
    # the plant by the measured operator wall time; "fixed" advances by
    # fixed_compute_latency_s for controlled what-if tests.
    compute_latency_mode: str = "none"  # "none", "profile", or "fixed"
    fixed_compute_latency_s: float = 0.0
    compute_latency_scale: float = 1.0
    max_simulated_compute_latency_s: float | None = None
    compute_latency_step_s: float | None = None

    # Dynamic resource / auxiliary heat model. Natural resources come from
    # predicted/measured T_stack and v_stack; auxiliary heat can raise Tg up to
    # aux_Tg_max_C but cannot create extra mass flow.
    stack_to_preheater_loss_C: float = 0.0
    extractable_velocity_fraction: float = 1.0
    aux_Tg_max_C: float = 1100.0

    # Disturbance observer: NMPC receives estimated additive disturbance, not
    # the scenario's ground-truth disturbance.
    disturbance_observer_alpha: float = 0.05

    case_name: str = "case"
    out_dir: str = "runtime/results"
    notes: str = ""
    event_start_s: float | None = None
    event_end_s: float | None = None
    tail_window_s: float = 1200.0
    settle_hold_safe_s: float = 300.0
    settle_hold_ref_s: float = 600.0
    overshoot_deadband_C: float = 2.0

    # Safety recovery guard. This is a controller/executor guard rail, not the
    # economic NMPC objective. Hysteresis prevents on/off chatter near 850 C.
    recovery_guard_enabled: bool = True
    recovery_guard_exit_C: float = 865.0
    recovery_guard_min_hold_s: float = 60.0

    # More realistic initialization support than the ad-hoc cold-start note.
    furnace_init_mode: str = "warm"   # "warm" or "custom"
    T_avg_init_C: float | None = None
    T_stack_init_C: float | None = None
    v_stack_init_mps: float | None = None
    omega_out_init: float | None = None

    @property
    def wet_mass_flow_kgps(self) -> float:
        # Default wet-feed mass flow is the plant daily treatment capacity
        # converted to a continuous kg/s stream.  Use the override only for
        # controlled what-if studies.
        if self.wet_mass_flow_override_kgps is not None:
            return float(self.wet_mass_flow_override_kgps)
        return float(self.daily_wet_feed_kg) / 86400.0

    @property
    def T_set_C(self) -> float:
        # Main compliance/control target is a temperature target, not the
        # furnace surrogate value implied by a fixed omega_ref.
        return float(self.T_target_C)


@dataclass
class History:
    t: List[float] = field(default_factory=list)
    T_avg: List[float] = field(default_factory=list)
    T_stack: List[float] = field(default_factory=list)
    v_stack: List[float] = field(default_factory=list)
    T_set: List[float] = field(default_factory=list)
    omega_est: List[float] = field(default_factory=list)
    slot_omega0: List[float] = field(default_factory=list)
    omega_opt: List[float] = field(default_factory=list)
    omega_out: List[float] = field(default_factory=list)
    Tg_cmd: List[float] = field(default_factory=list)
    vg_cmd: List[float] = field(default_factory=list)
    Q_aux_heat: List[float] = field(default_factory=list)
    Tavg_pred: List[float] = field(default_factory=list)
    error_C: List[float] = field(default_factory=list)
    disturbance_Tavg: List[float] = field(default_factory=list)
    disturbance_Tstack: List[float] = field(default_factory=list)
    disturbance_vstack: List[float] = field(default_factory=list)
    feed_x1: List[float] = field(default_factory=list)
    feed_x2: List[float] = field(default_factory=list)
    feed_x3: List[float] = field(default_factory=list)
    feed_x4: List[float] = field(default_factory=list)
    feed_x5: List[float] = field(default_factory=list)
    feed_x6: List[float] = field(default_factory=list)
    feed_moisture_wb: List[float] = field(default_factory=list)
    feed_tref_min: List[float] = field(default_factory=list)
    feed_slope_min_per_C: List[float] = field(default_factory=list)
    feed_bulk_density_kg_m3: List[float] = field(default_factory=list)
    feed_wet_mass_flow_kgps: List[float] = field(default_factory=list)
    mdot_preheater_wet_in_kgps: List[float] = field(default_factory=list)
    mdot_furnace_dry_kgps: List[float] = field(default_factory=list)
    mdot_furnace_water_kgps: List[float] = field(default_factory=list)
    mdot_furnace_wet_kgps: List[float] = field(default_factory=list)
    rd_used: List[float] = field(default_factory=list)
    omega_target_from_T: List[float] = field(default_factory=list)
    T_target_C: List[float] = field(default_factory=list)
    T_compliance_min_C: List[float] = field(default_factory=list)
    T_static_from_current_furnace_feed: List[float] = field(default_factory=list)
    control_source: List[str] = field(default_factory=list)
    control_note: List[str] = field(default_factory=list)
    T_stack_available: List[float] = field(default_factory=list)
    v_stack_available: List[float] = field(default_factory=list)
    mdot_stack_available: List[float] = field(default_factory=list)
    mdot_preheater_cmd: List[float] = field(default_factory=list)
    aux_resource_required: List[float] = field(default_factory=list)
    aux_heat_enable: List[float] = field(default_factory=list)
    mdot_aux_flow: List[float] = field(default_factory=list)
    fan_circulation_power: List[float] = field(default_factory=list)
    disturbance_est_Tavg: List[float] = field(default_factory=list)
    disturbance_est_Tstack: List[float] = field(default_factory=list)
    disturbance_est_vstack: List[float] = field(default_factory=list)

    # Current-architecture diagnostics.
    Tg_ref: List[float] = field(default_factory=list)
    vg_ref: List[float] = field(default_factory=list)
    Tg_ref_minus_cmd: List[float] = field(default_factory=list)
    vg_ref_minus_cmd: List[float] = field(default_factory=list)
    operator_cost: List[float] = field(default_factory=list)
    operator_feasible: List[float] = field(default_factory=list)
    nmpc_plan_age_s: List[float] = field(default_factory=list)
    nmpc_last_solve_ms: List[float] = field(default_factory=list)
    nmpc_async_job_running: List[float] = field(default_factory=list)
    nmpc_stale_plan: List[float] = field(default_factory=list)
    operator_compute_wall_s: List[float] = field(default_factory=list)
    simulated_compute_latency_s: List[float] = field(default_factory=list)
    command_apply_delay_s: List[float] = field(default_factory=list)
    async_submitted_jobs: List[float] = field(default_factory=list)
    async_accepted_plans: List[float] = field(default_factory=list)
    async_discarded_plans: List[float] = field(default_factory=list)
    async_last_result_state_age_s: List[float] = field(default_factory=list)
    safety_reachable: List[float] = field(default_factory=list)
    nmpc_pred_min_Tavg: List[float] = field(default_factory=list)
    nmpc_pred_max_Tavg: List[float] = field(default_factory=list)
    omega_max_for_safety: List[float] = field(default_factory=list)
    safety_margin_C: List[float] = field(default_factory=list)
    recovery_guard_requested: List[float] = field(default_factory=list)
    recovery_guard_active: List[float] = field(default_factory=list)
    recovery_guard_reason: List[str] = field(default_factory=list)

    T_solid_out: List[float] = field(default_factory=list)
    outlet_cell_omega0: List[float] = field(default_factory=list)
    outlet_cell_tref_min: List[float] = field(default_factory=list)
    outlet_cell_slope_min_per_C: List[float] = field(default_factory=list)
    mdot_d_out: List[float] = field(default_factory=list)
    mdot_w_out: List[float] = field(default_factory=list)
    mdot_wet_out: List[float] = field(default_factory=list)
    water_evap_rate: List[float] = field(default_factory=list)
    inventory_dry: List[float] = field(default_factory=list)
    inventory_water: List[float] = field(default_factory=list)
    inventory_total: List[float] = field(default_factory=list)
    Tg_in: List[float] = field(default_factory=list)
    Tg_out: List[float] = field(default_factory=list)
    mdot_gas_preheater: List[float] = field(default_factory=list)
    U_eff: List[float] = field(default_factory=list)
    Q_gas_to_solid: List[float] = field(default_factory=list)
    Q_sensible: List[float] = field(default_factory=list)
    Q_latent: List[float] = field(default_factory=list)
    heat_balance_residual: List[float] = field(default_factory=list)

    cell_snapshots: List[dict] = field(default_factory=list)


def _make_command(time_s: float, Tg: float, vg: float) -> ActuatorCommand:
    return ActuatorCommand(
        time_s=float(time_s),
        Tg_cmd_C=float(Tg),
        vg_cmd_mps=float(vg),
        heater_enable=False,
        Q_aux_heat_kW=0.0,
        aux_resource_required=False,
        aux_heat_required=False,
        aux_circulation_required=False,
    )


def _normalize_disturbance(disturbance):
    if disturbance is None:
        return 0.0, 0.0, 0.0
    if isinstance(disturbance, (tuple, list)) and len(disturbance) == 3:
        return float(disturbance[0]), float(disturbance[1]), float(disturbance[2])
    return float(disturbance), 0.0, 0.0




def _simulated_compute_latency_s(cfg: SimConfig, measured_wall_s: float) -> float:
    """Return the simulated plant-time delay caused by controller compute.

    The default mode intentionally returns zero to preserve historical ideal
    simulation.  Latency-aware studies can use either the measured wall time or
    a fixed delay so scenarios remain reproducible across machines.
    """
    mode = str(getattr(cfg, "compute_latency_mode", "none") or "none").lower()
    if mode in {"", "none", "ideal", "off"}:
        return 0.0
    if mode == "fixed":
        delay = float(getattr(cfg, "fixed_compute_latency_s", 0.0) or 0.0)
    elif mode == "profile":
        delay = float(measured_wall_s) * float(getattr(cfg, "compute_latency_scale", 1.0) or 1.0)
    else:
        raise ValueError(f"Unknown compute_latency_mode={mode!r}; expected none/profile/fixed")
    delay = max(0.0, delay)
    cap = getattr(cfg, "max_simulated_compute_latency_s", None)
    if cap is not None:
        delay = min(delay, max(0.0, float(cap)))
    return delay


def _advance_plant_during_compute_latency(
    *,
    plant,
    state_estimator: ControllerStateEstimator,
    feedstock_schedule,
    command: ActuatorCommand,
    start_time_s: float,
    delay_s: float,
    cfg: SimConfig,
):
    """Advance plant while the controller is assumed to be computing.

    During compute latency the newly solved command is not yet available, so the
    previous command keeps acting on the plant.  The delay is split into small
    substeps so feed/disturbance schedules and the estimator do not jump across
    events in one large integration step.
    """
    remaining = max(0.0, float(delay_s))
    if remaining <= 0.0:
        return None
    max_step = getattr(cfg, "compute_latency_step_s", None)
    if max_step is None or max_step <= 0.0:
        max_step = cfg.dt_meas_s
    max_step = max(float(max_step), 1e-9)
    current_t = float(start_time_s)
    snapshot = None
    estimate = None
    feedstock = None
    while remaining > 1e-12:
        dt = min(max_step, remaining)
        current_t += dt
        feedstock = feedstock_schedule(current_t)
        snapshot = plant.step(PlantStepInput(time_s=current_t, dt_s=dt, command=command, feedstock=feedstock))
        estimate = state_estimator.update(
            snapshot,
            previous_command=command,
            feedstock=feedstock,
            dt_s=dt,
        )
        remaining -= dt
    return current_t, feedstock, snapshot, estimate

def _should_record_cell_snapshot(t: float, cfg: SimConfig) -> bool:
    points = {0.0, cfg.total_time_s, max(0.0, cfg.total_time_s - cfg.tail_window_s)}
    if cfg.event_start_s is not None:
        points.add(float(cfg.event_start_s))
    if cfg.event_end_s is not None:
        points.add(float(cfg.event_end_s))
    return any(abs(t - p) <= 0.5 * cfg.dt_meas_s + 1e-9 for p in points)


def _append_cell_snapshot(hist: History, t: float, pre_state) -> None:
    for cell in pre_state.cells:
        hist.cell_snapshots.append({
            'time_s': float(t),
            'cell_index': cell.index,
            'z_center_m': cell.z_center_m,
            'residence_left_s': cell.residence_left_s,
            'omega': cell.omega,
            'T_solid_C': cell.T_solid_C,
            'omega0': cell.omega0,
            'tref_min': cell.tref_min,
            'slope_min_per_C': cell.slope_min_per_c,
            'dry_mass_kg': cell.dry_mass_kg,
            'water_mass_kg': cell.water_mass_kg,
        })



def run_case(name: str, composition_schedule, disturbance_schedule=None, cfg: SimConfig | None = None):
    """Run a scenario using the Python plant backend and operator NMPC path.

    Current path:
      python plant model -> disturbance/resource estimation -> operator NMPC -> executor.
    """
    cfg = cfg or SimConfig(case_name=name)
    hist = History()
    tb = make_executor(cfg)
    predictor_resource_model = make_predictor_resource_model(cfg)
    resource = ResourceBoundary(cfg.aux_Tg_max_C, min(cfg.resource_v_stack_cap_mps, 12.0))

    def feedstock_schedule(time_s: float):
        return feedstock_from_composition(
            time_s,
            composition_schedule(time_s),
            wet_mass_flow_kgps=cfg.wet_mass_flow_kgps,
            source="scenario_composition_schedule",
        )

    feed_preview = KnownScheduleFeedPreview(feedstock_schedule)
    initial_feedstock = feedstock_schedule(0.0)
    plant = make_plant_backend(
        cfg,
        initial_feedstock=initial_feedstock,
        disturbance_schedule=disturbance_schedule,
    )

    # Controller-side predictor/estimator state.  The operator receives only
    # StateEstimate + predictor models, never the plant model instances.
    initial_snapshot = plant.reset()

    state_estimator = make_state_estimator(
        cfg,
        initial_snapshot=initial_snapshot,
        initial_feedstock=initial_feedstock,
        resource_model=predictor_resource_model,
    )

    mpc = make_operator(cfg, resource_model=predictor_resource_model)
    mpc.initialize(Tg_ref_C=cfg.nominal_Tg_C, vg_ref_mps=cfg.nominal_vg_mps, time_s=0.0)

    # Consistent warm initialization: actuator limiter, current command, and NMPC
    # warm start all begin from the same nominal hold point. This removes the
    # artificial initial kick caused by starting the plant near omega_ref while
    # commanding only ~215-300 C gas.
    if hasattr(tb, 'initialize_previous'):
        tb.initialize_previous(cfg.nominal_Tg_C, cfg.nominal_vg_mps)
    current_cmd = _make_command(0.0, cfg.nominal_Tg_C, cfg.nominal_vg_mps)
    current_decision = None
    recovery_guard_latched = False
    recovery_guard_enter_time_s: float | None = None
    last_recovery_guard_requested = False
    last_recovery_guard_active = False
    last_recovery_guard_reason = ""

    opt_elapsed = 0.0
    t = 0.0
    while t <= cfg.total_time_s + 1e-9:
        comp = composition_schedule(t)
        feedstock = feedstock_schedule(t)
        disturbance = None if disturbance_schedule is None else disturbance_schedule(t)
        d_avg, d_stack, d_v = _normalize_disturbance(disturbance)

        snapshot = plant.step(PlantStepInput(time_s=t, dt_s=cfg.dt_meas_s, command=current_cmd, feedstock=feedstock))
        obs = snapshot.furnace_obs
        Tavg, Tstack, vstack = obs.T_avg_C, obs.T_stack_C, obs.v_stack_mps
        estimate = state_estimator.update(
            snapshot,
            previous_command=current_cmd,
            feedstock=feedstock,
            dt_s=cfg.dt_meas_s,
        )
        pre_state = estimate.preheater_state_est
        omega_out = pre_state.omega_out
        stack_resource = estimate.stack_resource_est or snapshot.stack_resource or StackResourceMeasurement(t, Tstack, vstack, None)
        mdot_stack_available = (
            float(stack_resource.mdot_stack_available_kgps)
            if stack_resource.mdot_stack_available_kgps is not None
            else float("inf")
        )
        resource = ResourceBoundary(cfg.aux_Tg_max_C, min(cfg.resource_v_stack_cap_mps, 12.0))

        operator_wall_s = 0.0
        simulated_compute_latency_s = 0.0
        command_apply_delay_s = 0.0

        if opt_elapsed <= 1e-12:
            pre_compute_cmd = current_cmd
            op_t0 = time.perf_counter()
            current_decision = mpc.step_context(OperatorContext(
                estimate=estimate,
                predictors=state_estimator.get_predictor_bundle(),
                feedstock=feedstock,
                resource=resource,
                previous_command=current_cmd,
                feed_preview=feed_preview,
            ))
            operator_wall_s = time.perf_counter() - op_t0
            simulated_compute_latency_s = _simulated_compute_latency_s(cfg, operator_wall_s)
            command_apply_delay_s = simulated_compute_latency_s

            if simulated_compute_latency_s > 1e-12:
                advanced = _advance_plant_during_compute_latency(
                    plant=plant,
                    state_estimator=state_estimator,
                    feedstock_schedule=feedstock_schedule,
                    command=pre_compute_cmd,
                    start_time_s=t,
                    delay_s=simulated_compute_latency_s,
                    cfg=cfg,
                )
                if advanced is not None:
                    t, feedstock, snapshot, estimate = advanced
                    comp = composition_schedule(t)
                    disturbance = None if disturbance_schedule is None else disturbance_schedule(t)
                    d_avg, d_stack, d_v = _normalize_disturbance(disturbance)
                    obs = snapshot.furnace_obs
                    Tavg, Tstack, vstack = obs.T_avg_C, obs.T_stack_C, obs.v_stack_mps
                    pre_state = estimate.preheater_state_est
                    omega_out = pre_state.omega_out
                    stack_resource = estimate.stack_resource_est or snapshot.stack_resource or StackResourceMeasurement(t, Tstack, vstack, None)
                    mdot_stack_available = (
                        float(stack_resource.mdot_stack_available_kgps)
                        if stack_resource.mdot_stack_available_kgps is not None
                        else float("inf")
                    )
                    # Keep the active cached plan but make the note explicitly
                    # show that it was computed from a pre-latency state.
                    if current_decision is not None:
                        current_decision = type(current_decision)(
                            **{
                                **current_decision.__dict__,
                                "note": (current_decision.note + "; " if current_decision.note else "")
                                + f"command applied after simulated_compute_latency_s={simulated_compute_latency_s:.3f}",
                            }
                        )

            raw_recovery_request = bool(
                cfg.recovery_guard_enabled
                and (
                    Tavg < cfg.T_compliance_min_C
                    or (
                        current_decision is not None
                        and not getattr(current_decision, "safety_reachable", True)
                        and Tavg < cfg.T_set_C
                    )
                )
            )
            safety_reachable_now = bool(
                current_decision is None or getattr(current_decision, "safety_reachable", True)
            )
            if raw_recovery_request and not recovery_guard_latched:
                recovery_guard_latched = True
                recovery_guard_enter_time_s = float(t)
            hold_elapsed_s = (
                float("inf")
                if recovery_guard_enter_time_s is None
                else max(0.0, float(t) - float(recovery_guard_enter_time_s))
            )
            if recovery_guard_latched:
                hold_required = hold_elapsed_s < max(0.0, float(cfg.recovery_guard_min_hold_s))
                exit_ready = (
                    not raw_recovery_request
                    and not hold_required
                    and Tavg >= float(cfg.recovery_guard_exit_C)
                    and safety_reachable_now
                )
                if exit_ready:
                    recovery_guard_latched = False
                    recovery_guard_enter_time_s = None
            recovery_guard_requested = bool(raw_recovery_request)
            recovery_guard_active = bool(recovery_guard_latched)
            recovery_guard_reason = ""
            if recovery_guard_active:
                if Tavg < cfg.T_compliance_min_C:
                    recovery_guard_reason = f"T_avg below compliance floor ({Tavg:.2f} < {cfg.T_compliance_min_C:.2f})"
                elif current_decision is not None and not getattr(current_decision, "safety_reachable", True):
                    recovery_guard_reason = "NMPC predicted safety not reachable; recovery guard latched"
                elif hold_elapsed_s < max(0.0, float(cfg.recovery_guard_min_hold_s)):
                    recovery_guard_reason = f"recovery guard minimum hold active ({hold_elapsed_s:.1f}s elapsed)"
                elif Tavg < float(cfg.recovery_guard_exit_C):
                    recovery_guard_reason = f"recovery guard hysteresis active ({Tavg:.2f} < exit {cfg.recovery_guard_exit_C:.2f})"
                else:
                    recovery_guard_reason = "recovery guard latched"
            last_recovery_guard_requested = recovery_guard_requested
            last_recovery_guard_active = recovery_guard_active
            last_recovery_guard_reason = recovery_guard_reason
            setpoint = ControlSetpoint(
                time_s=t,
                Tg_ref_C=current_decision.Tg_ref_C,
                vg_ref_mps=current_decision.vg_ref_mps,
                source=current_decision.source,
                omega_target=current_decision.omega_target,
                omega_reachable=current_decision.omega_reachable,
                mdot_stack_cap_kgps=mdot_stack_available,
                T_stack_available_C=stack_resource.T_stack_available_C,
                v_stack_available_mps=stack_resource.v_stack_available_mps,
                recovery_guard_requested=recovery_guard_active,
                recovery_guard_reason=recovery_guard_reason,
            )
            current_cmd = tb.translate_setpoint(setpoint)
            if getattr(current_cmd, "recovery_guard_active", False) and current_decision is not None:
                current_decision = type(current_decision)(
                    **{
                        **current_decision.__dict__,
                        "source": current_decision.source + "_recovery_guard",
                        "note": (current_decision.note + "; " if current_decision.note else "") + recovery_guard_reason,
                    }
                )

        # Diagnostic equivalent moisture inferred from measured Tavg for plots/metrics.
        omega_est = float(np.clip((Tavg - 1294.871365) / (100.0 * -13.109632), 0.05, 0.95))
        omega_target = current_decision.omega_target if current_decision is not None else cfg.omega_ref
        pred_T = current_decision.predicted_Tavg_C if current_decision is not None else np.nan

        hist.t.append(t)
        hist.T_avg.append(Tavg)
        hist.T_stack.append(Tstack)
        hist.v_stack.append(vstack)
        hist.T_set.append(cfg.T_set_C)
        hist.slot_omega0.append(float(pre_state.cells[-1].omega0))
        hist.omega_opt.append(float(omega_target))
        hist.omega_out.append(float(omega_out))
        hist.Tg_cmd.append(current_cmd.Tg_cmd_C)
        hist.vg_cmd.append(current_cmd.vg_cmd_mps)
        hist.Q_aux_heat.append(current_cmd.Q_aux_heat_kW)
        hist.disturbance_Tavg.append(d_avg)
        hist.disturbance_Tstack.append(d_stack)
        hist.disturbance_vstack.append(d_v)
        feed_vals = list(comp) + [0.0] * max(0, 6 - len(comp))
        hist.feed_x1.append(feed_vals[0]); hist.feed_x2.append(feed_vals[1]); hist.feed_x3.append(feed_vals[2])
        hist.feed_x4.append(feed_vals[3]); hist.feed_x5.append(feed_vals[4]); hist.feed_x6.append(feed_vals[5])
        hist.feed_moisture_wb.append(float(feedstock.moisture_wb))
        hist.feed_tref_min.append(float(feedstock.drying_time_ref_min))
        hist.feed_slope_min_per_C.append(float(feedstock.drying_sensitivity_min_per_C))
        hist.feed_bulk_density_kg_m3.append(float(feedstock.bulk_density_kg_m3) if feedstock.bulk_density_kg_m3 is not None else float('nan'))
        hist.feed_wet_mass_flow_kgps.append(float(feedstock.wet_mass_flow_kgps) if feedstock.wet_mass_flow_kgps is not None else float('nan'))
        hist.mdot_preheater_wet_in_kgps.append(float(feedstock.wet_mass_flow_kgps) if feedstock.wet_mass_flow_kgps is not None else float('nan'))
        hist.omega_est.append(omega_est)
        hist.Tavg_pred.append(float(pred_T))
        hist.error_C.append(cfg.T_set_C - Tavg)
        hist.control_source.append(current_decision.source if current_decision is not None else '')
        hist.control_note.append(current_decision.note if current_decision is not None else '')
        hist.T_stack_available.append(float(stack_resource.T_stack_available_C))
        hist.v_stack_available.append(float(stack_resource.v_stack_available_mps))
        hist.mdot_stack_available.append(float(mdot_stack_available))
        hist.mdot_preheater_cmd.append(float(getattr(current_cmd, 'mdot_preheater_kgps', float('nan'))))
        hist.aux_resource_required.append(1.0 if current_cmd.aux_resource_required else 0.0)
        hist.aux_heat_enable.append(1.0 if current_cmd.heater_enable else 0.0)
        hist.mdot_aux_flow.append(float(getattr(current_cmd, 'mdot_aux_flow_kgps', 0.0)))
        hist.fan_circulation_power.append(float(getattr(current_cmd, 'fan_circulation_power_kW', 0.0)))
        hist.disturbance_est_Tavg.append(float(estimate.disturbance_est_Tavg_C))
        hist.disturbance_est_Tstack.append(float(estimate.disturbance_est_Tstack_C))
        hist.disturbance_est_vstack.append(float(estimate.disturbance_est_vstack_mps))

        # Current-architecture diagnostics.
        hist.Tg_ref.append(float(current_decision.Tg_ref_C if current_decision is not None else np.nan))
        hist.vg_ref.append(float(current_decision.vg_ref_mps if current_decision is not None else np.nan))
        hist.Tg_ref_minus_cmd.append(float((current_decision.Tg_ref_C if current_decision is not None else np.nan) - current_cmd.Tg_cmd_C))
        hist.vg_ref_minus_cmd.append(float((current_decision.vg_ref_mps if current_decision is not None else np.nan) - current_cmd.vg_cmd_mps))
        hist.operator_cost.append(float(current_decision.cost if current_decision is not None else np.nan))
        hist.operator_feasible.append(1.0 if (current_decision is not None and current_decision.feasible) else 0.0)
        status = getattr(mpc, 'status', None)
        solve_profile = getattr(mpc, 'last_solve_profile', None)
        plan_age = getattr(status, 'active_plan_age_s', None) if status is not None else None
        last_solve = getattr(status, 'last_solve_ms', None) if status is not None else (getattr(solve_profile, 'total_ms', None) if solve_profile is not None else None)
        job_running = bool(getattr(status, 'job_running', False)) if status is not None else False
        stale = bool(plan_age is not None and plan_age > cfg.nmpc_async_stale_plan_timeout_s)
        hist.nmpc_plan_age_s.append(float(plan_age) if plan_age is not None else np.nan)
        hist.nmpc_last_solve_ms.append(float(last_solve) if last_solve is not None else np.nan)
        hist.nmpc_async_job_running.append(1.0 if job_running else 0.0)
        hist.nmpc_stale_plan.append(1.0 if stale else 0.0)
        hist.operator_compute_wall_s.append(float(operator_wall_s))
        hist.simulated_compute_latency_s.append(float(simulated_compute_latency_s))
        hist.command_apply_delay_s.append(float(command_apply_delay_s))
        hist.async_submitted_jobs.append(float(getattr(status, 'submitted_job_count', 0) if status is not None else 0))
        hist.async_accepted_plans.append(float(getattr(status, 'accepted_plan_count', 0) if status is not None else 0))
        hist.async_discarded_plans.append(float(getattr(status, 'discarded_plan_count', 0) if status is not None else 0))
        result_age = getattr(status, 'last_result_state_age_s', None) if status is not None else None
        hist.async_last_result_state_age_s.append(float(result_age) if result_age is not None else np.nan)
        hist.safety_reachable.append(1.0 if (current_decision is not None and getattr(current_decision, "safety_reachable", True)) else 0.0)
        hist.nmpc_pred_min_Tavg.append(float(getattr(current_decision, "predicted_min_Tavg_C", np.nan)) if current_decision is not None else np.nan)
        hist.nmpc_pred_max_Tavg.append(float(getattr(current_decision, "predicted_max_Tavg_C", np.nan)) if current_decision is not None else np.nan)
        hist.omega_max_for_safety.append(float(getattr(current_decision, "omega_max_for_safety", np.nan)) if current_decision is not None else np.nan)
        hist.safety_margin_C.append(float(getattr(current_decision, "safety_margin_C", np.nan)) if current_decision is not None else np.nan)
        hist.recovery_guard_requested.append(1.0 if last_recovery_guard_requested else 0.0)
        hist.recovery_guard_active.append(1.0 if (last_recovery_guard_active or getattr(current_cmd, "recovery_guard_active", False)) else 0.0)
        hist.recovery_guard_reason.append(last_recovery_guard_reason)

        out_cell = pre_state.cells[-1]
        diag = snapshot.raw.get('preheater_diagnostics') if snapshot.raw is not None else None
        hist.T_solid_out.append(float(pre_state.T_solid_out_C))
        hist.outlet_cell_omega0.append(float(out_cell.omega0))
        hist.outlet_cell_tref_min.append(float(out_cell.tref_min))
        hist.outlet_cell_slope_min_per_C.append(float(out_cell.slope_min_per_c))
        def _diag(name, default=np.nan):
            return float(getattr(diag, name, default)) if diag is not None else float(default)
        mdot_d_out = _diag('dry_out_kgps')
        mdot_w_out = _diag('water_out_kgps')
        mdot_wet_out = _diag('wet_out_kgps')
        hist.mdot_d_out.append(mdot_d_out)
        hist.mdot_w_out.append(mdot_w_out)
        hist.mdot_wet_out.append(mdot_wet_out)
        furnace_feed_diag = furnace_feed_from_preheater_output(
            time_s=t,
            omega_b=float(omega_out),
            mdot_d_kgps=mdot_d_out if np.isfinite(mdot_d_out) else None,
            mdot_water_kgps=mdot_w_out if np.isfinite(mdot_w_out) else None,
            mdot_wet_kgps=mdot_wet_out if np.isfinite(mdot_wet_out) else None,
        )
        hist.mdot_furnace_dry_kgps.append(float(furnace_feed_diag.mdot_d_kgps))
        hist.mdot_furnace_water_kgps.append(float(furnace_feed_diag.mdot_water_kgps))
        hist.mdot_furnace_wet_kgps.append(float(furnace_feed_diag.mdot_wet_kgps))
        hist.rd_used.append(float(furnace_feed_diag.rd))
        hist.omega_target_from_T.append(float(omega_target))
        hist.T_target_C.append(float(cfg.T_set_C))
        hist.T_compliance_min_C.append(float(cfg.T_compliance_min_C))
        hist.T_static_from_current_furnace_feed.append(float(furnace_static_outputs_from_inputs(furnace_feed_diag.omega_b, furnace_feed_diag.rd).T_avg_C))
        hist.water_evap_rate.append(_diag('water_evap_kgps'))
        hist.inventory_dry.append(_diag('inventory_dry_kg'))
        hist.inventory_water.append(_diag('inventory_water_kg'))
        hist.inventory_total.append(_diag('inventory_total_kg'))
        hist.Tg_in.append(_diag('Tg_in_C'))
        hist.Tg_out.append(_diag('Tg_out_C'))
        hist.mdot_gas_preheater.append(_diag('mdot_gas_kgps'))
        hist.U_eff.append(_diag('U_eff_W_m2K'))
        hist.Q_gas_to_solid.append(_diag('Q_gas_to_solid_kW'))
        hist.Q_sensible.append(_diag('Q_sensible_kW'))
        hist.Q_latent.append(_diag('Q_latent_kW'))
        hist.heat_balance_residual.append(_diag('heat_balance_residual_kW'))
        if _should_record_cell_snapshot(t, cfg):
            _append_cell_snapshot(hist, t, pre_state)

        elapsed_this_loop_s = cfg.dt_meas_s + float(simulated_compute_latency_s)
        opt_elapsed += elapsed_this_loop_s
        if opt_elapsed >= cfg.dt_opt_s - 1e-12:
            opt_elapsed = 0.0
        t += cfg.dt_meas_s
    if hasattr(mpc, 'shutdown'):
        mpc.shutdown(wait=True)
    return hist

def _bool_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(float))) if mask.size else float('nan')


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')


def _first_sustained_time(t: np.ndarray, mask: np.ndarray, dt: float, hold_s: float, start_index: int = 0) -> float:
    if t.size == 0 or mask.size == 0:
        return float('nan')
    hold_n = max(1, int(round(hold_s / max(dt, 1e-9))))
    for i in range(max(0, start_index), len(mask) - hold_n + 1):
        if np.all(mask[i:i + hold_n]):
            return float(t[i])
    return float('nan')


def _count_zero_crossings_with_deadband(x: np.ndarray, deadband: float) -> int:
    if x.size == 0:
        return 0
    sign = np.zeros_like(x, dtype=int)
    sign[x > deadband] = 1
    sign[x < -deadband] = -1
    last = 0
    count = 0
    for s in sign:
        if s == 0:
            continue
        if last != 0 and s != last:
            count += 1
        last = s
    return int(count)


def _sample_durations(t: np.ndarray, nominal_dt_s: float) -> np.ndarray:
    """Approximate per-sample time weights for irregular latency-aware runs."""
    if t.size == 0:
        return np.asarray([], dtype=float)
    if t.size == 1:
        return np.asarray([max(0.0, float(nominal_dt_s))], dtype=float)
    diffs = np.diff(t)
    positive = diffs[diffs > 1e-12]
    last = float(np.nanmedian(positive)) if positive.size else max(0.0, float(nominal_dt_s))
    return np.concatenate([np.maximum(diffs, 0.0), np.asarray([last])])





def history_to_csv_rows(hist: History) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    n = len(hist.t)
    for i in range(n):
        rows.append({
            'time_s': hist.t[i],
            'T_avg_C': hist.T_avg[i],
            'T_stack_C': hist.T_stack[i],
            'v_stack_mps': hist.v_stack[i],
            'T_set_C': hist.T_set[i],
            'Tavg_error_C': hist.error_C[i],
            'omega_equiv_from_Tavg': hist.omega_est[i],
            'omega_out': hist.omega_out[i],
            'T_solid_out_C': hist.T_solid_out[i],
            'mdot_d_out_kgps': hist.mdot_d_out[i],
            'mdot_w_out_kgps': hist.mdot_w_out[i],
            'mdot_wet_out_kgps': hist.mdot_wet_out[i],
            'mdot_furnace_dry_kgps': hist.mdot_furnace_dry_kgps[i],
            'mdot_furnace_water_kgps': hist.mdot_furnace_water_kgps[i],
            'mdot_furnace_wet_kgps': hist.mdot_furnace_wet_kgps[i],
            'rd_used': hist.rd_used[i],
            'omega_target_from_T': hist.omega_target_from_T[i],
            'T_static_from_current_furnace_feed_C': hist.T_static_from_current_furnace_feed[i],
            'water_evap_kgps': hist.water_evap_rate[i],
            'inventory_dry_kg': hist.inventory_dry[i],
            'inventory_water_kg': hist.inventory_water[i],
            'inventory_total_kg': hist.inventory_total[i],
            'outlet_cell_omega0': hist.outlet_cell_omega0[i],
            'outlet_cell_tref_min': hist.outlet_cell_tref_min[i],
            'outlet_cell_slope_min_per_C': hist.outlet_cell_slope_min_per_C[i],
            'Tg_ref_C': hist.Tg_ref[i],
            'vg_ref_mps': hist.vg_ref[i],
            'Tg_cmd_C': hist.Tg_cmd[i],
            'vg_cmd_mps': hist.vg_cmd[i],
            'Tg_ref_minus_cmd_C': hist.Tg_ref_minus_cmd[i],
            'vg_ref_minus_cmd_mps': hist.vg_ref_minus_cmd[i],
            'operator_source': hist.control_source[i],
            'operator_note': hist.control_note[i],
            'operator_cost': hist.operator_cost[i],
            'operator_feasible': hist.operator_feasible[i],
            'nmpc_plan_age_s': hist.nmpc_plan_age_s[i],
            'nmpc_last_solve_ms': hist.nmpc_last_solve_ms[i],
            'nmpc_async_job_running': hist.nmpc_async_job_running[i],
            'nmpc_stale_plan': hist.nmpc_stale_plan[i],
            'operator_compute_wall_s': hist.operator_compute_wall_s[i],
            'simulated_compute_latency_s': hist.simulated_compute_latency_s[i],
            'command_apply_delay_s': hist.command_apply_delay_s[i],
            'async_submitted_jobs': hist.async_submitted_jobs[i],
            'async_accepted_plans': hist.async_accepted_plans[i],
            'async_discarded_plans': hist.async_discarded_plans[i],
            'async_last_result_state_age_s': hist.async_last_result_state_age_s[i],
            'safety_reachable': hist.safety_reachable[i],
            'nmpc_pred_min_Tavg_C': hist.nmpc_pred_min_Tavg[i],
            'nmpc_pred_max_Tavg_C': hist.nmpc_pred_max_Tavg[i],
            'omega_max_for_safety': hist.omega_max_for_safety[i],
            'safety_margin_C': hist.safety_margin_C[i],
            'recovery_guard_requested': hist.recovery_guard_requested[i],
            'recovery_guard_active': hist.recovery_guard_active[i],
            'recovery_guard_reason': hist.recovery_guard_reason[i],
            'nmpc_pred_terminal_Tavg_C': hist.Tavg_pred[i],
            'Tg_in_C': hist.Tg_in[i],
            'Tg_out_C': hist.Tg_out[i],
            'mdot_gas_preheater_kgps': hist.mdot_gas_preheater[i],
            'U_eff_W_m2K': hist.U_eff[i],
            'Q_gas_to_solid_kW': hist.Q_gas_to_solid[i],
            'Q_sensible_kW': hist.Q_sensible[i],
            'Q_latent_kW': hist.Q_latent[i],
            'heat_balance_residual_kW': hist.heat_balance_residual[i],
            'Q_aux_heat_kW': hist.Q_aux_heat[i],
            'T_stack_available_C': hist.T_stack_available[i],
            'v_stack_available_mps': hist.v_stack_available[i],
            'mdot_stack_available_kgps': hist.mdot_stack_available[i],
            'mdot_preheater_cmd_kgps': hist.mdot_preheater_cmd[i],
            'aux_resource_required': hist.aux_resource_required[i],
            'aux_heat_enable': hist.aux_heat_enable[i],
            'mdot_aux_flow_kgps': hist.mdot_aux_flow[i],
            'fan_circulation_power_kW': hist.fan_circulation_power[i],
            'disturbance_Tavg_C': hist.disturbance_Tavg[i],
            'disturbance_Tstack_C': hist.disturbance_Tstack[i],
            'disturbance_vstack_mps': hist.disturbance_vstack[i],
            'disturbance_est_Tavg_C': hist.disturbance_est_Tavg[i],
            'disturbance_est_Tstack_C': hist.disturbance_est_Tstack[i],
            'disturbance_est_vstack_mps': hist.disturbance_est_vstack[i],
            'disturbance_residual_Tavg_C': hist.disturbance_Tavg[i] - hist.disturbance_est_Tavg[i],
            'disturbance_residual_Tstack_C': hist.disturbance_Tstack[i] - hist.disturbance_est_Tstack[i],
            'disturbance_residual_vstack_mps': hist.disturbance_vstack[i] - hist.disturbance_est_vstack[i],
            'feed_x1': hist.feed_x1[i], 'feed_x2': hist.feed_x2[i], 'feed_x3': hist.feed_x3[i],
            'feed_x4': hist.feed_x4[i], 'feed_x5': hist.feed_x5[i], 'feed_x6': hist.feed_x6[i],
            'feed_wet_mass_flow_kgps': hist.feed_wet_mass_flow_kgps[i],
            'mdot_preheater_wet_in_kgps': hist.mdot_preheater_wet_in_kgps[i],
            'mdot_furnace_dry_kgps': hist.mdot_furnace_dry_kgps[i],
            'mdot_furnace_water_kgps': hist.mdot_furnace_water_kgps[i],
            'mdot_furnace_wet_kgps': hist.mdot_furnace_wet_kgps[i],
            'rd_used': hist.rd_used[i],
            'omega_target_from_T': hist.omega_target_from_T[i],
            'T_target_C': hist.T_target_C[i],
            'T_compliance_min_C': hist.T_compliance_min_C[i],
            'T_static_from_current_furnace_feed_C': hist.T_static_from_current_furnace_feed[i],
        })
    return rows


def _rows_at_control_period(hist: History, cfg: SimConfig) -> list[int]:
    out: list[int] = []
    period = max(float(cfg.dt_opt_s), 1e-9)
    for i, t in enumerate(hist.t):
        if abs((t / period) - round(t / period)) <= max(1e-9, 0.51 * cfg.dt_meas_s / period):
            out.append(i)
    return out


def control_event_rows(hist: History, cfg: SimConfig) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i in _rows_at_control_period(hist, cfg):
        rows.append({
            'time_s': hist.t[i],
            'operator_source': hist.control_source[i],
            'operator_feasible': hist.operator_feasible[i],
            'operator_cost': hist.operator_cost[i],
            'operator_note': hist.control_note[i],
            'Tg_ref_C': hist.Tg_ref[i],
            'vg_ref_mps': hist.vg_ref[i],
            'Tg_cmd_C': hist.Tg_cmd[i],
            'vg_cmd_mps': hist.vg_cmd[i],
            'Tg_ref_minus_cmd_C': hist.Tg_ref_minus_cmd[i],
            'vg_ref_minus_cmd_mps': hist.vg_ref_minus_cmd[i],
            'nmpc_plan_age_s': hist.nmpc_plan_age_s[i],
            'nmpc_last_solve_ms': hist.nmpc_last_solve_ms[i],
            'nmpc_async_job_running': hist.nmpc_async_job_running[i],
            'nmpc_stale_plan': hist.nmpc_stale_plan[i],
            'operator_compute_wall_s': hist.operator_compute_wall_s[i],
            'simulated_compute_latency_s': hist.simulated_compute_latency_s[i],
            'command_apply_delay_s': hist.command_apply_delay_s[i],
            'async_submitted_jobs': hist.async_submitted_jobs[i],
            'async_accepted_plans': hist.async_accepted_plans[i],
            'async_discarded_plans': hist.async_discarded_plans[i],
            'async_last_result_state_age_s': hist.async_last_result_state_age_s[i],
            'safety_reachable': hist.safety_reachable[i],
            'nmpc_pred_min_Tavg_C': hist.nmpc_pred_min_Tavg[i],
            'nmpc_pred_max_Tavg_C': hist.nmpc_pred_max_Tavg[i],
            'omega_max_for_safety': hist.omega_max_for_safety[i],
            'safety_margin_C': hist.safety_margin_C[i],
            'recovery_guard_requested': hist.recovery_guard_requested[i],
            'recovery_guard_active': hist.recovery_guard_active[i],
            'recovery_guard_reason': hist.recovery_guard_reason[i],
            'aux_resource_required': hist.aux_resource_required[i],
            'Q_aux_heat_kW': hist.Q_aux_heat[i],
            'mdot_aux_flow_kgps': hist.mdot_aux_flow[i],
            'fan_circulation_power_kW': hist.fan_circulation_power[i],
        })
    return rows


def preheater_diagnostic_rows(hist: History, cfg: SimConfig) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i in _rows_at_control_period(hist, cfg):
        rows.append({
            'time_s': hist.t[i],
            'omega_out': hist.omega_out[i],
            'T_solid_out_C': hist.T_solid_out[i],
            'mdot_d_out_kgps': hist.mdot_d_out[i],
            'mdot_w_out_kgps': hist.mdot_w_out[i],
            'mdot_wet_out_kgps': hist.mdot_wet_out[i],
            'mdot_furnace_dry_kgps': hist.mdot_furnace_dry_kgps[i],
            'mdot_furnace_water_kgps': hist.mdot_furnace_water_kgps[i],
            'mdot_furnace_wet_kgps': hist.mdot_furnace_wet_kgps[i],
            'rd_used': hist.rd_used[i],
            'omega_target_from_T': hist.omega_target_from_T[i],
            'T_static_from_current_furnace_feed_C': hist.T_static_from_current_furnace_feed[i],
            'water_evap_kgps': hist.water_evap_rate[i],
            'inventory_dry_kg': hist.inventory_dry[i],
            'inventory_water_kg': hist.inventory_water[i],
            'inventory_total_kg': hist.inventory_total[i],
            'Tg_in_C': hist.Tg_in[i],
            'Tg_out_C': hist.Tg_out[i],
            'mdot_gas_preheater_kgps': hist.mdot_gas_preheater[i],
            'U_eff_W_m2K': hist.U_eff[i],
            'Q_gas_to_solid_kW': hist.Q_gas_to_solid[i],
            'Q_sensible_kW': hist.Q_sensible[i],
            'Q_latent_kW': hist.Q_latent[i],
            'heat_balance_residual_kW': hist.heat_balance_residual[i],
        })
    return rows


def _segment_rows(hist: History, cfg: SimConfig) -> list[dict[str, float | str]]:
    t = np.asarray(hist.t, dtype=float)
    T_avg = np.asarray(hist.T_avg, dtype=float)
    T_set = np.asarray(hist.T_set, dtype=float)
    omega_out = np.asarray(hist.omega_out, dtype=float)
    Tsolid = np.asarray(hist.T_solid_out, dtype=float)
    wet_out = np.asarray(hist.mdot_wet_out, dtype=float)
    evap = np.asarray(hist.water_evap_rate, dtype=float)
    heater = np.asarray(hist.Q_aux_heat, dtype=float)
    fan_power = np.asarray(hist.fan_circulation_power, dtype=float)
    aux_flow = np.asarray(hist.mdot_aux_flow, dtype=float)
    aux_resource_required = np.asarray(hist.aux_resource_required, dtype=float)
    aux_heat_required = np.asarray(hist.aux_heat_enable, dtype=float)
    aux_circulation_required = (np.asarray(hist.mdot_aux_flow, dtype=float) > 1e-9).astype(float)
    # The current resource model does not hard-clamp Tg/vg by natural stack flow;
    # shortages are served by auxiliary heat/circulation and logged separately.
    command_hard_clipped_by_resource = np.zeros_like(aux_resource_required, dtype=float)
    fallback = np.asarray([0.0 if ('plan_hold' in s or 'block_slsqp' in s) else 1.0 for s in hist.control_source], dtype=float)
    async_hold = np.asarray([1.0 if s == 'async_nmpc_plan_hold' else 0.0 for s in hist.control_source], dtype=float)
    stale = np.asarray(hist.nmpc_stale_plan, dtype=float)
    compute_wall = np.asarray(hist.operator_compute_wall_s, dtype=float)
    sim_latency = np.asarray(hist.simulated_compute_latency_s, dtype=float)
    command_delay = np.asarray(hist.command_apply_delay_s, dtype=float)
    async_discards = np.asarray(hist.async_discarded_plans, dtype=float)
    async_accepts = np.asarray(hist.async_accepted_plans, dtype=float)
    async_result_age = np.asarray(hist.async_last_result_state_age_s, dtype=float)
    safety_reachable = np.asarray(hist.safety_reachable, dtype=float)
    safety_margin = np.asarray(hist.safety_margin_C, dtype=float)
    recovery_guard_requested = np.asarray(hist.recovery_guard_requested, dtype=float)
    recovery_guard = np.asarray(hist.recovery_guard_active, dtype=float)
    omega_max_for_safety = np.asarray(hist.omega_max_for_safety, dtype=float)
    Tg_cmd = np.asarray(hist.Tg_cmd, dtype=float)
    vg_cmd = np.asarray(hist.vg_cmd, dtype=float)
    Tg_ref = np.asarray(hist.Tg_ref, dtype=float)
    vg_ref = np.asarray(hist.vg_ref, dtype=float)
    dt = cfg.dt_meas_s
    weights = _sample_durations(t, dt)
    tail_start = max(0.0, cfg.total_time_s - cfg.tail_window_s)

    segments: list[tuple[str, np.ndarray]] = [('full', np.ones_like(t, dtype=bool))]
    if cfg.event_start_s is not None:
        segments.append(('pre_event', t < cfg.event_start_s))
        if cfg.event_end_s is not None and cfg.event_end_s > cfg.event_start_s:
            segments.append(('event_window', (t >= cfg.event_start_s) & (t < cfg.event_end_s)))
            segments.append(('post_event', t >= cfg.event_end_s))
        else:
            segments.append(('post_event', t >= cfg.event_start_s))
    segments.append(('tail', t >= tail_start))

    rows: list[dict[str, float | str]] = []
    for segment_name, mask in segments:
        if not np.any(mask):
            continue
        Ts = T_avg[mask]
        Es = T_set[mask] - Ts
        rows.append({
            'case_name': cfg.case_name,
            'segment': segment_name,
            't_start_s': float(t[mask][0]),
            't_end_s': float(t[mask][-1]),
            'duration_s': float(np.nansum(weights[mask])),
            'Tavg_mean_C': float(np.mean(Ts)),
            'Tavg_min_C': float(np.min(Ts)),
            'Tavg_max_C': float(np.max(Ts)),
            'Tavg_MAE_C': float(np.mean(np.abs(Es))),
            'Tavg_RMSE_C': _rms(Es),
            'Tavg_pp_C': float(np.max(Ts) - np.min(Ts)),
            'ratio_in_ref_band': _bool_ratio((Ts >= REFERENCE_TAVG_BAND[0]) & (Ts <= REFERENCE_TAVG_BAND[1])),
            'ratio_in_safe_band': _bool_ratio((Ts >= SAFE_TAVG_BAND[0]) & (Ts <= SAFE_TAVG_BAND[1])),
            'ratio_in_supervision_band': _bool_ratio((Ts >= SUPERVISION_TAVG_BAND[0]) & (Ts <= SUPERVISION_TAVG_BAND[1])),
            'omega_out_mean': float(np.nanmean(omega_out[mask])),
            'omega_out_min': float(np.nanmin(omega_out[mask])),
            'omega_out_max': float(np.nanmax(omega_out[mask])),
            'T_solid_out_mean_C': float(np.nanmean(Tsolid[mask])),
            'mdot_wet_out_mean_kgps': float(np.nanmean(wet_out[mask])),
            'mdot_furnace_dry_mean_kgps': float(np.nanmean(np.asarray(hist.mdot_furnace_dry_kgps, dtype=float)[mask])),
            'rd_used_mean': float(np.nanmean(np.asarray(hist.rd_used, dtype=float)[mask])),
            'omega_target_from_T_mean': float(np.nanmean(np.asarray(hist.omega_target_from_T, dtype=float)[mask])),
            'ratio_T_ge_compliance_min': _bool_ratio(Ts >= cfg.T_compliance_min_C),
            'water_evap_mean_kgps': float(np.nanmean(evap[mask])),
            'Q_aux_heat_energy_kJ': float(np.nansum(heater[mask] * weights[mask])),
            'fan_energy_kJ': float(np.nansum(fan_power[mask] * weights[mask])),
            'aux_flow_mean_kgps': float(np.nanmean(aux_flow[mask])),
            'aux_flow_max_kgps': float(np.nanmax(aux_flow[mask])),
            'operator_fallback_ratio': float(np.nanmean(fallback[mask])),
            'operator_async_hold_ratio': float(np.nanmean(async_hold[mask])),
            'operator_stale_plan_ratio': float(np.nanmean(stale[mask])),
            'operator_compute_wall_mean_s': float(np.nanmean(compute_wall[mask])),
            'operator_compute_wall_max_s': float(np.nanmax(compute_wall[mask])),
            'simulated_compute_latency_mean_s': float(np.nanmean(sim_latency[mask])),
            'simulated_compute_latency_max_s': float(np.nanmax(sim_latency[mask])),
            'command_apply_delay_mean_s': float(np.nanmean(command_delay[mask])),
            'async_discarded_plan_count_delta': float(np.nanmax(async_discards[mask]) - np.nanmin(async_discards[mask])),
            'async_accepted_plan_count_delta': float(np.nanmax(async_accepts[mask]) - np.nanmin(async_accepts[mask])),
            'async_last_result_state_age_max_s': float(np.nanmax(async_result_age[mask])) if np.any(np.isfinite(async_result_age[mask])) else float('nan'),
            'safety_reachable_ratio': float(np.nanmean(safety_reachable[mask])),
            'safety_margin_min_C': float(np.nanmin(safety_margin[mask])),
            'recovery_guard_requested_ratio': float(np.nanmean(recovery_guard_requested[mask])),
                'recovery_guard_ratio': float(np.nanmean(recovery_guard[mask])),
            'safety_margin_mean_C': float(np.nanmean(safety_margin[mask])),
            'omega_max_for_safety_mean': float(np.nanmean(omega_max_for_safety[mask])),
            'aux_resource_required_ratio': float(np.nanmean(aux_resource_required[mask])),
            'aux_heat_required_ratio': float(np.nanmean(aux_heat_required[mask])),
            'aux_circulation_required_ratio': float(np.nanmean(aux_circulation_required[mask])),
            'command_hard_clipped_by_resource_ratio': float(np.nanmean(command_hard_clipped_by_resource[mask])),
            'Tg_cmd_tv': float(np.nansum(np.abs(np.diff(Tg_cmd[mask])))),
            'vg_cmd_tv': float(np.nansum(np.abs(np.diff(vg_cmd[mask])))),
            'Tg_ref_tv': float(np.nansum(np.abs(np.diff(Tg_ref[mask])))),
            'vg_ref_tv': float(np.nansum(np.abs(np.diff(vg_ref[mask])))),
        })

    post_start = cfg.event_end_s if cfg.event_end_s is not None else cfg.event_start_s
    if post_start is not None:
        post_mask = t >= post_start
    else:
        post_mask = t >= tail_start
    if np.any(post_mask):
        Ts = T_avg[post_mask]
        Es = T_set[post_mask] - Ts
        start_idx = int(np.where(post_mask)[0][0])
        safe_mask = (T_avg >= SAFE_TAVG_BAND[0]) & (T_avg <= SAFE_TAVG_BAND[1])
        ref_mask = (T_avg >= REFERENCE_TAVG_BAND[0]) & (T_avg <= REFERENCE_TAVG_BAND[1])
        rows.append({
            'case_name': cfg.case_name,
            'segment': 'recovery',
            't_start_s': float(t[post_mask][0]),
            't_end_s': float(t[post_mask][-1]),
            'duration_s': float(np.nansum(weights[post_mask])),
            'Tavg_mean_C': float(np.mean(Ts)),
            'Tavg_min_C': float(np.min(Ts)),
            'Tavg_max_C': float(np.max(Ts)),
            'Tavg_MAE_C': float(np.mean(np.abs(Es))),
            'Tavg_RMSE_C': _rms(Es),
            'Tavg_pp_C': float(np.max(Ts) - np.min(Ts)),
            'ratio_in_ref_band': _bool_ratio((Ts >= REFERENCE_TAVG_BAND[0]) & (Ts <= REFERENCE_TAVG_BAND[1])),
            'ratio_in_safe_band': _bool_ratio((Ts >= SAFE_TAVG_BAND[0]) & (Ts <= SAFE_TAVG_BAND[1])),
            'ratio_in_supervision_band': _bool_ratio((Ts >= SUPERVISION_TAVG_BAND[0]) & (Ts <= SUPERVISION_TAVG_BAND[1])),
            'recovery_to_safe_s': _first_sustained_time(t, safe_mask, dt, cfg.settle_hold_safe_s, start_idx),
            'recovery_to_ref_s': _first_sustained_time(t, ref_mask, dt, cfg.settle_hold_ref_s, start_idx),
            'overshoot_crossings': _count_zero_crossings_with_deadband(Ts - T_set[post_mask], cfg.overshoot_deadband_C),
        })
    return rows



def _add_event_windows(axes, event_windows: Sequence[EventWindow] | None):
    for ev in event_windows or []:
        for ax in axes:
            ax.axvspan(ev.start_s, ev.end_s, alpha=0.12)


def _safe_series(values):
    return np.asarray(values, dtype=float)


def _plot_cell_snapshots_panel(ax_omega, ax_temp, hist: History):
    """Draw compact cell-state snapshots inside the overview figure."""
    if not hist.cell_snapshots:
        ax_omega.text(0.5, 0.5, 'no cell snapshots', ha='center', va='center', transform=ax_omega.transAxes)
        ax_omega.set_axis_off()
        ax_temp.set_axis_off()
        return
    rows = hist.cell_snapshots
    times = sorted({float(r['time_s']) for r in rows})
    max_cell_index = 0
    for tt in times:
        rs = sorted((r for r in rows if float(r['time_s']) == tt), key=lambda r: int(r['cell_index']))
        x = [int(r['cell_index']) for r in rs]
        if x:
            max_cell_index = max(max_cell_index, max(x))
        ax_omega.plot(x, [float(r['omega']) for r in rs], marker='o', markersize=3, label=f'{tt:g}s')
        ax_temp.plot(x, [float(r['T_solid_C']) for r in rs], marker='o', markersize=3, label=f'{tt:g}s')
    ax_omega.set_ylabel('cell omega')
    ax_temp.set_ylabel('cell T_solid C')
    ax_temp.set_xlabel('cell index')
    ax_omega.set_xlim(-0.5, max_cell_index + 0.5)
    ax_temp.set_xlim(-0.5, max_cell_index + 0.5)
    tick_step = 1 if max_cell_index <= 20 else max(1, int(round((max_cell_index + 1) / 10)))
    ticks = list(range(0, max_cell_index + 1, tick_step))
    ax_omega.set_xticks(ticks)
    ax_temp.set_xticks(ticks)
    ax_omega.legend(fontsize=7, ncol=4)


def plot_history(hist: History, out_png: str | Path, title: str, *, event_windows: Sequence[EventWindow] | None = None):
    """Write the single per-case overview image.

    This overview replaces the older set of separate thermal, preheater,
    control, resource, disturbance, and cell-snapshot PNG files.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 22))
    gs = fig.add_gridspec(7, 1, height_ratios=[1.15, 1.15, 1.05, 1.10, 1.05, 0.82, 0.82])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(7)]
    ax_thermal, ax_preheater, ax_control, ax_resource, ax_disturbance, ax_cell_omega, ax_cell_temp = axes
    _add_event_windows([ax_thermal, ax_preheater, ax_control, ax_resource, ax_disturbance], event_windows)
    t = _safe_series(hist.t)

    ax_thermal.plot(t, hist.T_avg, label='T_avg_C')
    ax_thermal.plot(t, hist.T_stack, label='T_stack_C')
    ax_thermal.plot(t, hist.T_set, linestyle='--', label='T_set_C')
    ax_thermal.axhline(REFERENCE_TAVG_BAND[0], linestyle=':', label='ref low')
    ax_thermal.axhline(REFERENCE_TAVG_BAND[1], linestyle=':', label='ref high')
    ax_thermal.axhline(SAFE_TAVG_BAND[0], linestyle='-.', label='safe low')
    ax_thermal.axhline(SAFE_TAVG_BAND[1], linestyle='-.', label='safe high')
    ax_thermal.set_ylabel('furnace C')
    ax_thermal.legend(fontsize=8, ncol=4)

    ax_preheater.plot(t, hist.omega_out, label='omega_out')
    ax_preheater.plot(t, hist.outlet_cell_omega0, linestyle=':', label='outlet cell omega0')
    ax_preheater.set_ylabel('moisture')
    if hist.omega_out:
        ymax = max(0.95, float(np.nanmax(_safe_series(hist.omega_out))) + 0.05)
    else:
        ymax = 0.95
    ax_preheater.set_ylim(0.0, ymax)
    ax_preheater_b = ax_preheater.twinx()
    ax_preheater_b.plot(t, hist.T_solid_out, linestyle='--', label='T_solid_out_C')
    ax_preheater_b.plot(t, hist.Tg_in, linestyle='-.', label='Tg_in_C')
    ax_preheater_b.plot(t, hist.Tg_out, linestyle=':', label='Tg_out_C')
    ax_preheater_b.set_ylabel('preheater C')
    lines, labels = ax_preheater.get_legend_handles_labels()
    lines2, labels2 = ax_preheater_b.get_legend_handles_labels()
    ax_preheater.legend(lines + lines2, labels + labels2, fontsize=8, ncol=4)

    ax_control.plot(t, hist.Tg_ref, label='Tg_ref_C')
    ax_control.plot(t, hist.Tg_cmd, linestyle='--', label='Tg_cmd_C')
    ax_control.set_ylabel('Tg C')
    ax_control_b = ax_control.twinx()
    ax_control_b.plot(t, hist.vg_ref, label='vg_ref_mps')
    ax_control_b.plot(t, hist.vg_cmd, linestyle='--', label='vg_cmd_mps')
    ax_control_b.plot(t, hist.nmpc_plan_age_s, linestyle=':', label='plan_age_s')
    ax_control_b.set_ylabel('vg / plan age')
    lines, labels = ax_control.get_legend_handles_labels()
    lines2, labels2 = ax_control_b.get_legend_handles_labels()
    ax_control.legend(lines + lines2, labels + labels2, fontsize=8, ncol=5)

    ax_resource.plot(t, hist.T_stack_available, label='T_stack_available_C')
    ax_resource.plot(t, hist.mdot_stack_available, label='mdot_stack_available_kgps')
    ax_resource.plot(t, hist.mdot_preheater_cmd, linestyle='--', label='mdot_preheater_cmd_kgps')
    ax_resource.plot(t, hist.mdot_aux_flow, linestyle=':', label='mdot_aux_flow_kgps')
    ax_resource.set_ylabel('resource')
    ax_resource_b = ax_resource.twinx()
    ax_resource_b.plot(t, hist.Q_aux_heat, linestyle='--', label='Q_aux_heat_kW')
    ax_resource_b.plot(t, hist.fan_circulation_power, linestyle=':', label='fan_power_kW')
    ax_resource_b.set_ylabel('kW')
    lines, labels = ax_resource.get_legend_handles_labels()
    lines2, labels2 = ax_resource_b.get_legend_handles_labels()
    ax_resource.legend(lines + lines2, labels + labels2, fontsize=8, ncol=4)

    ax_disturbance.plot(t, hist.error_C, label='Tavg_error_C')
    ax_disturbance.plot(t, hist.disturbance_Tavg, label='disturbance_Tavg_C')
    ax_disturbance.plot(t, hist.disturbance_est_Tavg, linestyle='--', label='dist_est_Tavg_C')
    ax_disturbance.plot(t, hist.disturbance_Tstack, label='disturbance_Tstack_C')
    ax_disturbance.plot(t, hist.disturbance_est_Tstack, linestyle='--', label='dist_est_Tstack_C')
    ax_disturbance.set_ylabel('disturbance / error')
    ax_disturbance.legend(fontsize=8, ncol=3)

    _plot_cell_snapshots_panel(ax_cell_omega, ax_cell_temp, hist)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    ax_cell_temp.set_xlabel('cell index')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.01, 1, 0.985])
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write a sequence of dictionary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _case_output_dir(cfg: SimConfig) -> Path:
    return Path(cfg.out_dir) / cfg.case_name


def save_case_artifacts(hist: History, cfg: SimConfig, title: str, *, event_windows: Sequence[EventWindow] | None = None) -> dict[str, Path]:
    case_dir = _case_output_dir(cfg)
    case_dir.mkdir(parents=True, exist_ok=True)

    overview_path = case_dir / 'overview.png'
    timeseries_path = case_dir / 'timeseries.csv'
    metrics_path = case_dir / 'metrics.csv'
    control_path = case_dir / 'control_events.csv'
    preheater_path = case_dir / 'preheater_diagnostics.csv'
    cells_path = case_dir / 'cell_snapshot.csv'

    plot_history(hist, overview_path, title, event_windows=event_windows)
    _write_csv(timeseries_path, history_to_csv_rows(hist))
    _write_csv(metrics_path, _segment_rows(hist, cfg))
    _write_csv(control_path, control_event_rows(hist, cfg))
    _write_csv(preheater_path, preheater_diagnostic_rows(hist, cfg))
    _write_csv(cells_path, hist.cell_snapshots)
    return {
        'artifact_dir': case_dir,
        'plot': overview_path,
        'overview_plot': overview_path,
        'thermal_plot': overview_path,
        'preheater_plot': overview_path,
        'control_plot': overview_path,
        'resource_plot': overview_path,
        'disturbance_plot': overview_path,
        'cell_plot': overview_path,
        'timeseries': timeseries_path,
        'metrics': metrics_path,
        'control_events': control_path,
        'preheater_diagnostics': preheater_path,
        'cell_snapshot': cells_path,
    }


def print_metrics_table(metrics_csv: Path):
    with metrics_csv.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    cols = [
        'segment','duration_s','Tavg_MAE_C','Tavg_RMSE_C','ratio_in_ref_band','ratio_in_safe_band',
        'omega_out_mean','T_solid_out_mean_C','safety_reachable_ratio','safety_margin_min_C','operator_fallback_ratio','recovery_guard_requested_ratio','recovery_guard_ratio','aux_resource_required_ratio',
        'recovery_to_safe_s','recovery_to_ref_s','overshoot_crossings'
    ]
    print(','.join(cols))
    for row in rows:
        print(','.join(row.get(c, '') for c in cols))
