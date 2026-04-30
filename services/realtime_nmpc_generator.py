from __future__ import annotations

"""Realtime FlameGuard-main plant + fast NMPC rollout generator.

The uploaded FlameGuard-main runtime test loop is batch/offline: it advances a
complete scenario and writes CSVs.  FlameGuardWeb needs the opposite: an
incremental wall-clock loop that returns immediately on every browser poll.

This module reuses FlameGuard-main's stable pieces (domain contracts, Python
plant, state estimator, predictor models, executor), and rebuilds only the web
runtime loop plus a fast receding-horizon NMPC rollout search.  The full
SLSQP-based ``controller.operator.nmpc_operator`` remains vendored in the
project, but it is intentionally not executed inside HTTP requests because the
pure-Python objective can monopolize the GIL and freeze dashboard refresh.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from controller.factory import make_executor, make_predictor_resource_model, make_state_estimator
from domain.types import ActuatorCommand, ControlSetpoint, MPCDecision, PlantStepInput, StackResourceMeasurement
from plant.factory import make_plant_backend
from plant.python_model.furnace import furnace_feed_from_preheater_output, furnace_static_outputs_from_inputs
from plant.python_model.material_model import feedstock_from_composition
from runtime.simulator import SimConfig

from .composition_adapter import DEFAULT_WET_MASS_FLOW_KGPS, NAMES, validate_composition


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


class RealisticDisturbanceModel:
    """Correlated random disturbances for live-looking furnace observations."""

    def __init__(self, seed: int = 20260430) -> None:
        self.rng = random.Random(seed)
        self.d_avg = 0.0
        self.d_stack = 0.0
        self.d_v = 0.0
        self.pulse_avg = 0.0
        self.pulse_stack = 0.0
        self.pulse_v = 0.0
        self.next_pulse_s = self.rng.uniform(45.0, 90.0)

    def step(self, time_s: float, dt_s: float) -> tuple[float, float, float]:
        dt = max(1e-6, float(dt_s))
        self.d_avg += (-self.d_avg / 55.0) * dt + self.rng.gauss(0.0, 0.10) * math.sqrt(dt)
        self.d_stack += (-self.d_stack / 70.0) * dt + self.rng.gauss(0.0, 0.14) * math.sqrt(dt)
        self.d_v += (-self.d_v / 35.0) * dt + self.rng.gauss(0.0, 0.014) * math.sqrt(dt)
        self.d_avg = _clamp(self.d_avg, -6.0, 6.0)
        self.d_stack = _clamp(self.d_stack, -8.5, 8.5)
        self.d_v = _clamp(self.d_v, -0.55, 0.55)
        if time_s >= self.next_pulse_s:
            sign = -1.0 if self.rng.random() < 0.72 else 1.0
            self.pulse_avg += sign * self.rng.uniform(3.0, 11.0)
            self.pulse_stack += sign * self.rng.uniform(1.5, 7.0)
            self.pulse_v += self.rng.uniform(-0.35, 0.35)
            self.next_pulse_s = time_s + self.rng.uniform(55.0, 135.0)
        self.pulse_avg *= math.exp(-dt / 18.0)
        self.pulse_stack *= math.exp(-dt / 18.0)
        self.pulse_v *= math.exp(-dt / 12.0)
        return (
            _clamp(self.d_avg + self.pulse_avg + self.rng.gauss(0.0, 0.25), -16.0, 14.0),
            _clamp(self.d_stack + self.pulse_stack + self.rng.gauss(0.0, 0.35), -18.0, 16.0),
            _clamp(self.d_v + self.pulse_v + self.rng.gauss(0.0, 0.020), -1.2, 1.2),
        )


@dataclass
class FastSolveProfile:
    solve_ms: float = 0.0
    candidates: int = 0
    horizon_steps: int = 0
    best_cost: float = float("nan")


class RealtimeNMPCGenerator:
    def __init__(self) -> None:
        self.cfg = SimConfig(
            case_name="FlameGuardWeb_realtime_fast_nmpc",
            dt_meas_s=0.1,
            dt_opt_s=0.5,
            total_time_s=3600.0,
            mpc_horizon_s=60.0,
            mpc_dt_s=10.0,
            preheater_n_cells=8,
            preheater_warmup_s=1.2 * 985.0,
            preheater_warmup_dt_s=40.0,
            wet_mass_flow_override_kgps=DEFAULT_WET_MASS_FLOW_KGPS,
            resource_v_stack_cap_mps=18.0,
        )
        self.running = True
        self.source = "realtime_fast_nmpc"
        self.confidence = 1.0
        self.composition = [0.20, 0.20, 0.10, 0.20, 0.20, 0.10]
        self._feed_variation = [0.0] * 6
        self._wet_flow = DEFAULT_WET_MASS_FLOW_KGPS
        self._last_wall_time = time.monotonic()
        self._last_control_time_s = -1.0e9
        self._t = 0.0
        self._disturbance = (0.0, 0.0, 0.0)
        self._disturbance_model = RealisticDisturbanceModel()
        self._recovery_latched = False
        self._recovery_enter_time_s: float | None = None
        self._last_recovery_reason = ""
        self._solve_profile = FastSolveProfile()
        self.current_decision: MPCDecision | None = None
        self._initialize_core()

    @property
    def time_s(self) -> float:
        return self._t

    def start(self) -> None:
        self.running = True
        self._last_wall_time = time.monotonic()

    def stop(self) -> None:
        self._advance_to_now()
        self.running = False

    def reset(self) -> None:
        old_comp = list(self.composition)
        old_source = self.source
        old_conf = self.confidence
        old_flow = self._wet_flow
        self.composition = old_comp
        self.source = old_source
        self.confidence = old_conf
        self._wet_flow = old_flow
        self._feed_variation = [0.0] * 6
        self._t = 0.0
        self._disturbance_model = RealisticDisturbanceModel()
        self._initialize_core()

    def shutdown(self) -> None:
        # Kept for adapter compatibility. No background solver thread is used.
        return None

    def update_feedstock(self, composition: list[float], *, source: str = "manual", confidence: float = 1.0, wet_mass_flow_kgps: float | None = None) -> dict:
        self._advance_to_now()
        self.composition = validate_composition(composition, normalize=False)
        self.source = str(source or "manual")
        self.confidence = _clamp(float(confidence), 0.0, 1.0)
        if wet_mass_flow_kgps is not None:
            self._wet_flow = max(0.01, float(wet_mass_flow_kgps))
            self.cfg.wet_mass_flow_override_kgps = self._wet_flow
        return self.snapshot()

    def snapshot(self) -> dict:
        self._advance_to_now()
        return self._build_dashboard_payload()

    def _initialize_core(self) -> None:
        self.current_cmd = self._make_command(0.0, self.cfg.nominal_Tg_C, self.cfg.nominal_vg_mps)
        self.current_decision = None
        self.executor = make_executor(self.cfg)
        if hasattr(self.executor, "initialize_previous"):
            self.executor.initialize_previous(self.cfg.nominal_Tg_C, self.cfg.nominal_vg_mps)
        self.predictor_resource_model = make_predictor_resource_model(self.cfg)
        self.initial_feedstock = self._feedstock_at(0.0)
        self.plant = make_plant_backend(
            self.cfg,
            initial_feedstock=self.initial_feedstock,
            disturbance_schedule=lambda _time_s: self._disturbance,
        )
        self.snapshot_obj = self.plant.reset()
        self.state_estimator = make_state_estimator(
            self.cfg,
            initial_snapshot=self.snapshot_obj,
            initial_feedstock=self.initial_feedstock,
            resource_model=self.predictor_resource_model,
        )
        self.estimate = self.state_estimator.update(
            self.snapshot_obj,
            previous_command=self.current_cmd,
            feedstock=self.initial_feedstock,
            dt_s=self.cfg.dt_meas_s,
        )
        self._last_control_time_s = -1.0e9
        self._last_wall_time = time.monotonic()

    def _feedstock_at(self, time_s: float):
        return feedstock_from_composition(
            float(time_s),
            self._composition_with_variation(),
            normalize=False,
            wet_mass_flow_kgps=self._wet_flow,
            source=self.source,
            confidence=self.confidence,
        )

    def _composition_with_variation(self) -> list[float]:
        vals = [max(0.0, c + dv) for c, dv in zip(self.composition, self._feed_variation)]
        total = sum(vals)
        return list(self.composition) if total <= 1e-12 else [v / total for v in vals]

    def _advance_feed_variation(self, dt_s: float) -> None:
        rng = self._disturbance_model.rng
        for i in range(6):
            self._feed_variation[i] += (-self._feed_variation[i] / 80.0) * dt_s + rng.gauss(0.0, 0.0005) * math.sqrt(max(dt_s, 0.0))
            self._feed_variation[i] = _clamp(self._feed_variation[i], -0.022, 0.022)

    def _advance_to_now(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_wall_time)
        self._last_wall_time = now
        if not self.running:
            return
        remaining = min(elapsed, 2.0)
        while remaining > 1e-9:
            dt = min(float(self.cfg.dt_meas_s), remaining)
            self._step(dt)
            remaining -= dt

    def _step(self, dt_s: float) -> None:
        self._t += float(dt_s)
        self._advance_feed_variation(float(dt_s))
        self._disturbance = self._disturbance_model.step(self._t, float(dt_s))
        feedstock = self._feedstock_at(self._t)
        self.snapshot_obj = self.plant.step(PlantStepInput(time_s=self._t, dt_s=float(dt_s), command=self.current_cmd, feedstock=feedstock))
        self.estimate = self.state_estimator.update(self.snapshot_obj, previous_command=self.current_cmd, feedstock=feedstock, dt_s=float(dt_s))
        if (self._t - self._last_control_time_s) >= float(self.cfg.dt_opt_s) - 1e-12:
            self._run_control_step(feedstock)
            self._last_control_time_s = self._t

    def _run_control_step(self, feedstock) -> None:
        decision = self._fast_nmpc_decision(feedstock)
        self.current_decision = decision
        recovery_active, recovery_reason = self._update_recovery_guard(decision)
        self._last_recovery_reason = recovery_reason
        obs = self.snapshot_obj.furnace_obs
        stack_resource = self.estimate.stack_resource_est or self.snapshot_obj.stack_resource or StackResourceMeasurement(self._t, obs.T_stack_C, obs.v_stack_mps, None)
        mdot_stack_available = stack_resource.mdot_stack_available_kgps
        if mdot_stack_available is None:
            mdot_stack_available = float("inf")
        setpoint = ControlSetpoint(
            time_s=self._t,
            Tg_ref_C=decision.Tg_ref_C,
            vg_ref_mps=decision.vg_ref_mps,
            source=decision.source,
            omega_target=decision.omega_target,
            omega_reachable=decision.omega_reachable,
            mdot_stack_cap_kgps=float(mdot_stack_available),
            T_stack_available_C=float(stack_resource.T_stack_available_C),
            v_stack_available_mps=float(stack_resource.v_stack_available_mps),
            recovery_guard_requested=bool(recovery_active),
            recovery_guard_reason=recovery_reason,
        )
        self.current_cmd = self.executor.translate_setpoint(setpoint)

    def _candidate_controls(self, feedstock) -> list[tuple[float, float]]:
        obs = self.snapshot_obj.furnace_obs
        err = float(self.cfg.T_set_C - obs.T_avg_C)
        flow_factor = ((feedstock.wet_mass_flow_kgps or DEFAULT_WET_MASS_FLOW_KGPS) - DEFAULT_WET_MASS_FLOW_KGPS) / DEFAULT_WET_MASS_FLOW_KGPS
        wet_bias = (float(feedstock.moisture_wb) - 0.74) * 95.0
        heuristic_Tg = _clamp(800.0 + 2.8 * err + wet_bias, 450.0, 1100.0)
        heuristic_vg = _clamp(10.0 + 0.035 * err + 1.2 * flow_factor, 3.0, 12.0)
        pairs = {
            (self.current_cmd.Tg_cmd_C, self.current_cmd.vg_cmd_mps),
            (float(self.cfg.nominal_Tg_C), float(self.cfg.nominal_vg_mps)),
            (heuristic_Tg, heuristic_vg),
            (650.0, 6.0),
            (750.0, 8.0),
            (850.0, 10.0),
            (950.0, 12.0),
            (1100.0, 12.0),
        }
        if self.current_decision is not None:
            for dT in (-80.0, -40.0, 40.0, 80.0):
                pairs.add((_clamp(self.current_decision.Tg_ref_C + dT, 400.0, 1100.0), self.current_decision.vg_ref_mps))
            for dv in (-1.5, 1.5):
                pairs.add((self.current_decision.Tg_ref_C, _clamp(self.current_decision.vg_ref_mps + dv, 3.0, 12.0)))
        return sorted((_clamp(tg, 100.0, 1100.0), _clamp(vg, 3.0, 12.0)) for tg, vg in pairs)

    def _fast_nmpc_decision(self, feedstock) -> MPCDecision:
        t0 = time.perf_counter()
        obs = self.snapshot_obj.furnace_obs
        bundle = self.state_estimator.get_predictor_bundle()
        horizon_dt = max(1.0, float(self.cfg.mpc_dt_s))
        horizon_s = max(horizon_dt, float(self.cfg.mpc_horizon_s))
        n_steps = max(1, int(round(horizon_s / horizon_dt)))
        best: dict[str, float] | None = None
        candidates = self._candidate_controls(feedstock)
        for Tg, vg in candidates:
            pre = bundle.preheater.clone().load_state(self.estimate.preheater_state_est, feedstock=feedstock)
            furnace = bundle.furnace.clone()
            cost = 0.0
            min_T = float("inf")
            max_T = -float("inf")
            final_T = float(obs.T_avg_C)
            final_w = float(self.estimate.preheater_state_est.omega_out)
            pred_mdot_d = float("nan")
            for k in range(n_steps):
                tt = self._t + (k + 1) * horizon_dt
                f = self._feedstock_at(tt)
                out = pre.step_fast(f, Tg, vg, horizon_dt)
                ff = furnace_feed_from_preheater_output(
                    time_s=tt,
                    omega_b=float(out.omega_out if out.omega_out is not None else final_w),
                    mdot_d_kgps=out.dry_out_kgps,
                    mdot_water_kgps=out.water_out_kgps,
                    mdot_wet_kgps=out.wet_out_kgps,
                )
                pred_mdot_d = float(ff.mdot_d_kgps)
                Tavg, _Tstack, _vstack = furnace.step(ff.omega_b, mdot_d_kgps=ff.mdot_d_kgps, dt_s=horizon_dt, disturbance=self._disturbance)
                final_T = float(Tavg)
                final_w = float(out.omega_out if out.omega_out is not None else final_w)
                min_T = min(min_T, final_T)
                max_T = max(max_T, final_T)
                e = final_T - float(self.cfg.T_set_C)
                cost += e * e
                if final_T < float(self.cfg.T_compliance_min_C):
                    cost += 550.0 * (float(self.cfg.T_compliance_min_C) - final_T) ** 2
                cost += 120.0 * (final_w - float(self.cfg.omega_ref)) ** 2
            cost += 0.0015 * (Tg - 780.0) ** 2 + 1.5 * (vg - 10.0) ** 2
            cost += 0.006 * (Tg - self.current_cmd.Tg_cmd_C) ** 2 + 3.0 * (vg - self.current_cmd.vg_cmd_mps) ** 2
            if best is None or cost < best["cost"]:
                best = {"Tg": Tg, "vg": vg, "cost": cost, "final_T": final_T, "final_w": final_w, "min_T": min_T, "max_T": max_T, "mdot_d": pred_mdot_d}
        assert best is not None
        solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_profile = FastSolveProfile(solve_ms=solve_ms, candidates=len(candidates), horizon_steps=n_steps, best_cost=best["cost"])
        mdot_for_target = best["mdot_d"] if math.isfinite(best["mdot_d"]) else 0.052
        omega_target = self._omega_target_for_temperature(float(self.cfg.T_set_C), mdot_for_target)
        omega_max_safe = self._omega_target_for_temperature(float(self.cfg.T_compliance_min_C), mdot_for_target)
        safety_margin = float(best["min_T"] - float(self.cfg.T_compliance_min_C))
        feasible = bool(math.isfinite(best["cost"]) and safety_margin >= -8.0)
        return MPCDecision(
            time_s=self._t,
            Tg_ref_C=float(best["Tg"]),
            vg_ref_mps=float(best["vg"]),
            omega_target=float(omega_target),
            omega_reachable=float(best["final_w"]),
            predicted_Tavg_C=float(best["final_T"]),
            predicted_omega_out=float(best["final_w"]),
            cost=float(best["cost"]),
            feasible=feasible,
            source="realtime_rollout_nmpc",
            safety_reachable=safety_margin >= -1e-6,
            predicted_min_Tavg_C=float(best["min_T"]),
            predicted_max_Tavg_C=float(best["max_T"]),
            omega_max_for_safety=float(omega_max_safe),
            safety_margin_C=float(safety_margin),
            note=f"fast rollout NMPC; candidates={len(candidates)}; horizon={horizon_s:g}s; dt={horizon_dt:g}s; solve_ms={solve_ms:.1f}",
        )

    def _omega_target_for_temperature(self, target_C: float, mdot_d_kgps: float) -> float:
        lo, hi = 0.0, 0.5
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            rd = furnace_feed_from_preheater_output(time_s=0.0, omega_b=mid, mdot_d_kgps=mdot_d_kgps).rd
            T_mid = furnace_static_outputs_from_inputs(mid, rd).T_avg_C
            if T_mid > target_C:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def _update_recovery_guard(self, decision: MPCDecision) -> tuple[bool, str]:
        Tavg = float(self.snapshot_obj.furnace_obs.T_avg_C)
        raw_request = bool(self.cfg.recovery_guard_enabled and (Tavg < float(self.cfg.T_compliance_min_C) or (not decision.safety_reachable and Tavg < self.cfg.T_set_C)))
        if raw_request and not self._recovery_latched:
            self._recovery_latched = True
            self._recovery_enter_time_s = self._t
        hold_elapsed_s = float("inf") if self._recovery_enter_time_s is None else max(0.0, self._t - self._recovery_enter_time_s)
        if self._recovery_latched:
            exit_ready = (not raw_request and hold_elapsed_s >= float(self.cfg.recovery_guard_min_hold_s) and Tavg >= float(self.cfg.recovery_guard_exit_C) and decision.safety_reachable)
            if exit_ready:
                self._recovery_latched = False
                self._recovery_enter_time_s = None
        if self._recovery_latched:
            if Tavg < float(self.cfg.T_compliance_min_C):
                return True, f"T_avg below compliance floor ({Tavg:.2f} < {self.cfg.T_compliance_min_C:.2f})"
            if not decision.safety_reachable:
                return True, "NMPC predicted safety not reachable; recovery guard latched"
            if hold_elapsed_s < float(self.cfg.recovery_guard_min_hold_s):
                return True, f"recovery guard minimum hold active ({hold_elapsed_s:.1f}s elapsed)"
            return True, "recovery guard latched"
        return False, ""

    @staticmethod
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

    def _build_dashboard_payload(self) -> dict[str, Any]:
        feedstock = self._feedstock_at(self._t)
        snap = self.snapshot_obj
        obs = snap.furnace_obs
        pre_state = self.estimate.preheater_state_est
        pre_out = snap.preheater_output
        stack = self.estimate.stack_resource_est or snap.stack_resource
        diag = snap.raw.get("preheater_diagnostics") if snap.raw is not None else None
        decision = self.current_decision
        cmd = self.current_cmd
        mdot_stack_available = _finite(getattr(stack, "mdot_stack_available_kgps", float("nan")), float("nan")) if stack is not None else float("nan")
        dry_out = _finite(getattr(pre_out, "dry_out_kgps", float("nan")), 0.0) if pre_out is not None else 0.0
        water_out = _finite(getattr(pre_out, "water_out_kgps", float("nan")), 0.0) if pre_out is not None else 0.0
        wet_out = _finite(getattr(pre_out, "wet_out_kgps", dry_out + water_out), dry_out + water_out) if pre_out is not None else dry_out + water_out
        ffeed = furnace_feed_from_preheater_output(
            time_s=self._t,
            omega_b=float(pre_state.omega_out),
            mdot_d_kgps=dry_out if dry_out > 0 else None,
            mdot_water_kgps=water_out if water_out >= 0 else None,
            mdot_wet_kgps=wet_out if wet_out > 0 else None,
        )
        try:
            T_static = furnace_static_outputs_from_inputs(ffeed.omega_b, ffeed.rd).T_avg_C
        except Exception:
            T_static = float("nan")
        if decision is None:
            decision = MPCDecision(
                time_s=self._t,
                Tg_ref_C=float(cmd.Tg_cmd_C),
                vg_ref_mps=float(cmd.vg_cmd_mps),
                omega_target=float(self.cfg.omega_ref),
                omega_reachable=float(pre_state.omega_out),
                predicted_Tavg_C=float(obs.T_avg_C),
                predicted_omega_out=float(pre_state.omega_out),
                cost=float("inf"),
                feasible=False,
                source="realtime_rollout_nmpc_initializing",
                safety_reachable=float(obs.T_avg_C) >= float(self.cfg.T_compliance_min_C),
                predicted_min_Tavg_C=float(obs.T_avg_C),
                predicted_max_Tavg_C=float(obs.T_avg_C),
                omega_max_for_safety=float(self.cfg.omega_ref),
                safety_margin_C=float(obs.T_avg_C) - float(self.cfg.T_compliance_min_C),
                note="waiting for first realtime rollout NMPC control tick",
            )
        T_set_C = float(self.cfg.T_set_C)
        T_floor = float(self.cfg.T_compliance_min_C)
        safety_margin_C = float(decision.safety_margin_C)
        recovery_guard_active = bool(getattr(cmd, "recovery_guard_active", False) or self._recovery_latched)
        health_ok = bool(float(obs.T_avg_C) >= T_floor and not recovery_guard_active)
        return {
            "success": True,
            "schema": "FlameGuardWeb.dashboard.v2.realtime_fast_nmpc",
            "time_s": round(float(self._t), 3),
            "running": self.running,
            "feedstock": {
                "time_s": float(self._t),
                "moisture_wb": float(feedstock.moisture_wb),
                "drying_time_ref_min": float(feedstock.drying_time_ref_min),
                "drying_sensitivity_min_per_C": float(feedstock.drying_sensitivity_min_per_C),
                "bulk_density_kg_m3": feedstock.bulk_density_kg_m3,
                "wet_mass_flow_kgps": feedstock.wet_mass_flow_kgps,
                "source": feedstock.source,
                "confidence": feedstock.confidence,
                "raw": dict(feedstock.raw),
                "composition": self._composition_with_variation(),
                "composition_manual": list(self.composition),
                "composition_names": list(NAMES),
            },
            "furnace": {
                "T_avg_C": float(obs.T_avg_C),
                "T_stack_C": float(obs.T_stack_C),
                "v_stack_mps": float(obs.v_stack_mps),
                "T_set_C": T_set_C,
                "T_compliance_min_C": T_floor,
                "temperature_error_C": float(obs.T_avg_C - T_set_C),
                "T_static_from_current_furnace_feed_C": _finite(T_static, float("nan")),
                "disturbance_Tavg_C": float(self._disturbance[0]),
                "disturbance_Tstack_C": float(self._disturbance[1]),
                "disturbance_vstack_mps": float(self._disturbance[2]),
            },
            "preheater": {
                "omega_out": float(pre_state.omega_out),
                "T_solid_out_C": float(pre_state.T_solid_out_C),
                "Tg_in_C": float(getattr(diag, "Tg_in_C", cmd.Tg_cmd_C)) if diag is not None else float(cmd.Tg_cmd_C),
                "Tg_out_C": float(getattr(diag, "Tg_out_C", getattr(pre_out, "Tg_out_C", float("nan")))) if diag is not None else _finite(getattr(pre_out, "Tg_out_C", float("nan")), float("nan")),
                "wet_out_kgps": float(wet_out),
                "dry_out_kgps": float(dry_out),
                "water_out_kgps": float(water_out),
                "water_evap_kgps": _finite(getattr(diag, "water_evap_kgps", float("nan")), float("nan")) if diag is not None else float("nan"),
                "Q_gas_to_solid_kW": _finite(getattr(diag, "Q_gas_to_solid_kW", float("nan")), float("nan")) if diag is not None else float("nan"),
                "Q_sensible_kW": _finite(getattr(diag, "Q_sensible_kW", float("nan")), float("nan")) if diag is not None else float("nan"),
                "Q_latent_kW": _finite(getattr(diag, "Q_latent_kW", float("nan")), float("nan")) if diag is not None else float("nan"),
            },
            "control": {
                "Tg_ref_C": float(decision.Tg_ref_C),
                "Tg_cmd_C": float(cmd.Tg_cmd_C),
                "vg_ref_mps": float(decision.vg_ref_mps),
                "vg_cmd_mps": float(cmd.vg_cmd_mps),
                "omega_target": float(decision.omega_target),
                "omega_reachable": float(decision.omega_reachable),
                "predicted_Tavg_C": float(decision.predicted_Tavg_C),
                "predicted_min_Tavg_C": float(decision.predicted_min_Tavg_C),
                "predicted_max_Tavg_C": float(decision.predicted_max_Tavg_C),
                "Q_aux_heat_kW": float(cmd.Q_aux_heat_kW),
                "operator_feasible": bool(decision.feasible),
                "operator_source": decision.source,
                "operator_note": decision.note,
                "operator_cost": _finite(decision.cost, float("nan")),
                "recovery_guard_active": recovery_guard_active,
                "recovery_guard_reason": self._last_recovery_reason,
                "safety_reachable": bool(decision.safety_reachable),
                "safety_margin_C": float(safety_margin_C),
                "mdot_stack_available_kgps": mdot_stack_available,
                "mdot_preheater_kgps": float(getattr(cmd, "mdot_preheater_kgps", float("nan"))),
                "mdot_aux_flow_kgps": float(getattr(cmd, "mdot_aux_flow_kgps", 0.0)),
                "fan_circulation_power_kW": float(getattr(cmd, "fan_circulation_power_kW", 0.0)),
                "aux_resource_required": bool(getattr(cmd, "aux_resource_required", False)),
                "aux_heat_required": bool(getattr(cmd, "aux_heat_required", False)),
                "aux_circulation_required": bool(getattr(cmd, "aux_circulation_required", False)),
                "nmpc_async_job_running": False,
                "nmpc_plan_age_s": max(0.0, float(self._t - decision.time_s)),
                "nmpc_last_solve_ms": float(self._solve_profile.solve_ms),
                "nmpc_stale_plan": False,
                "nmpc_candidates": int(self._solve_profile.candidates),
                "nmpc_horizon_steps": int(self._solve_profile.horizon_steps),
                "async_submitted_jobs": 0,
                "async_accepted_plans": 0,
                "async_discarded_plans": 0,
            },
            "health": {
                "ok": health_ok,
                "status": "running" if self.running else "paused",
                "stale": not self.running,
                "comms_ok": True,
                "sensors_ok": True,
                "actuators_ok": True,
                "message": "Realtime Python plant + fast rollout NMPC" if self.running else "监控已暂停，数据保持最近状态",
            },
        }
