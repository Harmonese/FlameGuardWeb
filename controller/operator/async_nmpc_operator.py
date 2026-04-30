from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np

from domain.types import ActuatorCommand, FeedstockObservation, FurnaceObservation, MPCDecision, OperatorContext, ResourceBoundary, StateEstimate
from controller.operator.nmpc_operator import NMPCConfig, NonlinearMPCController
from controller.predictor.preheater import PreheaterForwardModel
from controller.predictor.furnace import FurnaceDyn
from controller.predictor.feed_preview import FeedPreviewProvider
from controller.predictor.resource import ResourceModel


@dataclass
class AsyncNMPCStatus:
    """Runtime diagnostics for the asynchronous NMPC wrapper."""

    job_running: bool = False
    last_submit_time_s: float | None = None
    last_success_time_s: float | None = None
    last_solve_ms: float | None = None
    last_status: str = "idle"
    last_error: str = ""
    active_plan_start_s: float | None = None
    active_plan_age_s: float | None = None
    submitted_job_count: int = 0
    accepted_plan_count: int = 0
    discarded_plan_count: int = 0
    last_result_state_time_s: float | None = None
    last_result_state_age_s: float | None = None
    last_accepted_state_time_s: float | None = None


class AsyncNonlinearMPCController:
    """Non-blocking wrapper around :class:`NonlinearMPCController`.

    The synchronous NMPC solver can require several seconds when SLSQP evaluates
    many full preheater/furnace rollouts.  This wrapper keeps the foreground
    control loop deterministic: ``step`` immediately returns the currently active
    cached plan while a single background worker solves the next NMPC problem on
    cloned predictor states.  When the worker succeeds, the solved plan atomically
    replaces the active plan.

    The wrapper is intended for deployment-style timing.  Offline closed-loop
    studies may still use ``NonlinearMPCController`` directly when they need the
    simulated clock to wait for every optimized plan.
    """

    def __init__(
        self,
        cfg: NMPCConfig | None = None,
        *,
        resource_model: ResourceModel | None = None,
        max_workers: int = 1,
        stale_plan_timeout_s: float = 300.0,
    ):
        self.cfg = cfg or NMPCConfig()
        self.resource_model = resource_model or ResourceModel()
        self.stale_plan_timeout_s = float(stale_plan_timeout_s)
        self._active = NonlinearMPCController(self.cfg, resource_model=self.resource_model)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="flameguard-nmpc")
        self._lock = threading.RLock()
        self._future: Future | None = None
        self._job_id = 0
        self.status = AsyncNMPCStatus()

    def initialize(self, *, Tg_ref_C: float, vg_ref_mps: float, time_s: float = 0.0) -> None:
        with self._lock:
            self._active.initialize(Tg_ref_C=Tg_ref_C, vg_ref_mps=vg_ref_mps, time_s=time_s)
            self.status.active_plan_start_s = float(time_s)
            self.status.active_plan_age_s = 0.0
            self.status.last_status = "initialized"

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def _solver_snapshot(self) -> NonlinearMPCController:
        """Copy warm-start state from the active solver into an isolated solver."""
        solver = NonlinearMPCController(self.cfg, resource_model=self.resource_model)
        with self._lock:
            solver.prev_Tg_ref_C = self._active.prev_Tg_ref_C
            solver.prev_vg_ref_mps = self._active.prev_vg_ref_mps
            solver.prev_solution = None if self._active.prev_solution is None else np.asarray(self._active.prev_solution, dtype=float).copy()
            solver.plan_start_s = self._active.plan_start_s
            # Force the isolated solver to optimize immediately.
            solver.next_reopt_s = -math.inf
            solver.next_emergency_reopt_s = -math.inf
            solver.last_decision = self._active.last_decision
        return solver

    @staticmethod
    def _solve_job(
        *,
        job_id: int,
        solver: NonlinearMPCController,
        estimate: StateEstimate,
        preheater_predictor: PreheaterForwardModel,
        furnace_predictor: FurnaceDyn,
        feed: FeedstockObservation,
        resource: ResourceBoundary,
        prev_cmd: ActuatorCommand | None,
        disturbance: Any,
        feed_preview: FeedPreviewProvider | None,
    ):
        t0 = time.perf_counter()
        decision = solver.step(
            estimate=estimate,
            preheater_predictor=preheater_predictor,
            furnace_predictor=furnace_predictor,
            feed=feed,
            resource=resource,
            prev_cmd=prev_cmd,
            disturbance=disturbance,
            feed_preview=feed_preview,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0
        return job_id, solver, decision, solve_ms

    def _collect_completed(self, *, current_time_s: float | None = None) -> None:
        fut = self._future
        if fut is None or not fut.done():
            return
        with self._lock:
            fut = self._future
            if fut is None or not fut.done():
                return
            self._future = None
            self.status.job_running = False
            try:
                job_id, solver, decision, solve_ms = fut.result()
            except Exception as exc:  # pragma: no cover - defensive path
                self.status.last_status = "failed"
                self.status.last_error = str(exc)
                return

            self.status.last_solve_ms = float(solve_ms)
            self.status.last_submit_time_s = decision.time_s
            self.status.last_result_state_time_s = decision.time_s
            if current_time_s is not None:
                self.status.last_result_state_age_s = max(0.0, float(current_time_s) - float(decision.time_s))
            else:
                self.status.last_result_state_age_s = None

            if job_id != self._job_id:
                self.status.discarded_plan_count += 1
                self.status.last_status = "stale_result_discarded"
                return

            result_too_old = (
                self.status.last_result_state_age_s is not None
                and self.status.last_result_state_age_s > self.stale_plan_timeout_s
            )
            if result_too_old:
                self.status.discarded_plan_count += 1
                self.status.last_status = "stale_result_age_discarded"
                self.status.last_error = f"result_state_age_s={self.status.last_result_state_age_s:.1f}"
                return

            plan_valid = (
                decision.source == "nmpc_block_slsqp"
                and math.isfinite(float(decision.cost))
                and math.isfinite(float(decision.Tg_ref_C))
                and math.isfinite(float(decision.vg_ref_mps))
                and self.cfg.Tg_min_C - 1e-6 <= float(decision.Tg_ref_C) <= self.cfg.Tg_max_C + 1e-6
                and self.cfg.vg_min_mps - 1e-6 <= float(decision.vg_ref_mps) <= self.cfg.vg_max_mps + 1e-6
            )
            if plan_valid:
                # SLSQP often returns a useful bounded best-effort plan with
                # success=False when it hits maxiter.  Do not discard that plan;
                # reserve decision.feasible for optimizer-success telemetry.
                self._active = solver
                self.status.accepted_plan_count += 1
                self.status.last_success_time_s = decision.time_s if decision.feasible else self.status.last_success_time_s
                self.status.last_accepted_state_time_s = decision.time_s
                self.status.active_plan_start_s = solver.plan_start_s
                self.status.last_status = "success" if decision.feasible else "accepted_best_effort"
                self.status.last_error = "" if decision.feasible else decision.note
            else:
                # Keep the previous active plan when the background job produced
                # no finite bounded NMPC trajectory. Foreground safety remains
                # handled by executor recovery guard.
                self.status.discarded_plan_count += 1
                self.status.last_status = f"ignored_{decision.source}"
                self.status.last_error = decision.note

    def _nominal_decision(self, estimate: StateEstimate, *, note: str) -> MPCDecision:
        omega = float(estimate.preheater_state_est.omega_out)
        obs = estimate.furnace_obs
        return MPCDecision(
            time_s=obs.time_s,
            Tg_ref_C=self.cfg.nominal_Tg_C,
            vg_ref_mps=self.cfg.nominal_vg_mps,
            omega_target=self.cfg.omega_ref,
            omega_reachable=omega,
            predicted_Tavg_C=obs.T_avg_C,
            predicted_omega_out=omega,
            cost=float("inf"),
            feasible=False,
            source="async_nmpc_nominal_hold",
            safety_reachable=obs.T_avg_C >= self.cfg.T_safe_low_C,
            predicted_min_Tavg_C=obs.T_avg_C,
            predicted_max_Tavg_C=obs.T_avg_C,
            safety_margin_C=obs.T_avg_C - self.cfg.T_safe_low_C,
            note=note,
        )

    def _submit_if_due(
        self,
        *,
        estimate: StateEstimate,
        preheater_predictor: PreheaterForwardModel,
        furnace_predictor: FurnaceDyn,
        feed: FeedstockObservation,
        resource: ResourceBoundary,
        prev_cmd: ActuatorCommand | None,
        disturbance: Any,
        feed_preview: FeedPreviewProvider | None,
    ) -> bool:
        obs = estimate.furnace_obs
        with self._lock:
            if self._future is not None:
                return False
            due = obs.time_s + 1e-9 >= self._active.next_reopt_s
            emergency_due = self._active._emergency_reoptimize_due(obs)
            if not (due or emergency_due):
                return False
            self._job_id += 1
            job_id = self._job_id
            solver = self._solver_snapshot()
            self.status.job_running = True
            self.status.submitted_job_count += 1
            self.status.last_submit_time_s = obs.time_s
            self.status.last_status = "running"
            self.status.last_error = ""

        # Clone outside the lock so foreground access is not blocked by copies.
        p = preheater_predictor.clone()
        f = furnace_predictor.clone()
        fut = self._pool.submit(
            self._solve_job,
            job_id=job_id,
            solver=solver,
            estimate=estimate,
            preheater_predictor=p,
            furnace_predictor=f,
            feed=feed,
            resource=resource,
            prev_cmd=prev_cmd,
            disturbance=disturbance,
            feed_preview=feed_preview,
        )
        with self._lock:
            self._future = fut
        return True


    def step_context(self, context: OperatorContext) -> MPCDecision:
        predictors = context.predictors
        return self.step(
            estimate=context.estimate,
            preheater_predictor=predictors.preheater,
            furnace_predictor=predictors.furnace,
            feed=context.feedstock,
            resource=context.resource,
            prev_cmd=context.previous_command,
            disturbance=None,
            feed_preview=context.feed_preview,
        )

    def step(
        self,
        *,
        estimate: StateEstimate,
        preheater_predictor: PreheaterForwardModel,
        furnace_predictor: FurnaceDyn,
        feed: FeedstockObservation,
        resource: ResourceBoundary,
        prev_cmd: ActuatorCommand | None = None,
        disturbance=None,
        feed_preview: FeedPreviewProvider | None = None,
    ) -> MPCDecision:
        obs = estimate.furnace_obs
        self._collect_completed(current_time_s=obs.time_s)

        with self._lock:
            planned = self._active._planned_decision(estimate)
            active_plan_start_s = self._active.plan_start_s
            if active_plan_start_s is not None:
                self.status.active_plan_start_s = active_plan_start_s
                self.status.active_plan_age_s = max(0.0, obs.time_s - active_plan_start_s)
            else:
                self.status.active_plan_age_s = None

        submitted = self._submit_if_due(
            estimate=estimate,
            preheater_predictor=preheater_predictor,
            furnace_predictor=furnace_predictor,
            feed=feed,
            resource=resource,
            prev_cmd=prev_cmd,
            disturbance=disturbance,
            feed_preview=feed_preview,
        )

        if planned is None:
            return self._nominal_decision(
                estimate,
                note="No active NMPC plan available; submitted background optimization" if submitted else "No active NMPC plan available",
            )

        age = self.status.active_plan_age_s
        stale = age is not None and age > self.stale_plan_timeout_s
        note_parts = ["Using active NMPC plan; background optimization is non-blocking"]
        if submitted:
            note_parts.append("submitted background solve")
        if self.status.job_running:
            note_parts.append("solve running")
        if self.status.last_solve_ms is not None:
            note_parts.append(f"last_solve_ms={self.status.last_solve_ms:.1f}")
        if stale:
            note_parts.append(f"WARNING stale_plan_age_s={age:.1f}")

        return MPCDecision(
            time_s=planned.time_s,
            Tg_ref_C=planned.Tg_ref_C,
            vg_ref_mps=planned.vg_ref_mps,
            omega_target=planned.omega_target,
            omega_reachable=planned.omega_reachable,
            predicted_Tavg_C=planned.predicted_Tavg_C,
            predicted_omega_out=planned.predicted_omega_out,
            cost=planned.cost,
            feasible=planned.feasible and not stale,
            source="async_nmpc_plan_hold",
            safety_reachable=planned.safety_reachable,
            predicted_min_Tavg_C=planned.predicted_min_Tavg_C,
            predicted_max_Tavg_C=planned.predicted_max_Tavg_C,
            omega_max_for_safety=planned.omega_max_for_safety,
            safety_margin_C=planned.safety_margin_C,
            note="; ".join(note_parts),
        )
