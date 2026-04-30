from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path

from domain.types import ActuatorCommand, ControlSetpoint, OperatorContext, PlantStepInput, ResourceBoundary, StackResourceMeasurement
from controller.predictor.feed_preview import KnownScheduleFeedPreview
from controller.factory import make_executor, make_operator, make_predictor_resource_model, make_state_estimator
from plant.python_model.material_model import feedstock_from_composition
from plant.factory import make_plant_backend
from runtime.simulator import SimConfig, _make_command, _normalize_disturbance
from runtime.tests.scenario_suite import CASES


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float('nan')
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def _summary(rows: list[dict], key: str) -> dict[str, float]:
    vals = [float(r[key]) for r in rows]
    return {
        'n': len(vals),
        'mean_ms': statistics.fmean(vals) if vals else float('nan'),
        'median_ms': _percentile(vals, 50.0),
        'p90_ms': _percentile(vals, 90.0),
        'p95_ms': _percentile(vals, 95.0),
        'p99_ms': _percentile(vals, 99.0),
        'max_ms': max(vals) if vals else float('nan'),
    }


def run_latency_case(
    case_name: str,
    *,
    total_time_s: float | None = None,
    nmpc_reoptimize_s: float | None = None,
    horizon_s: float | None = None,
    dt_pred_s: float | None = None,
    rollout_dt_s: float | None = None,
    maxiter: int | None = None,
    async_nmpc: bool = True,
    out_csv: Path | None = None,
) -> list[dict]:
    spec = CASES[case_name]
    cfg = spec.cfg
    # Avoid mutating the shared scenario config object in-place.
    cfg = SimConfig(**cfg.__dict__)
    if total_time_s is not None:
        cfg.total_time_s = float(total_time_s)
    if nmpc_reoptimize_s is not None:
        cfg.nmpc_reoptimize_s = float(nmpc_reoptimize_s)
    if horizon_s is not None:
        cfg.mpc_horizon_s = float(horizon_s)
    if dt_pred_s is not None:
        cfg.mpc_dt_s = float(dt_pred_s)
    if rollout_dt_s is not None:
        cfg.nmpc_rollout_dt_s = float(rollout_dt_s)
    if maxiter is not None:
        cfg.nmpc_maxiter = int(maxiter)

    tb = make_executor(cfg)
    predictor_resource_model = make_predictor_resource_model(cfg)
    resource = ResourceBoundary(cfg.aux_Tg_max_C, min(cfg.resource_v_stack_cap_mps, 12.0))

    def feedstock_schedule(time_s: float):
        return feedstock_from_composition(
            time_s,
            spec.composition_schedule(time_s),
            wet_mass_flow_kgps=cfg.wet_mass_flow_kgps,
            source="latency_test_composition_schedule",
        )

    feed_preview = KnownScheduleFeedPreview(feedstock_schedule)
    initial_feedstock = feedstock_schedule(0.0)
    plant = make_plant_backend(
        cfg,
        initial_feedstock=initial_feedstock,
        disturbance_schedule=spec.disturbance_schedule,
    )
    initial_snapshot = plant.reset()
    state_estimator = make_state_estimator(
        cfg,
        initial_snapshot=initial_snapshot,
        initial_feedstock=initial_feedstock,
        resource_model=predictor_resource_model,
    )
    cfg.nmpc_async = bool(async_nmpc)
    mpc = make_operator(cfg, resource_model=predictor_resource_model)
    mpc.initialize(Tg_ref_C=cfg.nominal_Tg_C, vg_ref_mps=cfg.nominal_vg_mps, time_s=0.0)

    if hasattr(tb, 'initialize_previous'):
        tb.initialize_previous(cfg.nominal_Tg_C, cfg.nominal_vg_mps)
    current_cmd: ActuatorCommand = _make_command(0.0, cfg.nominal_Tg_C, cfg.nominal_vg_mps)

    rows: list[dict] = []
    opt_elapsed = 0.0
    t = 0.0
    while t <= cfg.total_time_s + 1e-9:
        disturbance = None if spec.disturbance_schedule is None else spec.disturbance_schedule(t)
        d_avg, d_stack, d_v = _normalize_disturbance(disturbance)
        feedstock = feedstock_schedule(t)
        snapshot = plant.step(PlantStepInput(time_s=t, dt_s=cfg.dt_meas_s, command=current_cmd, feedstock=feedstock))
        obs = snapshot.furnace_obs
        Tavg, Tstack, vstack = obs.T_avg_C, obs.T_stack_C, obs.v_stack_mps
        estimate = state_estimator.update(
            snapshot,
            previous_command=current_cmd,
            feedstock=feedstock,
            dt_s=cfg.dt_meas_s,
        )
        stack_resource = estimate.stack_resource_est or snapshot.stack_resource or StackResourceMeasurement(t, Tstack, vstack, None)
        mdot_stack_available = (
            float(stack_resource.mdot_stack_available_kgps)
            if stack_resource.mdot_stack_available_kgps is not None
            else float("inf")
        )
        resource = ResourceBoundary(cfg.aux_Tg_max_C, min(cfg.resource_v_stack_cap_mps, 12.0))

        if opt_elapsed <= 1e-12:
            wall0 = time.perf_counter()
            op0 = time.perf_counter()
            decision = mpc.step_context(OperatorContext(
                estimate=estimate,
                predictors=state_estimator.get_predictor_bundle(),
                feedstock=feedstock,
                resource=resource,
                previous_command=current_cmd,
                feed_preview=feed_preview,
            ))
            op1 = time.perf_counter()
            setpoint = ControlSetpoint(
                time_s=t,
                Tg_ref_C=decision.Tg_ref_C,
                vg_ref_mps=decision.vg_ref_mps,
                source=decision.source,
                omega_target=decision.omega_target,
                omega_reachable=decision.omega_reachable,
                mdot_stack_cap_kgps=mdot_stack_available,
                T_stack_available_C=stack_resource.T_stack_available_C,
                v_stack_available_mps=stack_resource.v_stack_available_mps,
            )
            ex0 = time.perf_counter()
            current_cmd = tb.translate_setpoint(setpoint)
            ex1 = time.perf_counter()
            wall1 = time.perf_counter()
            profile = getattr(mpc, 'last_solve_profile', None)
            rows.append({
                'time_s': f'{t:.6f}',
                'source': decision.source,
                'operator_ms': f'{(op1 - op0) * 1000.0:.3f}',
                'executor_ms': f'{(ex1 - ex0) * 1000.0:.3f}',
                'total_control_ms': f'{(wall1 - wall0) * 1000.0:.3f}',
                'nmpc_total_solve_ms': '' if profile is None else f'{profile.total_ms:.3f}',
                'nmpc_seed_eval_ms': '' if profile is None else f'{profile.seed_eval_ms:.3f}',
                'nmpc_slsqp_minimize_ms': '' if profile is None else f'{profile.minimize_ms:.3f}',
                'nmpc_final_eval_ms': '' if profile is None else f'{profile.final_eval_ms:.3f}',
                'nmpc_nfev': '' if profile is None else str(profile.nfev),
                'nmpc_nit': '' if profile is None else str(profile.nit),
                'nmpc_success': '' if profile is None else str(profile.success),
                'T_avg_C': f'{Tavg:.6f}',
                'Tg_ref_C': f'{decision.Tg_ref_C:.6f}',
                'vg_ref_mps': f'{decision.vg_ref_mps:.6f}',
                'Tg_cmd_C': f'{current_cmd.Tg_cmd_C:.6f}',
                'vg_cmd_mps': f'{current_cmd.vg_cmd_mps:.6f}',
                'note': decision.note,
                'async_job_running': str(getattr(getattr(mpc, 'status', None), 'job_running', '')),
                'async_last_solve_ms': str(getattr(getattr(mpc, 'status', None), 'last_solve_ms', '')),
                'async_plan_age_s': str(getattr(getattr(mpc, 'status', None), 'active_plan_age_s', '')),
            })

        opt_elapsed += cfg.dt_meas_s
        if opt_elapsed >= cfg.dt_opt_s - 1e-12:
            opt_elapsed = 0.0
        t += cfg.dt_meas_s

    if hasattr(mpc, 'shutdown'):
        mpc.shutdown(wait=True)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Measure controller decision latency from observation to ActuatorCommand.')
    ap.add_argument('--case', default='steady_hold', choices=sorted(CASES))
    ap.add_argument('--sim-time', type=float, default=300.0, help='Physical seconds to simulate for the benchmark.')
    ap.add_argument('--reoptimize-s', type=float, default=None, help='Override NMPC reoptimization period.')
    ap.add_argument('--horizon-s', type=float, default=None, help='Override NMPC prediction horizon.')
    ap.add_argument('--dt-pred-s', type=float, default=None, help='Override NMPC prediction grid step.')
    ap.add_argument('--rollout-dt', type=float, default=None, help='Override NMPC rollout integration step.')
    ap.add_argument('--maxiter', type=int, default=None, help='Override SLSQP max iterations.')
    ap.add_argument('--out', default='runtime/results/control_latency.csv')
    ap.add_argument('--disable-scipy', action='store_true', help='Force nominal fallback to benchmark non-SLSQP overhead.')
    ap.add_argument('--sync-nmpc', action='store_true', help='Use synchronous SLSQP NMPC instead of the default asynchronous wrapper.')
    args = ap.parse_args()
    if args.disable_scipy:
        os.environ['FLAMEGUARD_DISABLE_SCIPY'] = '1'
    rows = run_latency_case(
        args.case,
        total_time_s=args.sim_time,
        nmpc_reoptimize_s=args.reoptimize_s,
        horizon_s=args.horizon_s,
        dt_pred_s=args.dt_pred_s,
        rollout_dt_s=args.rollout_dt,
        maxiter=args.maxiter,
        async_nmpc=not args.sync_nmpc,
        out_csv=Path(args.out),
    )
    print(f'latency csv -> {args.out}')
    for label, subset in [('all', rows), ('reoptimize', [r for r in rows if r['source'] == 'nmpc_block_slsqp']), ('plan_hold', [r for r in rows if r['source'] in ('nmpc_plan_hold', 'async_nmpc_plan_hold')]), ('fallback', [r for r in rows if 'fallback' in r['source']])]:
        if not subset:
            continue
        stats = _summary(subset, 'total_control_ms')
        print(f'{label}: n={stats["n"]} mean={stats["mean_ms"]:.3f} ms median={stats["median_ms"]:.3f} ms p95={stats["p95_ms"]:.3f} ms p99={stats["p99_ms"]:.3f} ms max={stats["max_ms"]:.3f} ms')
    sources = {}
    for r in rows:
        sources[r['source']] = sources.get(r['source'], 0) + 1
    print('sources:', sources)


if __name__ == '__main__':
    main()
