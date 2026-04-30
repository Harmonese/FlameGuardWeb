from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from runtime.simulator import SimConfig, run_case, save_case_artifacts
from runtime.tests.scenario_suite import CASES


def _parse_delays(text: str) -> list[float]:
    vals: list[float] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals or [0.0]


def _read_metric(path: Path, segment: str, key: str) -> str:
    if not path.exists():
        return ''
    with path.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('segment') == segment:
                return row.get(key, '')
    return ''


def run_latency_effect(
    *,
    case_name: str,
    delays_s: list[float],
    sim_time_s: float | None,
    out_dir: str,
    fast_operator: bool,
    save_artifacts: bool,
) -> list[dict[str, str | float]]:
    spec = CASES[case_name]
    summary: list[dict[str, str | float]] = []
    for delay in delays_s:
        cfg = SimConfig(**spec.cfg.__dict__)
        if sim_time_s is not None:
            cfg.total_time_s = float(sim_time_s)
            cfg.tail_window_s = min(cfg.tail_window_s, max(1.0, 0.25 * cfg.total_time_s))
        cfg.case_name = f'{case_name}_compute_latency_{delay:g}s'
        cfg.out_dir = out_dir
        cfg.compute_latency_mode = 'none' if delay <= 0.0 else 'fixed'
        cfg.fixed_compute_latency_s = max(0.0, float(delay))
        cfg.compute_latency_step_s = cfg.dt_meas_s
        # Keep this test usable in normal development loops.  The goal is to
        # isolate the effect of command-application delay, not to benchmark full
        # SLSQP quality.  Use --full-operator to keep scenario defaults.
        if fast_operator:
            cfg.mpc_horizon_s = min(cfg.mpc_horizon_s, 300.0)
            cfg.nmpc_rollout_dt_s = max(cfg.nmpc_rollout_dt_s, 10.0)
            cfg.nmpc_maxiter = min(cfg.nmpc_maxiter, 5)
        hist = run_case(spec.name, spec.composition_schedule, spec.disturbance_schedule, cfg)
        title = f'{spec.title} | compute latency {delay:g}s'
        outputs = save_case_artifacts(hist, cfg, title, event_windows=spec.event_windows) if save_artifacts else {}
        metrics_path = outputs.get('metrics')
        row = {
            'case_name': case_name,
            'fixed_compute_latency_s': delay,
            'artifact_dir': str(outputs.get('artifact_dir', '')),
            'timeseries': str(outputs.get('timeseries', '')),
            'metrics': str(metrics_path or ''),
            'full_Tavg_min_C': _read_metric(metrics_path, 'full', 'Tavg_min_C') if metrics_path else '',
            'full_safe_ratio': _read_metric(metrics_path, 'full', 'ratio_in_safe_band') if metrics_path else '',
            'recovery_to_safe_s': _read_metric(metrics_path, 'recovery', 'recovery_to_safe_s') if metrics_path else '',
            'tail_Tavg_mean_C': _read_metric(metrics_path, 'tail', 'Tavg_mean_C') if metrics_path else '',
            'simulated_compute_latency_max_s': _read_metric(metrics_path, 'full', 'simulated_compute_latency_max_s') if metrics_path else '',
            'operator_compute_wall_max_s': _read_metric(metrics_path, 'full', 'operator_compute_wall_max_s') if metrics_path else '',
            'operator_stale_plan_ratio': _read_metric(metrics_path, 'full', 'operator_stale_plan_ratio') if metrics_path else '',
        }
        summary.append(row)
        print(
            f"delay={delay:g}s safe_ratio={row['full_safe_ratio']} "
            f"recovery_to_safe={row['recovery_to_safe_s']} tail_Tavg={row['tail_Tavg_mean_C']}"
        )
    out_path = Path(out_dir) / f'{case_name}_compute_latency_effect_summary.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(f'compute latency summary -> {out_path}')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare closed-loop behavior under fixed controller compute latency.')
    parser.add_argument('--case', default='furnace_temp_permanent_disturbance', choices=sorted(CASES.keys()))
    parser.add_argument('--delays', default='0,2,5,10,30', help='Comma-separated fixed compute delays in seconds.')
    parser.add_argument('--sim-time', type=float, default=None, help='Override scenario total_time_s.')
    parser.add_argument('--out-dir', default='runtime/results')
    parser.add_argument('--full-operator', action='store_true', help='Use scenario NMPC defaults instead of faster test operator settings.')
    parser.add_argument('--no-artifacts', action='store_true', help='Do not write timeseries/metrics/plots; summary rows will be sparse.')
    args = parser.parse_args()
    run_latency_effect(
        case_name=args.case,
        delays_s=_parse_delays(args.delays),
        sim_time_s=args.sim_time,
        out_dir=args.out_dir,
        fast_operator=not args.full_operator,
        save_artifacts=not args.no_artifacts,
    )


if __name__ == '__main__':
    main()
