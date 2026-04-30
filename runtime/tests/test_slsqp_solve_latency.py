from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from runtime.tests.scenario_suite import CASES
from runtime.tests.test_control_latency import run_latency_case


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def _nums(rows: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = str(r.get(key, "")).strip()
        if not v:
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def _summary(rows: list[dict], key: str) -> dict[str, float]:
    vals = _nums(rows, key)
    return {
        "n": len(vals),
        "mean_ms": statistics.fmean(vals) if vals else float("nan"),
        "median_ms": _percentile(vals, 50.0),
        "p90_ms": _percentile(vals, 90.0),
        "p95_ms": _percentile(vals, 95.0),
        "p99_ms": _percentile(vals, 99.0),
        "max_ms": max(vals) if vals else float("nan"),
    }


def _print_summary(label: str, rows: list[dict], key: str) -> None:
    s = _summary(rows, key)
    if s["n"] == 0:
        print(f"{label}: n=0")
        return
    print(
        f"{label}: n={s['n']} mean={s['mean_ms']:.3f} ms "
        f"median={s['median_ms']:.3f} ms p95={s['p95_ms']:.3f} ms "
        f"p99={s['p99_ms']:.3f} ms max={s['max_ms']:.3f} ms"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark synchronous SLSQP NMPC solve time. Defaults use a reduced "
            "horizon/maxiter so the script returns quickly; pass --full-config for "
            "the production-size 600 s horizon and scenario maxiter."
        )
    )
    ap.add_argument("--case", default="steady_hold", choices=sorted(CASES))
    ap.add_argument("--sim-time", type=float, default=1.0, help="Physical seconds to simulate.")
    ap.add_argument("--out", default="runtime/results/slsqp_solve_latency.csv")
    ap.add_argument("--full-config", action="store_true", help="Use scenario horizon/rollout/maxiter instead of quick benchmark defaults.")
    ap.add_argument("--horizon-s", type=float, default=120.0, help="Quick-mode NMPC horizon override.")
    ap.add_argument("--dt-pred-s", type=float, default=20.0, help="Quick-mode NMPC prediction grid override.")
    ap.add_argument("--rollout-dt", type=float, default=20.0, help="Quick-mode rollout step override. NMPC caps it at dt_pred_s.")
    ap.add_argument("--maxiter", type=int, default=1, help="Quick-mode SLSQP maxiter override.")
    ap.add_argument("--reoptimize-s", type=float, default=0.0, help="Force reoptimization cadence; 0 means every control call.")
    args = ap.parse_args()

    rows = run_latency_case(
        args.case,
        total_time_s=args.sim_time,
        nmpc_reoptimize_s=args.reoptimize_s,
        horizon_s=None if args.full_config else args.horizon_s,
        dt_pred_s=None if args.full_config else args.dt_pred_s,
        rollout_dt_s=None if args.full_config else args.rollout_dt,
        maxiter=None if args.full_config else args.maxiter,
        async_nmpc=False,
        out_csv=Path(args.out),
    )
    solve_rows = [r for r in rows if str(r.get("source", "")) == "nmpc_block_slsqp"]
    print(f"slsqp latency csv -> {args.out}")
    print(f"rows={len(rows)} slsqp_solves={len(solve_rows)}")
    _print_summary("total_sync_operator", solve_rows, "operator_ms")
    _print_summary("total_nmpc_solve", solve_rows, "nmpc_total_solve_ms")
    _print_summary("seed_rollouts", solve_rows, "nmpc_seed_eval_ms")
    _print_summary("slsqp_minimize", solve_rows, "nmpc_slsqp_minimize_ms")
    _print_summary("final_rollout", solve_rows, "nmpc_final_eval_ms")
    nfev = _nums(solve_rows, "nmpc_nfev")
    nit = _nums(solve_rows, "nmpc_nit")
    if nfev:
        print(f"nfev: mean={statistics.fmean(nfev):.1f} max={max(nfev):.0f}")
    if nit:
        print(f"nit: mean={statistics.fmean(nit):.1f} max={max(nit):.0f}")
    if not solve_rows:
        sources: dict[str, int] = {}
        for r in rows:
            src = str(r.get("source", ""))
            sources[src] = sources.get(src, 0) + 1
        print(f"sources: {sources}")
        print("No nmpc_block_slsqp rows were produced. Check that scipy is installed and not disabled by FLAMEGUARD_DISABLE_SCIPY.")
    if solve_rows:
        print("first_solve:")
        first = solve_rows[0]
        for key in [
            "time_s", "nmpc_total_solve_ms", "nmpc_seed_eval_ms", "nmpc_slsqp_minimize_ms",
            "nmpc_final_eval_ms", "nmpc_nfev", "nmpc_nit", "nmpc_success", "note",
        ]:
            print(f"  {key}: {first.get(key, '')}")


if __name__ == "__main__":
    main()
