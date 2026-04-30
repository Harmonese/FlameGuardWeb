from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plant.python_model.material_model import feedstock_from_composition
from plant.python_model.preheater import PreheaterForwardConfig, PreheaterForwardModel
from plant.python_model.furnace import furnace_outputs_from_omega

BASELINE_COMP = [0.20, 0.15, 0.15, 0.10, 0.20, 0.20]


def _run_constant_case(*, Tg_C: float, vg_mps: float, dt_s: float, horizon_s: float = 1800.0):
    model = PreheaterForwardModel(PreheaterForwardConfig(n_cells=20, tau_residence_s=985.0, feed_delay_s=5.0))
    model.initialize(feedstock_from_composition(0.0, BASELINE_COMP, source="test_composition"), omega_init=0.3218, T_solid_init_C=120.0, time_s=0.0)
    n = max(1, int(round(horizon_s / dt_s)))
    state = model.state()
    for k in range(n):
        t = (k + 1) * dt_s
        feed = feedstock_from_composition(t, BASELINE_COMP, source="test_composition")
        state = model.step(feed, Tg_C, vg_mps, dt_s)
    T_avg, _, _ = furnace_outputs_from_omega(state.omega_out)
    return state.omega_out, T_avg


def main() -> None:
    out_dir = Path('runtime/results')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'preheater_timestep_invariance.csv'
    dt_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    cases = [(800.0, 12.0), (930.0, 10.5), (930.0, 12.0)]
    rows = []
    failures = []
    for Tg, vg in cases:
        baseline_omega, baseline_T = _run_constant_case(Tg_C=Tg, vg_mps=vg, dt_s=1.0)
        for dt in dt_values:
            omega, T = _run_constant_case(Tg_C=Tg, vg_mps=vg, dt_s=dt)
            row = {
                'Tg_C': Tg,
                'vg_mps': vg,
                'dt_s': dt,
                'omega_out': omega,
                'Tavg_equiv_C': T,
                'delta_omega_vs_1s': omega - baseline_omega,
                'delta_Tavg_vs_1s_C': T - baseline_T,
            }
            rows.append(row)
            # The NMPC now uses a 5 s internal rollout by default.  Its steady
            # prediction should be reasonably close to a 1 s reference.  Larger
            # dt values are kept as diagnostics to show why the former 20 s
            # rollout was too coarse, but they are not hard-failed here.
            if dt <= 5.0 and abs(row['delta_omega_vs_1s']) > 0.003:
                failures.append(row)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'timestep invariance diagnostics -> {out_path}')
    if failures:
        for r in failures:
            print('FAIL', r)
        raise SystemExit(1)
    print('PASS: dt<=5s preheater steady omega stays within 0.003 of the 1s reference.')


if __name__ == '__main__':
    main()
