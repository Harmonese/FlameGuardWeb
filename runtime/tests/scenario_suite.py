from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import csv

from runtime.simulator import EventWindow, SimConfig, print_metrics_table, run_case, save_case_artifacts


CompositionFn = Callable[[float], list[float]]
DisturbanceFn = Callable[[float], float | tuple[float, float, float]]


BASELINE_COMP = [0.20, 0.15, 0.15, 0.10, 0.20, 0.20]
WETTER_COMP = [0.25, 0.25, 0.20, 0.05, 0.15, 0.10]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    title: str
    cfg: SimConfig
    composition_schedule: CompositionFn
    disturbance_schedule: DisturbanceFn | None
    event_windows: Sequence[EventWindow]
    description: str


def _constant_comp(comp: list[float]) -> CompositionFn:
    def f(_: float) -> list[float]:
        return list(comp)
    return f


def _feed_step_comp(step_time_s: float, before: list[float], after: list[float]) -> CompositionFn:
    def f(t: float) -> list[float]:
        return list(before if t < step_time_s else after)
    return f


def _disturbance_pulse(start_s: float, end_s: float, magnitude: float) -> DisturbanceFn:
    def f(t: float) -> float:
        return float(magnitude if start_s <= t < end_s else 0.0)
    return f


def _disturbance_step(start_s: float, magnitude: float) -> DisturbanceFn:
    def f(t: float) -> float:
        return float(magnitude if t >= start_s else 0.0)
    return f


def build_cases() -> dict[str, CaseSpec]:
    out_dir = 'runtime/results'
    return {
        'steady_hold': CaseSpec(
            name='steady_hold',
            title='Closed-loop test: steady hold under fixed feed and no external disturbance',
            description='No-disturbance hold used to evaluate low-frequency hunting and steady metrics.',
            cfg=SimConfig(case_name='steady_hold', total_time_s=3600.0, out_dir=out_dir, tail_window_s=900.0),
            composition_schedule=_constant_comp(BASELINE_COMP),
            disturbance_schedule=None,
            event_windows=[],
        ),
        'feed_step_change': CaseSpec(
            name='feed_step_change',
            title='Closed-loop test: permanent feed-composition step at 900 s',
            description='Permanent wetter feed step after a shortened baseline settling period.',
            cfg=SimConfig(case_name='feed_step_change', total_time_s=5400.0, out_dir=out_dir,
                          event_start_s=900.0, event_end_s=900.0, tail_window_s=900.0),
            composition_schedule=_feed_step_comp(900.0, BASELINE_COMP, WETTER_COMP),
            disturbance_schedule=None,
            event_windows=[EventWindow(900.0, 5400.0, 'feed step')],
        ),
        'furnace_temp_temporary_disturbance': CaseSpec(
            name='furnace_temp_temporary_disturbance',
            title='Closed-loop test: temporary furnace temperature bias pulse (900-1200 s)',
            description='Moderate additive T_avg disturbance pulse to test emergency recovery.',
            cfg=SimConfig(case_name='furnace_temp_temporary_disturbance', total_time_s=5400.0, out_dir=out_dir,
                          event_start_s=900.0, event_end_s=1200.0, tail_window_s=900.0),
            composition_schedule=_constant_comp(BASELINE_COMP),
            disturbance_schedule=_disturbance_pulse(900.0, 1200.0, -120.0),
            event_windows=[EventWindow(900.0, 1200.0, 'temp pulse')],
        ),
        'furnace_temp_permanent_disturbance': CaseSpec(
            name='furnace_temp_permanent_disturbance',
            title='Closed-loop test: permanent furnace temperature bias after 900 s',
            description='Moderate permanent additive T_avg disturbance to evaluate long-run adaptation and control effort.',
            cfg=SimConfig(case_name='furnace_temp_permanent_disturbance', total_time_s=6000.0, out_dir=out_dir,
                          event_start_s=900.0, event_end_s=900.0, tail_window_s=900.0),
            composition_schedule=_constant_comp(BASELINE_COMP),
            disturbance_schedule=_disturbance_step(900.0, -80.0),
            event_windows=[EventWindow(900.0, 6000.0, 'temp step')],
        ),
        'cold_start': CaseSpec(
            name='cold_start',
            title='Closed-loop test: near-valid cold-start initialization without external disturbance',
            description='Custom lower-temperature initial furnace state within the proxy model validity neighborhood.',
            cfg=SimConfig(case_name='cold_start', total_time_s=6000.0, out_dir=out_dir, tail_window_s=900.0,
                          furnace_init_mode='custom', T_avg_init_C=700.0, T_stack_init_C=820.0,
                          v_stack_init_mps=17.0, omega_out_init=0.45),
            composition_schedule=_constant_comp(BASELINE_COMP),
            disturbance_schedule=None,
            event_windows=[EventWindow(0.0, 900.0, 'startup')],
        ),
    }


CASES = build_cases()


def run_standard_case(case_name: str):
    spec = CASES[case_name]
    hist = run_case(spec.name, spec.composition_schedule, spec.disturbance_schedule, spec.cfg)
    outputs = save_case_artifacts(hist, spec.cfg, spec.title, event_windows=spec.event_windows)
    print(f'[{spec.name}] {spec.description}')
    print(f'artifacts -> {outputs["artifact_dir"]}')
    print(f'overview  -> {outputs["plot"]}')
    print(f'timeseries-> {outputs["timeseries"]}')
    print(f'metrics   -> {outputs["metrics"]}')
    print_metrics_table(outputs['metrics'])
    return hist, outputs


def run_all_cases(out_dir: str = 'runtime/results'):
    summary_rows = []
    for name in CASES:
        _, outputs = run_standard_case(name)
        with outputs['metrics'].open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        recovery = next((r for r in rows if r.get('segment') == 'recovery'), None)
        tail = next((r for r in rows if r.get('segment') == 'tail'), None)
        full = next((r for r in rows if r.get('segment') == 'full'), None)
        summary_rows.append({
            'case_name': name,
            'description': CASES[name].description,
            'artifact_dir': str(outputs.get('artifact_dir', '')),
            'overview_plot': str(outputs.get('overview_plot', outputs.get('plot', ''))),
            'timeseries': str(outputs['timeseries']),
            'metrics': str(outputs['metrics']),
            'control_events': str(outputs.get('control_events', '')),
            'preheater_diagnostics': str(outputs.get('preheater_diagnostics', '')),
            'cell_snapshot': str(outputs.get('cell_snapshot', '')),
            'full_MAE_C': '' if full is None else full.get('Tavg_MAE_C', ''),
            'full_safe_ratio': '' if full is None else full.get('ratio_in_safe_band', ''),
            'tail_MAE_C': '' if tail is None else tail.get('Tavg_MAE_C', ''),
            'tail_ref_ratio': '' if tail is None else tail.get('ratio_in_ref_band', ''),
            'tail_omega_out_mean': '' if tail is None else tail.get('omega_out_mean', ''),
            'tail_Tsolid_out_mean_C': '' if tail is None else tail.get('T_solid_out_mean_C', ''),
            'tail_aux_heat_energy_kJ': '' if tail is None else tail.get('Q_aux_heat_energy_kJ', ''),
            'tail_fan_energy_kJ': '' if tail is None else tail.get('fan_energy_kJ', ''),
            'operator_fallback_ratio': '' if full is None else full.get('operator_fallback_ratio', ''),
            'operator_stale_plan_ratio': '' if full is None else full.get('operator_stale_plan_ratio', ''),
            'aux_resource_required_ratio': '' if full is None else full.get('aux_resource_required_ratio', ''),
            'recovery_to_safe_s': '' if recovery is None else recovery.get('recovery_to_safe_s', ''),
            'recovery_to_ref_s': '' if recovery is None else recovery.get('recovery_to_ref_s', ''),
            'overshoot_crossings': '' if recovery is None else recovery.get('overshoot_crossings', ''),
        })
    out_path = Path(out_dir) / 'suite_summary.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f'suite summary -> {out_path}')
    return out_path
