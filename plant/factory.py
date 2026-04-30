from __future__ import annotations

import math
from typing import Any, Callable

from domain.interfaces import PlantBackend
from domain.types import FeedstockObservation

from plant.python_model.backend import PythonPlantBackend
from plant.python_model.furnace import (
    FurnaceDyn,
    FurnaceDynConfig,
    furnace_feed_from_preheater_output,
    furnace_outputs_from_omega,
)
from plant.python_model.preheater import PreheaterForwardConfig, PreheaterForwardModel
from plant.python_model.resource import ResourceModel, ResourceModelConfig


def _warm_start_preheater(preheater: PreheaterForwardModel, *, feedstock: FeedstockObservation, cfg: Any) -> None:
    warm_s = max(0.0, float(getattr(cfg, "preheater_warmup_s", 0.0)))
    if warm_s <= 0.0:
        return
    dt = max(1e-6, float(getattr(cfg, "preheater_warmup_dt_s", 20.0)))
    n_steps = max(1, int(math.ceil(warm_s / dt)))
    nominal_Tg_C = float(getattr(cfg, "nominal_Tg_C", 800.0))
    nominal_vg_mps = float(getattr(cfg, "nominal_vg_mps", 12.0))
    for k in range(n_steps):
        t_warm = -warm_s + (k + 1) * dt
        warm_feedstock = type(feedstock)(
            time_s=t_warm,
            moisture_wb=feedstock.moisture_wb,
            drying_time_ref_min=feedstock.drying_time_ref_min,
            drying_sensitivity_min_per_C=feedstock.drying_sensitivity_min_per_C,
            bulk_density_kg_m3=feedstock.bulk_density_kg_m3,
            wet_mass_flow_kgps=feedstock.wet_mass_flow_kgps,
            source=feedstock.source,
            confidence=feedstock.confidence,
            raw=feedstock.raw,
        )
        preheater.step(warm_feedstock, nominal_Tg_C, nominal_vg_mps, dt)
    preheater.time_s = 0.0


def make_python_plant_backend(
    cfg: Any,
    *,
    initial_feedstock: FeedstockObservation,
    disturbance_schedule: Callable[[float], float | tuple[float, float, float]] | None = None,
) -> PythonPlantBackend:
    """Build the default Python low-order plant backend from runtime config.

    Runtime should not manually construct the preheater/furnace/resource model;
    this factory is the single assembly point for the Python plant backend.
    """
    init_omega = (
        float(getattr(cfg, "omega_out_init"))
        if getattr(cfg, "omega_out_init", None) is not None
        else float(getattr(cfg, "omega_ref", 0.3218))
    )
    preheater = PreheaterForwardModel(
        PreheaterForwardConfig(
            n_cells=int(getattr(cfg, "preheater_n_cells", 20)),
            tau_residence_s=float(getattr(cfg, "pre_tau_s", 985.0)),
            feed_delay_s=float(getattr(cfg, "pre_dead_s", 5.0)),
        )
    )
    preheater.initialize(initial_feedstock, omega_init=init_omega, T_solid_init_C=120.0, time_s=0.0)
    if getattr(cfg, "furnace_init_mode", "warm") != "custom":
        _warm_start_preheater(preheater, feedstock=initial_feedstock, cfg=cfg)

    furnace = FurnaceDyn(
        FurnaceDynConfig(
            omega_ref=float(getattr(cfg, "omega_ref", 0.3218)),
            dead_s=float(getattr(cfg, "furnace_dead_s", 5.0)),
            tau1_s=float(getattr(cfg, "furnace_tau1_s", 0.223)),
            tau2_s=float(getattr(cfg, "furnace_tau2_s", 75.412)),
            dt_s=float(getattr(cfg, "dt_meas_s", 0.1)),
        )
    )
    if getattr(cfg, "furnace_init_mode", "warm") == "custom":
        T_avg = getattr(cfg, "T_avg_init_C", None)
        T_stack = getattr(cfg, "T_stack_init_C", None)
        v_stack = getattr(cfg, "v_stack_init_mps", None)
        omega_ref = float(getattr(cfg, "omega_ref", 0.3218))
        default_Tavg, default_Tstack, default_vstack = furnace_outputs_from_omega(omega_ref)
        furnace.initialize_outputs(
            float(T_avg) if T_avg is not None else float(getattr(cfg, "T_set_C", default_Tavg)),
            float(T_stack) if T_stack is not None else float(default_Tstack),
            float(v_stack) if v_stack is not None else float(default_vstack),
        )
    else:
        init_pre_state = preheater.state(time_s=0.0)
        diag = getattr(preheater, "last_diagnostics", None)
        furnace_feed = furnace_feed_from_preheater_output(
            time_s=0.0,
            omega_b=init_pre_state.omega_out,
            mdot_d_kgps=None if diag is None else getattr(diag, "dry_out_kgps", None),
            mdot_water_kgps=None if diag is None else getattr(diag, "water_out_kgps", None),
            mdot_wet_kgps=None if diag is None else getattr(diag, "wet_out_kgps", None),
            mdot_d_ref_kgps=furnace.cfg.mdot_d_ref_kgps,
        )
        furnace.initialize_from_omega(furnace_feed.omega_b, mdot_d_kgps=furnace_feed.mdot_d_kgps)

    resource_model = ResourceModel(
        ResourceModelConfig(
            stack_to_preheater_loss_C=float(getattr(cfg, "stack_to_preheater_loss_C", 0.0)),
            extractable_velocity_fraction=float(getattr(cfg, "extractable_velocity_fraction", 1.0)),
            aux_Tg_max_C=float(getattr(cfg, "aux_Tg_max_C", 1100.0)),
            vg_cmd_max_mps=12.0,
        )
    )
    return PythonPlantBackend(
        preheater=preheater,
        furnace=furnace,
        resource_model=resource_model,
        disturbance_schedule=disturbance_schedule,
    )


def make_plant_backend(
    cfg: Any,
    *,
    initial_feedstock: FeedstockObservation,
    disturbance_schedule: Callable[[float], float | tuple[float, float, float]] | None = None,
) -> PlantBackend:
    backend_name = str(getattr(cfg, "plant_backend", "python") or "python").lower()
    if backend_name in {"python", "python_model", "python-model"}:
        return make_python_plant_backend(
            cfg,
            initial_feedstock=initial_feedstock,
            disturbance_schedule=disturbance_schedule,
        )
    if backend_name == "comsol":
        from plant.comsol.backend import ComsolPlantBackend
        return ComsolPlantBackend()
    if backend_name == "hardware":
        from plant.hardware.backend import HardwarePlantBackend
        return HardwarePlantBackend()
    raise ValueError(f"Unknown plant_backend={backend_name!r}; expected python/comsol/hardware")
