from __future__ import annotations

from typing import Any

from domain.types import FeedstockObservation, PlantSnapshot

from controller.estimator.furnace_disturbance_observer import FurnaceDisturbanceObserver
from controller.estimator.state_estimator import ControllerStateEstimator
from controller.executor.executor import ControlExecutor
from controller.operator.async_nmpc_operator import AsyncNonlinearMPCController
from controller.operator.nmpc_operator import NMPCConfig, NonlinearMPCController
from controller.predictor.furnace import FurnaceDyn, FurnaceDynConfig
from controller.predictor.preheater import PreheaterForwardConfig, PreheaterForwardModel
from controller.predictor.resource import ResourceModel, ResourceModelConfig
from plant.python_model.furnace import furnace_feed_from_preheater_output


def make_predictor_resource_model(cfg: Any) -> ResourceModel:
    return ResourceModel(
        ResourceModelConfig(
            stack_to_preheater_loss_C=float(getattr(cfg, "stack_to_preheater_loss_C", 0.0)),
            extractable_velocity_fraction=float(getattr(cfg, "extractable_velocity_fraction", 1.0)),
            aux_Tg_max_C=float(getattr(cfg, "aux_Tg_max_C", 1100.0)),
            vg_cmd_max_mps=12.0,
        )
    )


def make_state_estimator(
    cfg: Any,
    *,
    initial_snapshot: PlantSnapshot,
    initial_feedstock: FeedstockObservation,
    resource_model: ResourceModel | None = None,
) -> ControllerStateEstimator:
    resource_model = resource_model or make_predictor_resource_model(cfg)
    predictor_preheater = PreheaterForwardModel(
        PreheaterForwardConfig(
            n_cells=int(getattr(cfg, "preheater_n_cells", 20)),
            tau_residence_s=float(getattr(cfg, "pre_tau_s", 985.0)),
            feed_delay_s=float(getattr(cfg, "pre_dead_s", 5.0)),
        )
    )
    if initial_snapshot.preheater_state is not None:
        predictor_preheater.load_state(initial_snapshot.preheater_state, feedstock=initial_feedstock)

    furnace_nominal = FurnaceDyn(
        FurnaceDynConfig(
            omega_ref=float(getattr(cfg, "omega_ref", 0.3218)),
            dead_s=float(getattr(cfg, "furnace_dead_s", 5.0)),
            tau1_s=float(getattr(cfg, "furnace_tau1_s", 0.223)),
            tau2_s=float(getattr(cfg, "furnace_tau2_s", 75.412)),
            dt_s=float(getattr(cfg, "dt_meas_s", 0.1)),
        )
    )
    if getattr(cfg, "furnace_init_mode", "warm") == "custom":
        from plant.python_model.furnace import furnace_outputs_from_omega
        omega_ref = float(getattr(cfg, "omega_ref", 0.3218))
        default_Tavg, default_Tstack, default_vstack = furnace_outputs_from_omega(omega_ref)
        furnace_nominal.initialize_outputs(
            float(getattr(cfg, "T_avg_init_C")) if getattr(cfg, "T_avg_init_C", None) is not None else float(getattr(cfg, "T_set_C", default_Tavg)),
            float(getattr(cfg, "T_stack_init_C")) if getattr(cfg, "T_stack_init_C", None) is not None else float(default_Tstack),
            float(getattr(cfg, "v_stack_init_mps")) if getattr(cfg, "v_stack_init_mps", None) is not None else float(default_vstack),
        )
    else:
        init_pre = initial_snapshot.preheater_state or predictor_preheater.state(time_s=0.0)
        diag = initial_snapshot.raw.get("preheater_diagnostics") if initial_snapshot.raw is not None else None
        furnace_feed = furnace_feed_from_preheater_output(
            time_s=0.0,
            omega_b=init_pre.omega_out,
            mdot_d_kgps=None if diag is None else getattr(diag, "dry_out_kgps", None),
            mdot_water_kgps=None if diag is None else getattr(diag, "water_out_kgps", None),
            mdot_wet_kgps=None if diag is None else getattr(diag, "wet_out_kgps", None),
            mdot_d_ref_kgps=furnace_nominal.cfg.mdot_d_ref_kgps,
        )
        furnace_nominal.initialize_from_omega(furnace_feed.omega_b, mdot_d_kgps=furnace_feed.mdot_d_kgps)

    disturbance_observer = FurnaceDisturbanceObserver()
    disturbance_observer.cfg = type(disturbance_observer.cfg)(
        alpha=float(getattr(cfg, "disturbance_observer_alpha", 0.05))
    )
    estimator = ControllerStateEstimator(
        predictor_preheater,
        furnace_predictor=furnace_nominal,
        disturbance_observer=disturbance_observer,
        resource_model=resource_model,
    )
    estimator.reset(initial_snapshot, feedstock=initial_feedstock)
    return estimator


def make_operator(cfg: Any, *, resource_model: ResourceModel | None = None):
    resource_model = resource_model or make_predictor_resource_model(cfg)
    mpc_cfg = NMPCConfig(
        dt_pred_s=float(getattr(cfg, "mpc_dt_s", 20.0)),
        horizon_s=float(getattr(cfg, "mpc_horizon_s", 600.0)),
        reoptimize_s=float(getattr(cfg, "nmpc_reoptimize_s", 60.0)),
        rollout_dt_s=float(getattr(cfg, "nmpc_rollout_dt_s", 5.0)),
        T_set_C=float(getattr(cfg, "T_set_C", getattr(cfg, "T_target_C", 873.0))),
        omega_ref=float(getattr(cfg, "omega_ref", 0.3218)),
        nominal_Tg_C=float(getattr(cfg, "nominal_Tg_C", 800.0)),
        nominal_vg_mps=float(getattr(cfg, "nominal_vg_mps", 12.0)),
        maxiter=int(getattr(cfg, "nmpc_maxiter", 20)),
    )
    if bool(getattr(cfg, "nmpc_async", False)):
        return AsyncNonlinearMPCController(
            mpc_cfg,
            resource_model=resource_model,
            stale_plan_timeout_s=float(getattr(cfg, "nmpc_async_stale_plan_timeout_s", 300.0)),
        )
    return NonlinearMPCController(mpc_cfg, resource_model=resource_model)


def make_executor(cfg: Any | None = None) -> ControlExecutor:
    return ControlExecutor()
