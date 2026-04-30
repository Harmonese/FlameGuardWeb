from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Tuple


@dataclass(frozen=True)
class EquivalentProperties:
    """Equivalent material properties carried by a waste parcel or cell.

    These are model-facing properties. External feed characterization should
    enter the system as FeedstockObservation and then be converted to this
    compact form by the plant/predictor model that consumes it.
    """

    omega0: float
    tref_min: float
    slope_min_per_c: float
    bulk_density_kg_m3: float | None = None
    wet_mass_flow_kgps: float | None = None


@dataclass(frozen=True)
class ResourceBoundary:
    """Effective flue-gas resource boundary used by controller/executor."""

    T_stack_cap_C: float
    v_stack_cap_mps: float


@dataclass(frozen=True)
class FurnaceObservation:
    time_s: float
    T_avg_C: float
    T_stack_C: float
    v_stack_mps: float


@dataclass(frozen=True)
class FeedstockObservation:
    """Characterized incoming waste stream seen by the plant/controller.

    This protocol intentionally does not expose fixed waste categories. Upstream
    perception/manual/lab modules may infer these fields however they want.
    """

    time_s: float
    moisture_wb: float
    drying_time_ref_min: float
    drying_sensitivity_min_per_C: float
    bulk_density_kg_m3: float | None = None
    wet_mass_flow_kgps: float | None = None
    source: str = "unknown"
    confidence: float = 1.0
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActuatorCommand:
    time_s: float
    Tg_cmd_C: float
    vg_cmd_mps: float
    heater_enable: bool
    Q_aux_heat_kW: float
    aux_resource_required: bool
    aux_heat_required: bool
    aux_circulation_required: bool
    mdot_preheater_kgps: float = 0.0
    mdot_stack_cap_kgps: float = 0.0
    mdot_aux_flow_kgps: float = 0.0
    fan_circulation_power_kW: float = 0.0
    recovery_guard_active: bool = False


@dataclass(frozen=True)
class PreheaterCellState:
    """One axial cell in a preheater state estimate or proxy model."""

    index: int
    z_center_m: float
    residence_left_s: float
    omega: float
    T_solid_C: float
    omega0: float
    tref_min: float
    slope_min_per_c: float
    dry_mass_kg: float = 0.0
    water_mass_kg: float = 0.0
    bulk_density_kg_m3: float | None = None


@dataclass(frozen=True)
class PreheaterState:
    time_s: float
    cells: Tuple[PreheaterCellState, ...]
    omega_out: float
    T_solid_out_C: float
    Tg_profile_C: Tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PreheaterDiagnostics:
    """Per-step heat, mass, and gas-path diagnostics for the preheater model."""

    time_s: float = 0.0
    Tg_in_C: float = float("nan")
    Tg_out_C: float = float("nan")
    vg_mps: float = float("nan")
    mdot_gas_kgps: float = float("nan")
    U_eff_W_m2K: float = float("nan")
    Q_gas_to_solid_kW: float = 0.0
    Q_sensible_kW: float = 0.0
    Q_latent_kW: float = 0.0
    heat_balance_residual_kW: float = 0.0
    water_evap_kgps: float = 0.0
    dry_out_kgps: float = 0.0
    water_out_kgps: float = 0.0
    wet_out_kgps: float = 0.0
    inventory_dry_kg: float = 0.0
    inventory_water_kg: float = 0.0
    inventory_total_kg: float = 0.0
    Tg_cell_min_C: float = float("nan")
    Tg_cell_max_C: float = float("nan")


@dataclass(frozen=True)
class PreheaterOutput:
    """Lightweight preheater output usable when full state is unavailable."""

    time_s: float
    omega_out: float | None = None
    T_solid_out_C: float | None = None
    wet_out_kgps: float | None = None
    dry_out_kgps: float | None = None
    water_out_kgps: float | None = None
    Tg_out_C: float | None = None




@dataclass(frozen=True)
class FurnaceFeed:
    """Mass-flow based feed entering the furnace.

    omega_b is furnace-inlet wet-basis moisture.  mdot_d_kgps is the
    engineering primary load variable.  rd is retained as a derived COMSOL
    surrogate coordinate: rd = mdot_d_kgps / mdot_d_ref_kgps.
    """

    time_s: float
    omega_b: float
    mdot_d_kgps: float
    mdot_water_kgps: float
    mdot_wet_kgps: float
    rd: float

@dataclass(frozen=True)
class StackResourceMeasurement:
    """Natural stack resource measured or inferred by a plant backend."""

    time_s: float
    T_stack_available_C: float
    v_stack_available_mps: float
    mdot_stack_available_kgps: float | None = None


@dataclass(frozen=True)
class ActuatorFeedback:
    """Actual applied actuator state, if known from simulation or hardware."""

    time_s: float
    Tg_actual_C: float | None = None
    vg_actual_mps: float | None = None
    heater_enabled_actual: bool | None = None
    fan_speed_actual: float | None = None
    valve_position_actual: float | None = None
    command_applied: bool = True


@dataclass(frozen=True)
class PlantHealth:
    ok: bool
    status: str = "ok"
    stale: bool = False
    comms_ok: bool = True
    sensors_ok: bool = True
    actuators_ok: bool = True
    message: str = ""


@dataclass(frozen=True)
class PlantInitialState:
    time_s: float = 0.0
    preheater_state: PreheaterState | None = None
    furnace_obs: FurnaceObservation | None = None


@dataclass(frozen=True)
class PlantStepInput:
    time_s: float
    dt_s: float
    command: ActuatorCommand
    feedstock: FeedstockObservation


@dataclass(frozen=True)
class PlantSnapshot:
    time_s: float
    furnace_obs: FurnaceObservation
    preheater_output: PreheaterOutput | None = None
    preheater_state: PreheaterState | None = None
    stack_resource: StackResourceMeasurement | None = None
    actuator_feedback: ActuatorFeedback | None = None
    health: PlantHealth | None = None
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StateEstimate:
    time_s: float
    furnace_obs: FurnaceObservation
    preheater_state_est: PreheaterState
    stack_resource_est: StackResourceMeasurement | None = None
    disturbance_est_Tavg_C: float = 0.0
    disturbance_est_Tstack_C: float = 0.0
    disturbance_est_vstack_mps: float = 0.0
    confidence: float = 1.0
    source: str = "estimator"




class PreheaterPredictorProtocol(Protocol):
    def clone(self) -> "PreheaterPredictorProtocol": ...
    def load_state(self, state: PreheaterState, *, feedstock: FeedstockObservation | None = None) -> "PreheaterPredictorProtocol": ...
    def step_fast(self, feed, Tg_in_C: float, vg_mps: float, dt_s: float) -> PreheaterOutput: ...
    def output(self, time_s: float | None = None) -> PreheaterOutput: ...
    def state(self, time_s: float | None = None) -> PreheaterState: ...


class FurnacePredictorProtocol(Protocol):
    def clone(self) -> "FurnacePredictorProtocol": ...
    def step(self, omega_in: float, **kwargs): ...


class ResourcePredictorProtocol(Protocol):
    def from_observation(self, obs: FurnaceObservation): ...


@dataclass(frozen=True)
class PredictorBundle:
    """Controller-owned predictor objects used by the operator.

    The concrete classes live under controller/predictor, but this bundle is
    typed by structural Protocols so domain remains independent of controller
    implementation modules while still enforcing the methods the operator uses.
    """

    preheater: PreheaterPredictorProtocol
    furnace: FurnacePredictorProtocol
    resource_model: ResourcePredictorProtocol | None = None


@dataclass(frozen=True)
class OperatorContext:
    """Complete operator input for one control update."""

    estimate: StateEstimate
    predictors: PredictorBundle
    feedstock: FeedstockObservation
    resource: ResourceBoundary
    previous_command: ActuatorCommand | None = None
    feed_preview: Any | None = None


@dataclass(frozen=True)
class ControlSetpoint:
    time_s: float
    Tg_ref_C: float
    vg_ref_mps: float
    source: str = "operator"
    omega_target: float = 0.3218
    omega_reachable: float = 0.3218
    power_kW: float = 0.0
    Qreq_kW: float = 0.0
    Qsup_kW: float = 0.0
    mdot_stack_cap_kgps: float = float("inf")
    mdot_preheater_kgps: float = 0.0
    T_stack_available_C: float = 1100.0
    v_stack_available_mps: float = 18.0
    recovery_guard_requested: bool = False
    recovery_guard_reason: str = ""


@dataclass(frozen=True)
class MPCDecision:
    time_s: float
    Tg_ref_C: float
    vg_ref_mps: float
    omega_target: float
    omega_reachable: float
    predicted_Tavg_C: float
    predicted_omega_out: float
    cost: float
    feasible: bool
    source: str = "operator"
    safety_reachable: bool = True
    predicted_min_Tavg_C: float = float("nan")
    predicted_max_Tavg_C: float = float("nan")
    omega_max_for_safety: float = float("nan")
    safety_margin_C: float = float("nan")
    note: str = ""
