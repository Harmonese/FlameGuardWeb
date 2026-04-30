from __future__ import annotations

from typing import Protocol

from .types import (
    ActuatorCommand,
    ControlSetpoint,
    FeedstockObservation,
    MPCDecision,
    OperatorContext,
    PredictorBundle,
    PlantInitialState,
    PlantSnapshot,
    PlantStepInput,
    StateEstimate,
)


class PlantBackend(Protocol):
    def reset(self, initial: PlantInitialState | None = None) -> PlantSnapshot: ...
    def step(self, step_input: PlantStepInput) -> PlantSnapshot: ...
    def observe(self) -> PlantSnapshot | None: ...


class StateEstimator(Protocol):
    def reset(self, snapshot: PlantSnapshot, *, feedstock: FeedstockObservation | None = None) -> StateEstimate: ...
    def update(self, snapshot: PlantSnapshot, previous_command: ActuatorCommand, feedstock: FeedstockObservation, *, dt_s: float | None = None) -> StateEstimate: ...
    def get_predictor_bundle(self) -> PredictorBundle: ...


class Operator(Protocol):
    def initialize(self, *, Tg_ref_C: float, vg_ref_mps: float, time_s: float = 0.0) -> None: ...
    def step_context(self, context: OperatorContext) -> MPCDecision: ...


class Executor(Protocol):
    def translate_setpoint(self, setpoint: ControlSetpoint) -> ActuatorCommand: ...
    def step(self, setpoint: ControlSetpoint, snapshot: PlantSnapshot, estimate: StateEstimate, dt_s: float) -> ActuatorCommand: ...


class FeedstockPreviewProvider(Protocol):
    def get(self, time_s: float, *, horizon_s: float, dt_s: float) -> list[FeedstockObservation]: ...
