from __future__ import annotations

from domain.types import PlantInitialState, PlantSnapshot, PlantStepInput


class HardwarePlantBackend:
    """Placeholder for future hardware/real-plant backend.

    Future implementation should translate PlantStepInput into PLC/DAQ commands
    and return a PlantSnapshot assembled from sensors and actuator feedback.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Hardware plant backend is not implemented in this cleanup pass.")

    def reset(self, initial: PlantInitialState | None = None) -> PlantSnapshot:
        raise NotImplementedError

    def step(self, step_input: PlantStepInput) -> PlantSnapshot:
        raise NotImplementedError

    def observe(self) -> PlantSnapshot | None:
        raise NotImplementedError
