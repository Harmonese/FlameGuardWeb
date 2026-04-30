from __future__ import annotations

from domain.types import PlantInitialState, PlantSnapshot, PlantStepInput


class ComsolPlantBackend:
    """Placeholder for future COMSOL co-simulation backend.

    Future implementation should translate PlantStepInput into COMSOL boundary
    conditions and return a PlantSnapshot.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("COMSOL plant backend is not implemented in this cleanup pass.")

    def reset(self, initial: PlantInitialState | None = None) -> PlantSnapshot:
        raise NotImplementedError

    def step(self, step_input: PlantStepInput) -> PlantSnapshot:
        raise NotImplementedError

    def observe(self) -> PlantSnapshot | None:
        raise NotImplementedError
