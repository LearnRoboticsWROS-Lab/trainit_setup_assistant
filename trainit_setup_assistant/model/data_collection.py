"""The project's DATA COLLECTION section (TrainIt D0.8, ADR-0029 Decision 7, D0_8_DESIGN.md §7).

What the user configures for collecting data with this cell -- never how it is acquired or stored. TSA is the
control plane: this section is AUTHORED and VALIDATED here, GENERATED into the bundle as the recorder's
``trainit-recording/1.0`` file (``generator/data_collection.py``), and the recorder (``trainit_data_ros``) and the
canonical data engine (``trainit_data``) do the rest::

    data_collection:
      dataset: {id: bin_parts, description: "...", classification: internal}
      sensors: [{camera: bin_camera, streams: [color, depth]}]       # a perception.camera id; exactly one (D0.8)
      triggers:
        - {id: manual, kind: manual}
        - {id: every_2s, kind: interval, period_ms: 2000, max_captures: 500}
        - {id: on_glue, kind: event, process_channel: glue, edge: rising}
        - {id: before_grasp, kind: application_state, task: pick_one, step: pick, transition: enter,
           pick_attempt: pre_grasp}
      sync: {timeout_ms: 1000, tolerance_ms: 20}                      # optional: the recorder's defaults otherwise
      synthetic: {provider: isaac_replicator_bin, seed: 7, runs: 24, views: 2,
                  parameters: {objects_per_run: [4, 9], lights.dome_intensity: [150, 600]}}

The STORE location is not here (machine-local, ``TRAINIT_DATA_ROOT`` at launch): a bundle carries no machine path.
The model checks SHAPES only (strict keys, types); what refers to something else -- the camera id, the tasks and
steps, the process channels, the recorder's vocabulary -- is judged by ONE validator,
:func:`trainit_setup_assistant.generator.data_collection.data_collection_problems`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class _Strict(BaseModel):
    class Config:
        extra = 'forbid'
        validate_assignment = True


TRIGGER_KINDS = ('manual', 'interval', 'event', 'application_state')
CLASSIFICATIONS = ('internal', 'customer_restricted', 'public_approved')
STREAMS = ('color', 'depth')
# every field a trigger kind may carry (the rest must be absent); the recorder's exported vocabulary is the
# authority on the values (pinned in the catalog, checked by the validator)
TRIGGER_FIELDS = {'manual': (), 'interval': ('period_ms', 'max_captures'),
                  'event': ('process_channel', 'edge'),
                  'application_state': ('task', 'step', 'transition', 'pick_attempt')}


class DatasetRef(_Strict):
    """The dataset this cell's data is meant for (its identity is created in the store by ``trainit-data``)."""

    id: str
    description: Optional[str] = None
    classification: str = 'internal'

    @validator('classification')
    def _classification(cls, v):  # noqa: N805
        if v not in CLASSIFICATIONS:
            raise ValueError(f'R-D0-01: data collection classification {v!r} is not one of '
                             f'{", ".join(CLASSIFICATIONS)}')
        return v


class DataSensor(_Strict):
    camera: str
    streams: List[str] = Field(default_factory=lambda: list(STREAMS))

    @validator('streams')
    def _streams(cls, v):  # noqa: N805
        if not v or len(set(v)) != len(v) or any(s not in STREAMS for s in v):
            raise ValueError(f'R-D0-01: data collection streams {v!r}: one or both of {", ".join(STREAMS)}, '
                             'each once')
        return v


class DataTrigger(_Strict):
    id: str
    kind: str
    period_ms: Optional[int] = None
    max_captures: Optional[int] = None
    process_channel: Optional[str] = None
    edge: Optional[str] = None
    task: Optional[str] = None
    step: Optional[str] = None
    transition: Optional[str] = None
    pick_attempt: Optional[str] = None

    @validator('kind')
    def _kind(cls, v):  # noqa: N805
        if v not in TRIGGER_KINDS:
            raise ValueError(f'R-D0-01: trigger kind {v!r} is not one of {", ".join(TRIGGER_KINDS)}')
        return v

    def set_fields(self) -> Dict[str, Any]:
        """The kind-specific fields that are set."""
        return {name: getattr(self, name) for name in TRIGGER_FIELDS[self.kind]
                if getattr(self, name) is not None}

    def stray_fields(self) -> List[str]:
        own = set(TRIGGER_FIELDS[self.kind])
        return [name for name in ('period_ms', 'max_captures', 'process_channel', 'edge', 'task', 'step',
                                  'transition', 'pick_attempt')
                if name not in own and getattr(self, name) is not None]


class DataSync(_Strict):
    timeout_ms: Optional[int] = None
    tolerance_ms: Optional[float] = None


class SyntheticRecipe(_Strict):
    """The saved synthetic recipe: a provider id and the values of ITS declared parameters (dotted keys of the
    provider's configuration), validated against the provider's descriptor where the provider is installed."""

    provider: str
    seed: Optional[int] = None
    runs: Optional[int] = None
    views: Optional[int] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class DataCollectionSpec(_Strict):
    dataset: DatasetRef
    sensors: List[DataSensor] = Field(default_factory=list)
    triggers: List[DataTrigger] = Field(default_factory=list)
    sync: Optional[DataSync] = None
    synthetic: Optional[SyntheticRecipe] = None

    def records_live(self) -> bool:
        """The bundle gets a recording configuration (a camera and at least one trigger)."""
        return bool(self.sensors and self.triggers)


def data_collection_as_written(spec: DataCollectionSpec) -> dict:
    """The section as the file states it: unset optional fields absent (never ``null``)."""
    import json
    return json.loads(spec.json(exclude_none=True))
