from pathlib import Path
from typing import Sequence

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationInfo, field_validator

from organelle_mapping.config.augmentations import AugmentationPipeline
from organelle_mapping.config.data import DataConfig
from organelle_mapping.config.models import Architecture


def load_subconfig(value, target_cls):
    if isinstance(value, str):
        config = open(Path(value))
        return TypeAdapter(target_cls).validate_python(yaml.safe_load(config))

    return value


class RunConfig(BaseModel):
    iterations: int
    labels: Sequence[str] = Field(min_items=1)
    label_weights: Sequence[float] = Field(default=())
    sampling: dict[str, int] = Field(
        default_factory=lambda: {"x": 8, "y": 8, "z": 8},
        description="Sampling rates for x, y, and z dimensions",
    )
    architecture: Architecture
    data: DataConfig
    augmentations: AugmentationPipeline
    precache_size: int = 48
    precache_workers: int = 20
    lr: float = 5e-5
    log_frequency: int = 20
    checkpoint_frequency: int = 2000
    batch_size: int = 1
    min_valid_fraction: float = Field(
        default=0.0,
        description="Minimum fraction of valid (non-unknown) area required in a sample to avoid rejection",
        ge=0.0,
        le=1.0,
    )

    @field_validator("augmentations", mode="before")
    @classmethod
    def load_augmentation_config(cls, value) -> AugmentationPipeline:
        return load_subconfig(value, AugmentationPipeline)

    @field_validator("data", mode="before")
    @classmethod
    def load_data_config(cls, value) -> DataConfig:
        return load_subconfig(value, DataConfig)

    @field_validator("architecture", mode="before")
    @classmethod
    def load_model_config(cls, value) -> Architecture:
        return load_subconfig(value, Architecture)

    @field_validator("label_weights", mode="after")
    @classmethod
    def normalize_weights(cls, value: Sequence[float], info: ValidationInfo) -> Sequence[float]:
        if len(value) == 0:
            value = [1.0] * len(info.data["labels"])
        if len(value) != len(info.data["labels"]):
            msg = (
                f"Length of label_weights ({len(value)}) does not match the number of labels "
                f"({len(info.data['labels'])})."
            )
            raise ValueError(msg)
        normalizer = sum(value)
        return [lw / normalizer for lw in value]
