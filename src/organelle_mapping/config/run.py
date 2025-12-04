from pathlib import Path
from typing import Sequence

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationInfo, field_validator, model_validator

from organelle_mapping.config.augmentations import AugmentationPipeline
from organelle_mapping.config.data import DataConfig
from organelle_mapping.config.models import Architecture
from organelle_mapping.config.target import Target


def load_subconfig(value, target_cls):
    if isinstance(value, str):
        config = open(Path(value))
        return TypeAdapter(target_cls).validate_python(yaml.safe_load(config))

    return value


class RunConfig(BaseModel):
    iterations: int
    targets: Sequence[Target] = Field(min_items=1, description="Target outputs with loss functions")
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

    @property
    def total_channels(self) -> int:
        """Calculate total output channels across all targets."""
        return sum(
            sum(target_transform.num_channels for target_transform in target.transforms)
            for target in self.targets
        )

    @model_validator(mode="after")
    def validate_architecture_channels(self):
        """Validate that architecture output channels match total transform channels."""
        if self.architecture.out_channels != self.total_channels:
            msg = f"Architecture output channels ({self.architecture.out_channels}) must match total transform channels ({self.total_channels})"
            raise ValueError(msg)
        return self

    # TODO: Add weight normalization validator
    # Step 1: Normalize within each target (sum to 1.0)
    # Step 2: Scale across targets (total sum to 1.0) for learning rate consistency
