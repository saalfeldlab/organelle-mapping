from pathlib import Path
from typing import Optional, Sequence

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationInfo, field_validator

from organelle_mapping.config.augmentations import AugmentationPipeline
from organelle_mapping.config.data import DataConfig
from organelle_mapping.config.models import Architecture
from organelle_mapping.config.checkpoint_edit import CheckpointEditConfig


def load_subconfig(value, target_cls, info: ValidationInfo):
    if isinstance(value, str):
        config_path = Path(value)

        # Try to get base_dir from validation context
        base_dir = info.context.get('base_dir') if info.context else "."

        # Resolve path relative to base_dir if provided
        if not config_path.is_absolute():
            config_path = Path(base_dir) / config_path

        with open(config_path) as config:
            return TypeAdapter(target_cls).validate_python(yaml.safe_load(config), context=info.context)

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
    finetuning: Optional[CheckpointEditConfig] = Field(
        default=None,
        description="Optional finetuning configuration for transfer learning"
    )


    @field_validator("augmentations", mode="before")
    @classmethod
    def load_augmentation_config(cls, value, info: ValidationInfo) -> AugmentationPipeline:
        return load_subconfig(value, AugmentationPipeline, info)

    @field_validator("data", mode="before")
    @classmethod
    def load_data_config(cls, value, info: ValidationInfo) -> DataConfig:
        return load_subconfig(value, DataConfig, info)

    @field_validator("architecture", mode="before")
    @classmethod
    def load_model_config(cls, value, info: ValidationInfo) -> Architecture:
        return load_subconfig(value, Architecture, info)

    @field_validator("finetuning", mode="before")
    @classmethod
    def load_finetuning_config(cls, value, info: ValidationInfo) -> Optional[CheckpointEditConfig]:
        if value is None:
            return None
        return load_subconfig(value, CheckpointEditConfig, info)

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

    @field_validator("finetuning", mode="after")
    @classmethod
    def validate_finetuning(cls, value: Optional[CheckpointEditConfig], info: ValidationInfo) -> Optional[CheckpointEditConfig]:
        if value is not None:
            # Check that target labels match architecture out_channels
            architecture = info.data.get("architecture")
            labels = info.data.get("labels")
            if architecture and hasattr(architecture, 'out_channels') and architecture.out_channels != len(labels):
                msg = (
                    f"Architecture out_channels ({architecture.out_channels}) must match "
                    f"the number of target labels ({len(labels)})."
                )
                raise ValueError(msg)
        return value
