from pathlib import Path
from typing import Optional, Sequence

import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationInfo, field_validator, model_validator

from organelle_mapping.config.augmentations import AugmentationPipeline
from organelle_mapping.config.data import DataConfig
from organelle_mapping.config.models import Architecture
from organelle_mapping.config.target import Target
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
    snapshot_frequency: int = 2000
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

    @property
    def total_channels(self) -> int:
        """Calculate total output channels across all targets."""
        return sum(
            sum(target_transform.num_channels for target_transform in target.transforms)
            for target in self.targets
        )

    @property
    def channel_descriptors(self) -> list[str]:
        """Get ordered list of channel descriptors for all output channels.

        Returns:
            List of channel descriptor strings matching the model output order.
        """
        descriptors = []
        for target in self.targets:
            for transform in target.transforms:
                descriptors.extend(transform.channel_descriptors)
        return descriptors

    @model_validator(mode="after")
    def validate_architecture_channels(self):
        """Validate that architecture output channels match total transform channels."""
        if self.architecture.out_channels != self.total_channels:
            msg = f"Architecture output channels ({self.architecture.out_channels}) must match total transform channels ({self.total_channels})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_finetuning_channel_mapping(self):
        """Validate that finetuning channel_mapping keys match channel_descriptors."""
        if self.finetuning is not None and self.finetuning.channel_mapping is not None:
            mapping_keys = set(self.finetuning.channel_mapping.keys())
            expected_keys = set(self.channel_descriptors)
            if mapping_keys != expected_keys:
                missing = expected_keys - mapping_keys
                extra = mapping_keys - expected_keys
                msg_parts = ["Finetuning channel_mapping keys must match run's channel_descriptors."]
                if missing:
                    msg_parts.append(f"Missing: {sorted(missing)}")
                if extra:
                    msg_parts.append(f"Extra: {sorted(extra)}")
                raise ValueError(" ".join(msg_parts))
        return self

    # TODO: Add weight normalization validator
    # Step 1: Normalize within each target (sum to 1.0)
    # Step 2: Scale across targets (total sum to 1.0) for learning rate consistency

