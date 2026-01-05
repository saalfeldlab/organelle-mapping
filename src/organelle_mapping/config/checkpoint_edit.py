from pathlib import Path
from typing import Optional, Sequence
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from organelle_mapping.config.utils import resolve_path


class CheckpointEditConfig(BaseModel):
    """Configuration for checkpoint weight transfer between models.

    Supports transferring weights between models with different channel configurations
    using channel descriptors (e.g., 'mito_binary', 'er_lsd_0') to match channels.

    Path resolution:
    - source_experiment: resolved relative to the config file containing this config
    - source_checkpoint: resolved relative to source_experiment's directory
    """

    source_checkpoint: Path = Field(
        description="Path to source checkpoint file (relative to source_experiment directory)"
    )
    source_experiment: Path = Field(
        description="Path to source experiment run.yaml config file"
    )
    channel_mapping: Optional[dict[str, Optional[str]]] = Field(
        default=None,
        description="""Mapping from target channel descriptors to source channel descriptors.
        Keys are target descriptors, values are source descriptors (or None to initialize fresh).
        If not provided, auto-matches by descriptor name."""
    )
    heads_keys: Optional[Sequence[str]] = Field(
        default=None,
        description="State dict keys containing output heads. If None, auto-detects 'final_conv' keys"
    )

    @field_validator('source_experiment', mode='before')
    @classmethod
    def resolve_source_experiment(cls, value, info: ValidationInfo) -> Path:
        """Resolve source_experiment relative to the config's base_dir."""
        return Path(resolve_path(value, info))

    @model_validator(mode='after')
    def resolve_checkpoint_and_validate(self) -> "CheckpointEditConfig":
        """Resolve source_checkpoint and validate paths exist."""
        # Resolve source_checkpoint relative to source_experiment's directory
        if not self.source_checkpoint.is_absolute():
            self.source_checkpoint = self.source_experiment.parent / self.source_checkpoint

        # Validate existence after resolution
        if not self.source_experiment.exists():
            raise ValueError(f"Source experiment config does not exist: {self.source_experiment}")
        if not self.source_checkpoint.exists():
            raise ValueError(f"Source checkpoint does not exist: {self.source_checkpoint}")

        return self