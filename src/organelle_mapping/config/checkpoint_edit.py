from pathlib import Path
from typing import Optional, Sequence
from pydantic import BaseModel, Field, field_validator


class CheckpointEditConfig(BaseModel):
    """Configuration for checkpoint weight transfer between models.

    Supports transferring weights between models with different channel configurations
    using channel descriptors (e.g., 'mito_binary', 'er_lsd_0') to match channels.
    """

    source_checkpoint: Path = Field(
        description="Path to source checkpoint file"
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

    @field_validator('source_checkpoint')
    @classmethod
    def validate_checkpoint_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Source checkpoint does not exist: {v}")
        return v

    @field_validator('source_experiment')
    @classmethod
    def validate_experiment_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Source experiment config does not exist: {v}")
        return v