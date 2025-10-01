from pathlib import Path
from typing import Optional, Sequence
from pydantic import BaseModel, Field, field_validator

class CheckpointEditConfig(BaseModel):
    source_checkpoint: Path = Field(
        description="Path to source checkpoint file"
    )
    source_experiment: Path = Field(
        description="Path to source experiment run.yaml config file"
    )
    target_labels: Optional[Sequence[str]] = Field(
        default=None,
        description="Target labels for the output model. If None, uses all labels as-is"
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