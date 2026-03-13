from abc import ABC, abstractmethod
from typing import Annotated, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from organelle_mapping.config.data import DataConfig
from organelle_mapping.config.models import Architecture
from organelle_mapping.config.run import RunConfig
from organelle_mapping.config.utils import load_subconfig


class PostprocessingConfig(BaseModel, ABC):
    """Base class for all postprocessing configurations."""

    type: str = Field(description="Type of postprocessing to apply")

    @abstractmethod
    def apply(self, pred_channel: np.ndarray) -> tuple[dict, np.ndarray]:
        """Apply postprocessing to a prediction channel.

        Returns:
            (params, processed_prediction) where params is a dict of
            DB column name → value (e.g. {"threshold": 0.5}).
        """
        ...


class ThresholdPostprocessing(PostprocessingConfig):
    """Threshold-based binarization."""

    type: Literal["threshold"] = "threshold"
    threshold: float = Field(..., description="Threshold value for binarization")

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v):
        """Ensure threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            msg = f"Threshold {v} must be between 0.0 and 1.0"
            raise ValueError(msg)
        return v

    def apply(self, pred_channel: np.ndarray) -> tuple[dict, np.ndarray]:
        """Binarize predictions by thresholding."""
        return {"threshold": self.threshold}, (pred_channel > self.threshold).astype(np.uint8)


Postprocessing = Annotated[Union[ThresholdPostprocessing], Field(discriminator="type")]


class EvalChannelConfig(BaseModel):
    """Configuration for evaluating a single model output channel."""

    channel: str = Field(..., description="Channel descriptor (e.g. 'mito_binary', 'er_binary')")
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Metrics to compute. Inherits from default_metrics if not specified.",
    )
    postprocessing: Optional[List[Postprocessing]] = Field(
        default=None,
        description="Postprocessing configs. Inherits from default_postprocessing if not specified.",
    )


class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""

    # Core experiment configuration
    experiment_run: RunConfig = Field(..., description="Training run configuration")
    # Architecture override
    eval_architecture: Optional[Architecture] = Field(
        default=None, description="Override architecture config (defaults to experiment_run.architecture)"
    )

    # Model checkpoints
    checkpoints: List[str] = Field(..., min_length=1, description="List of checkpoint names to evaluate")

    # Data configuration
    data: DataConfig = Field(..., description="Data config of files to be evaluated")

    # Default metrics (inherited by channels that don't specify their own)
    default_metrics: Optional[List[str]] = Field(
        default=None,
        description="Default metrics for channels that don't specify their own.",
    )

    # Default postprocessing (inherited by channels that don't specify their own)
    default_postprocessing: Optional[List[Postprocessing]] = Field(
        default=None,
        description="Default postprocessing for channels that don't specify their own.",
    )

    # Per-channel evaluation configuration (required)
    eval_channels: List[EvalChannelConfig] = Field(
        ..., min_length=1, description="Channels to evaluate with per-channel metrics and postprocessing."
    )

    # Database URL for storing results
    db_url: Optional[str] = Field(
        default=None,
        description="SQLAlchemy database URL (e.g. 'sqlite:///results.db' or 'postgresql://user:pass@host/db')",
    )

    @field_validator("experiment_run", mode="before")
    @classmethod
    def load_run_config(cls, value, info: ValidationInfo) -> RunConfig:
        """Load RunConfig from file path or return as-is if already a RunConfig."""
        return load_subconfig(value, RunConfig, info)

    @field_validator("eval_architecture", mode="before")
    @classmethod
    def load_eval_architecture(cls, value, info: ValidationInfo) -> Optional[Architecture]:
        """Load Architecture from file path or return as-is if already an Architecture."""
        if value is None:
            return None
        return load_subconfig(value, Architecture, info)

    @field_validator("data", mode="before")
    @classmethod
    def load_data_config(cls, value, info: ValidationInfo) -> DataConfig:
        """Load DataConfig from file path or return as-is if already a DataConfig."""
        return load_subconfig(value, DataConfig, info)

    @model_validator(mode="after")
    def set_architecture_default(self) -> "EvaluationConfig":
        """Set eval_architecture to experiment_run.architecture if not provided."""
        if self.eval_architecture is None:
            self.eval_architecture = self.experiment_run.architecture
        return self

    @model_validator(mode="after")
    def validate_and_inherit_defaults(self) -> "EvaluationConfig":
        """Validate channel descriptors and apply metrics/postprocessing inheritance."""
        valid_descriptors = set(self.experiment_run.channel_descriptors)
        for ec in self.eval_channels:
            if ec.channel not in valid_descriptors:
                msg = f"Invalid eval channel '{ec.channel}'. Valid: {sorted(valid_descriptors)}"
                raise ValueError(msg)
            if ec.metrics is None:
                if self.default_metrics is None:
                    msg = f"Channel '{ec.channel}' has no metrics and no default_metrics is set."
                    raise ValueError(msg)
                ec.metrics = list(self.default_metrics)
            if ec.postprocessing is None:
                if self.default_postprocessing is None:
                    msg = (
                        f"Channel '{ec.channel}' has no postprocessing and no default_postprocessing is set."
                    )
                    raise ValueError(msg)
                ec.postprocessing = list(self.default_postprocessing)
        return self
