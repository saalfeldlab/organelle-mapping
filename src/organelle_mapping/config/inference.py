from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from organelle_mapping.config.models import Architecture
from organelle_mapping.config.run import RunConfig
from organelle_mapping.config.target import Target
from organelle_mapping.config.utils import get_base_dir, load_subconfig, resolve_path


class RoiConfig(BaseModel):
    """Configuration for a region of interest (ROI) in world coordinates."""

    start: tuple[int, int, int]
    end: tuple[int, int, int]


class ChannelOutput(BaseModel):
    """Configuration for a single output channel or channel range."""

    name: str = Field(description="Name of the output dataset")
    channels: str = Field(description="Channel indices (e.g., '0' or '0-3')")


class InputConfig(BaseModel):
    """Configuration for input data."""

    container: str = Field(description="Path to input zarr container")
    dataset: str = Field(description="Dataset name in the container (for OME-Zarr: multiscale group path)")
    scale: Optional[str] = Field(
        default=None, description="Scale level name for OME-Zarr (e.g., 's0'). If None, uses regular zarr open_ds"
    )
    voxel_size: tuple[float, float, float] = Field(description="Voxel size in nm")
    min_raw: float = Field(default=0, description="Minimum raw value for normalization")
    max_raw: float = Field(default=255, description="Maximum raw value for normalization")


class OutputConfig(BaseModel):
    """Configuration for the output of the inference process."""

    container: str = Field(description="Path to output zarr container (relative paths resolved to config directory)")
    dataset: str = Field(description="Base dataset name in the container")
    outputs: list[ChannelOutput] = Field(description="List of output channels to generate")
    roi: Optional[RoiConfig] = None

    @field_validator("container", mode="before")
    @classmethod
    def resolve_container(cls, value, info: ValidationInfo) -> str:
        """Resolve relative container path to the config file's directory."""
        return resolve_path(value, info)


class InferenceConfig(BaseModel):
    """Complete configuration for inference.

    Path resolution:
    - run_config: resolved relative to the inference config file
    - checkpoint: resolved relative to run_config's directory
    """

    checkpoint: str = Field(description="Path to model checkpoint (relative to run_config directory)")
    run_config: RunConfig = Field(description="Run configuration for targets (path to run.yaml or inline)")
    architecture: Optional[Architecture] = Field(
        default=None, description="Model architecture configuration (defaults to run_config.architecture)"
    )
    input_data: InputConfig = Field(description="Input data configuration")
    output_data: OutputConfig = Field(description="Output data configuration")

    @field_validator("run_config", mode="before")
    @classmethod
    def load_run_config(cls, value, info: ValidationInfo) -> RunConfig:
        return load_subconfig(value, RunConfig, info)

    @field_validator("architecture", mode="before")
    @classmethod
    def load_architecture_config(cls, value, info: ValidationInfo) -> Optional[Architecture]:
        if value is None:
            return None
        return load_subconfig(value, Architecture, info)

    @model_validator(mode="before")
    @classmethod
    def resolve_checkpoint(cls, data, info: ValidationInfo):
        """Resolve checkpoint path relative to run_config's directory."""
        if not isinstance(data, dict):
            return data
        checkpoint = data.get("checkpoint")
        run_config_path = data.get("run_config")
        if isinstance(checkpoint, str) and isinstance(run_config_path, str):
            # Resolve run_config path first
            rc_path = Path(run_config_path)
            if not rc_path.is_absolute():
                rc_path = get_base_dir(info) / rc_path
            # Resolve checkpoint relative to run_config's directory
            cp_path = Path(checkpoint)
            if not cp_path.is_absolute():
                resolved = rc_path.parent / checkpoint
                # Normalize path if it exists, otherwise keep as-is
                data["checkpoint"] = str(resolved.resolve() if resolved.exists() else resolved)
        return data

    @model_validator(mode="after")
    def set_default_architecture(self) -> "InferenceConfig":
        """Default architecture to run_config.architecture if not provided."""
        if self.architecture is None:
            self.architecture = self.run_config.architecture
        return self

    @property
    def targets(self) -> Sequence[Target]:
        """Get targets from run config."""
        return self.run_config.targets
