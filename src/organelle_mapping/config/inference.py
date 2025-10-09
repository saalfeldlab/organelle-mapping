
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from organelle_mapping.config.models import Architecture
from organelle_mapping.config.run import load_subconfig

class RoiConfig(BaseModel):
    """
    Configuration for a region of interest (ROI) in world coordinates
    """
    start: tuple[int, int, int]
    end: tuple[int, int, int]

class ChannelOutput(BaseModel):
    """
    Configuration for a single output channel or channel range
    """
    name: str = Field(description="Name of the output dataset")
    channels: str = Field(description="Channel indices (e.g., '0' or '0-3')")

class InputConfig(BaseModel):
    """
    Configuration for input data
    """
    container: str = Field(description="Path to input zarr container")
    dataset: str = Field(description="Dataset name in the container")
    voxel_size: tuple[int, int, int] = Field(description="Voxel size in nm")
    min_raw: float = Field(default=0, description="Minimum raw value for normalization")
    max_raw: float = Field(default=255, description="Maximum raw value for normalization")

class OutputConfig(BaseModel):
    """
    Configuration for the output of the inference process.
    """
    container: str = Field(description="Path to output zarr container")
    dataset: str = Field(description="Base dataset name in the container")
    outputs: list[ChannelOutput] = Field(description="List of output channels to generate")
    roi: Optional[RoiConfig] = None

class InferenceConfig(BaseModel):
    """
    Complete configuration for inference
    """
    checkpoint: str = Field(description="Path to model checkpoint")
    architecture: Architecture = Field(description="Model architecture configuration")
    input_data: InputConfig = Field(description="Input data configuration")
    output_data: OutputConfig = Field(description="Output data configuration")
    
    @field_validator("architecture", mode="before")
    @classmethod
    def load_architecture_config(cls, value, info: ValidationInfo) -> Architecture:
        if isinstance(value, str):
            
            
        return load_subconfig(value, Architecture, info)
    