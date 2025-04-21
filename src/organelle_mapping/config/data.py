from typing import Dict, List

from pydantic import BaseModel, Field


class EMConfig(BaseModel):
    data: str = Field(..., description="Path to the EM data file")
    group: str = Field(..., description="Group within the EM data file")
    contrast: List[int] = Field(
        ...,
        description="Contrast range for the EM data (min and max values)",
        example=[65, 193],
    )


class LabelsConfig(BaseModel):
    data: str = Field(..., description="Path to the labels data file")
    group: str = Field(..., description="Group within the labels data file")
    crops: List[str] = Field(
        default_factory=list, description="List of crop identifiers"
    )


class DatasetConfig(BaseModel):
    em: EMConfig = Field(..., description="EM data configuration")
    labels: LabelsConfig = Field(..., description="Labels configuration")


class DataConfig(BaseModel):
    datasets: Dict[str, DatasetConfig] = Field(
        ..., description="Mapping of dataset names to their configurations"
    )
