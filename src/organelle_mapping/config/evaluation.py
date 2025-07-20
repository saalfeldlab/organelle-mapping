from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import yaml
from pydantic import TypeAdapter

from organelle_mapping.config.run import RunConfig
from organelle_mapping.config.data import DataConfig
from organelle_mapping.config.models import Architecture
from organelle_mapping.config.utils import load_subconfig

class EvaluationConfig(BaseModel):
    """Main evaluation configuration."""
    
    # Core experiment configuration  
    experiment_run: RunConfig = Field(..., description="Training run configuration")
    # Architecture override
    eval_architecture: Optional[Architecture] = Field(
        default=None, 
        description="Override architecture config (defaults to experiment_run.architecture)"
    )
    
    # Model checkpoints
    checkpoints: List[str] = Field(
        ..., 
        min_items=1,
        description="List of checkpoint names to evaluate"
    )
    
    # Data configuration
    data: DataConfig = Field(
        ..., 
        description="Data config of files to be evaluated"
    )

    # Evaluation settings
    thresholds: List[float] = Field(
        default_factory=lambda: [i/100.0 for i in range(1, 100)],
        description="List of thresholds to evaluate (0.01 to 0.99)"
    )
    
    
    @field_validator("experiment_run", mode="before")
    @classmethod
    def load_run_config(cls, value) -> RunConfig:
        """Load RunConfig from file path or return as-is if already a RunConfig."""
        return load_subconfig(value, RunConfig)
    
    @field_validator("eval_architecture", mode="before")
    @classmethod
    def load_eval_architecture(cls, value) -> Optional[Architecture]:
        """Load Architecture from file path or return as-is if already an Architecture."""
        if value is None:
            return None
        return load_subconfig(value, Architecture)
    
    @field_validator("data", mode="before")
    @classmethod
    def load_data_config(cls, value) -> DataConfig:
        """Load DataConfig from file path or return as-is if already a DataConfig."""
        return load_subconfig(value, DataConfig)
    
    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v):
        """Ensure all thresholds are between 0 and 1."""
        for thresh in v:
            if not 0.0 <= thresh <= 1.0:
                raise ValueError(f"Threshold {thresh} must be between 0.0 and 1.0")
        return sorted(v)  # Sort for consistency
    
    @model_validator(mode="after")
    def set_architecture_default(self) -> "EvaluationConfig":
        """Set eval_architecture to experiment_run.architecture if not provided."""
        if self.eval_architecture is None:
            self.eval_architecture = self.experiment_run.architecture
        return self