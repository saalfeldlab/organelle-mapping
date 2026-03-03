from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import gunpowder as gp
import torch
from pydantic import BaseModel, Field, ValidationError, model_validator

from organelle_mapping.config.transform import Transform
from organelle_mapping.loss import MaskedMSELoss, MaskedMultiLabelBCEwithLogits


class TargetConfig(BaseModel, ABC):
    """Base class for target output configurations."""

    type: str = Field(description="Type of target output")
    name: Optional[str] = Field(default=None, description="Optional name for this target")
    weight: float = Field(default=1.0, description="Weight for this target in the total loss")
    transforms: Sequence[Transform] = Field(description="Source transformations for this target")
    model_config = {"discriminator": "type"}

    @abstractmethod
    def create_loss_function(self) -> torch.nn.Module:
        """Create the appropriate loss function for this target."""
        ...

    @property
    def num_channels(self) -> int:
        """Total number of output channels for this target."""
        return sum(tt.num_channels for tt in self.transforms)

    @property
    def output_keys(self) -> tuple[gp.ArrayKey, ...]:
        """Get the output array key for this target."""
        return tuple(tt.output_key for tt in self.transforms)

    @property
    def mask_keys(self) -> tuple[gp.ArrayKey, ...]:
        """Get the mask array key for this target."""
        return tuple(tt.mask_key for tt in self.transforms)


class MultiLabelBCETarget(TargetConfig):
    """Target using multi-label binary cross-entropy loss."""

    type: Literal["multi_label_bce"] = "multi_label_bce"
    pos_weights: Optional[Sequence[float]] = None

    @model_validator(mode="after")
    def validate_pos_weights(self):
        """Validate pos_weights length if provided."""
        if self.pos_weights is not None and len(self.pos_weights) != self.num_channels:
            msg = f"pos_weights length ({len(self.pos_weights)}) must match number of channels ({self.num_channels})"
            raise ValidationError(msg)
        return self

    def create_loss_function(self) -> torch.nn.Module:
        """Create MaskedMultiLabelBCE loss with pos_weights."""
        return MaskedMultiLabelBCEwithLogits(pos_weight=self.pos_weights)


class MSETarget(TargetConfig):
    """Target using mean squared error loss."""

    type: Literal["mse"] = "mse"

    def create_loss_function(self) -> torch.nn.Module:
        """Create masked MSE loss."""
        return MaskedMSELoss()


# Discriminated union for automatic subclass selection
Target = Union[MultiLabelBCETarget, MSETarget]
