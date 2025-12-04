from abc import ABC, abstractmethod
from typing import Literal, Optional, Annotated, Union

import torch
import gunpowder as gp
import corditea
from pydantic import BaseModel, Field, BeforeValidator, model_validator


def validate_activation_string(value: str) -> str:
    """Validator for torch.nn activation class names"""
    try:
        activation_class = getattr(torch.nn, value)
    except AttributeError:
        msg = f"torch.nn.{value} does not exist"
        raise ValueError(msg)

    if not (isinstance(activation_class, type) and
            issubclass(activation_class, torch.nn.Module)):
        msg = f"torch.nn.{value} is not a PyTorch module class"
        raise ValueError(msg)

    try:
        activation_class()
    except Exception as e:
        msg = f"Cannot instantiate torch.nn.{value}(): {e}"
        raise ValueError(msg)

    return value

ActivationStr = Annotated[str, BeforeValidator(validate_activation_string)]

class TransformConfig(BaseModel, ABC):
    """Base class for all output transform configurations."""

    type: str = Field(description="Type of transform to apply to the output")
    source: str = Field(description="Source label name (e.g., 'mito', 'er')")
    activation: ActivationStr = Field(description="Activation function to use during training")
    inference_activation: Optional[ActivationStr] = Field(
        default=None,
        description="Activation function to use during inference. If not specified, uses 'activation'"
    )
    model_config = {"discriminator": "type"}

    @model_validator(mode="after")
    def set_inference_activation_default(self):
        """Set inference_activation to activation if not specified."""
        if self.inference_activation is None:
            self.inference_activation = self.activation
        return self

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of output channels this transform produces."""
        ...
    @property
    @abstractmethod
    def mask_key(self) -> gp.ArrayKey:
        """Array Key for output mask"""
        ...
    @property
    @abstractmethod
    def output_key(self) -> gp.ArrayKey:
        """Array Key for output"""
        ...
        
    @abstractmethod
    def instantiate(self, source_key: gp.ArrayKey, source_mask_key: gp.ArrayKey) -> None | tuple[gp.BatchFilter]:
        """Get the pipeline nodes needed for this transform.

        Returns:
            Tuple of nodes to add to pipeline
        """
        ...

    def adjust_max_extent(self, max_extent: gp.Coordinate) -> gp.Coordinate:
        """Adjust maximum extent to account for transform's context requirements.

        Args:
            max_extent: Current maximum extent

        Returns:
            Adjusted maximum extent
        """
        return max_extent

    @property
    def channel_descriptors(self) -> list[str]:
        """Generate descriptors for each output channel.

        Returns:
            List of channel descriptor strings, one per output channel.
            Format: "{source}_{type}" for single-channel transforms,
                    "{source}_{type}_{index}" for multi-channel transforms.
        """
        if self.num_channels == 1:
            return [f"{self.source}_{self.type}"]
        else:
            return [f"{self.source}_{self.type}_{i}" for i in range(self.num_channels)]


class BinaryConfig(TransformConfig):
    """Configuration for binary segmentation output transform."""

    type: Literal["binary"] = "binary"
    activation: ActivationStr = "Identity"
    inference_activation: Optional[ActivationStr] = "Sigmoid"

    @property
    def num_channels(self) -> int:
        """Binary transform always produces 1 channel."""
        return 1

    def instantiate(self, source_key: gp.ArrayKey, source_mask_key: gp.ArrayKey) -> None | tuple[gp.BatchFilter]:
        """Binary transform: just copy the source array."""
        return None
    
    @property
    def output_key(self) -> gp.ArrayKey:
        return gp.ArrayKey(self.source.upper())
    
    @property
    def mask_key(self) -> gp.ArrayKey:
        return gp.ArrayKey(f"{self.source.upper()}_MASK")


class LSDConfig(TransformConfig):
    """Configuration for Local Shape Descriptor (LSD) output transform."""

    type: Literal["lsd"] = "lsd"
    activation: ActivationStr = "Tanh"
    sigma: float = Field(default=5.0, description="Context size for computing descriptors")
    downsample: Optional[int] = 1
    background_mode: Literal["exclude", "zero", "label"] = Field(
        default="exclude",
        description="""How to handle background pixels in LSD computation:
        - 'exclude': Background pixels are masked out and don't contribute to loss (sparse training signal)
        - 'zero': Background pixels get 0 LSD values but contribute to loss (dense training, explicit 'no shape' signal)
        - 'label': Background is treated as its own shape and gets computed LSDs"""
    )
    binary_threshold: float = 0.5

    @property
    def num_channels(self) -> int:
        """LSD transform always produces 10 channels."""
        return 10

    def instantiate(self, source_key: gp.ArrayKey, source_mask_key: gp.ArrayKey) -> tuple[gp.BatchFilter]:
        """Get the AddLSD node for this transform."""
        binary_key = gp.ArrayKey(f"{self.source.upper()}_BINARY")
        instances_key = gp.ArrayKey(f"{self.source.upper()}_INSTANCES")
        threshold_to_binary = corditea.Threshold(
            source = source_key,
            target = binary_key,
            threshold=self.binary_threshold,
            background_values=255,
        )

        binary_to_instances = corditea.BinaryToInstances(
            binary_key,
            instances_key
        )
        lsd = corditea.AddLSD(
            segmentation=instances_key,
            descriptor=self.output_key,
            lsds_mask=self.mask_key,
            labels_mask=source_mask_key,  # Use source mask for invalid regions
            background_mode=self.background_mode,
            background_value=(0,255),
            sigma=self.sigma,
            downsample=self.downsample
        )
        return (threshold_to_binary, binary_to_instances, lsd)
    
    @property
    def output_key(self) -> gp.ArrayKey:
        return gp.ArrayKey(f"{self.source.upper()}_LSD")

    @property
    def mask_key(self) -> gp.ArrayKey:
        return gp.ArrayKey(f"{self.source.upper()}_LSD_MASK")

    def adjust_max_extent(self, max_extent: gp.Coordinate) -> gp.Coordinate:
        """Add LSD context (sigma * 6) to maximum extent."""
        context = self.sigma * 6
        return max_extent + gp.Coordinate((context,) * len(max_extent))

Transform = Union[BinaryConfig, LSDConfig]