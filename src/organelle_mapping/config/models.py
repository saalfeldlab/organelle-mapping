from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import BaseModel, Field, field_validator, model_validator, TypeAdapter


class ArchitectureConfig(BaseModel, ABC):
    name: str
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    in_channels: int
    out_channels: int
    model_config = {"discriminator": "name"}

    @abstractmethod
    def instantiate(self):
        pass


NormName = Literal[
    "instance",
    "batch",
    "instance_nvfuser",
    "group",
    "layer",
    "localresponse",
    "syncbatch",
]


class SwinUNETRConfig(ArchitectureConfig):
    name: Literal["swin_unetr"]
    input_shape: tuple[int, int, int] = (196, 196, 196)
    output_shape: tuple[int, int, int] = (196, 196, 196)
    in_channels: int = Field(1, gt=0)
    out_channels: int = Field(..., gt=0)
    depths: Sequence[int] = Field((2, 2, 2, 2), min_items=1)
    num_heads: Sequence[int] = Field((3, 6, 12, 24), min_items=1)
    feature_size: int = Field(24, gt=0)
    norm_name: Union[NormName, tuple[Any, ...]] = "instance"
    drop_rate: float = Field(0.0, ge=0.0, le=1.0)
    attn_drop_rate: float = Field(0.0, ge=0.0, le=1.0)
    dropout_path_rate: float = Field(0.0, ge=0.0, le=1.0)
    normalize: bool = True
    use_checkpoint: bool = True
    downsample: Literal["merging", "mergingv2"] = "merging"
    use_v2: bool = False

    @model_validator(mode="after")
    def validate_input_shape(self):
        if self.input_shape != self.output_shape:
            msg = f"input_shape ({self.input_shape}) and output_shape ({self.output_shape}) must be the same for SwinUNETR."
            raise ValueError(msg)
        return self

    @field_validator("norm_name", mode="after")
    @classmethod
    def validate_norm_name(cls, value):
        if isinstance(value, tuple):
            if len(value) == 0:
                msg = f"norm_name ({value}) must be a single value or a tuple with one value."
                raise ValueError(msg)
            TypeAdapter(NormName).validate_python(value[0])
        return value

    @field_validator("feature_size")
    @classmethod
    def validate_feature_size(cls, value):
        if value % 12 != 0:
            msg = f"feature_size ({value}) must be divisible by 12."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_depths_and_num_heads(self):
        if len(self.depths) != len(self.num_heads):
            msg = f"depths ({self.depths}) and num_heads ({self.num_heads}) must have the same length."
            raise ValueError(msg)
        if not all(d > 0 for d in self.depths):
            msg = f"All values in depths ({self.depths}) must be positive."
            raise ValueError(msg)
        if not all(n > 0 for n in self.num_heads):
            msg = f"All values in num_heads ({self.num_heads}) must be positive."
            raise ValueError(msg)
        return self

    def instantiate(self):
        from monai.networks.nets import SwinUNETR

        return SwinUNETR(
            img_size=self.input_shape,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            feature_size=self.feature_size,
            norm_name=self.norm_name,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            dropout_path_rate=self.dropout_path_rate,
            normalize=self.normalize,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=3,
            downsample=self.downsample,
            use_v2=self.use_v2,
        )


class StandardUnetConfig(ArchitectureConfig):
    name: Literal["standard_unet"]
    input_shape: tuple[int, int, int] = (178, 178, 178)
    output_shape: tuple[int, int, int] = (56, 56, 56)
    in_channels: int = Field(1, gt=0)
    out_channels: int = Field(..., gt=0)
    num_fmaps: int = Field(16, gt=0)
    fmac_inc_factor: int = Field(6, gt=0)
    downsample_factors: Sequence[tuple[int, int, int]] = (
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    )
    kernel_size_down: Sequence[Sequence[tuple[int, int, int]]] = (((3, 3, 3), (3, 3, 3), (3, 3, 3)),) * 4
    kernel_size_up: Sequence[Sequence[tuple[int, int, int]]] = (((3, 3, 3), (3, 3, 3), (3, 3, 3)),) * 3

    @model_validator(mode="after")
    def validate_kernel_sizes(self):

        if len(self.kernel_size_down) != len(self.downsample_factors) + 1:
            msg = (
                f"kernel_size_down ({self.kernel_size_down}) must be one element longer "
                f"than downsample_factors ({self.downsample_factors})."
            )
            raise ValueError(msg)

        if len(self.kernel_size_up) != len(self.downsample_factors):
            msg = (
                f"kernel_size_up ({self.kernel_size_up}) must have the same length as "
                f"downsample_factors ({self.downsample_factors})."
            )
            raise ValueError(msg)

        for kernel_group in self.kernel_size_down + self.kernel_size_up:
            for kernel in kernel_group:
                if not all(k % 2 == 1 for k in kernel):
                    msg = f"All integers in kernel ({kernel}) must be odd."
                    raise ValueError(msg)
                if not all(k > 0 for k in kernel):
                    msg = f"Kernels ({kernel}) must be positive."
                    raise ValueError(msg)

        return self

    def instantiate(self):
        from organelle_mapping.model import StandardUnet

        return StandardUnet(
            self.in_channels,
            self.out_channels,
            num_fmaps=self.num_fmaps,
            fmap_inc_factor=self.fmac_inc_factor,
            downsample_factors=self.downsample_factors,
            kernel_size_down=self.kernel_size_down,
            kernel_size_up=self.kernel_size_up,
        )


Architecture = Annotated[
    Union[StandardUnetConfig, SwinUNETRConfig],
    Field(discriminator="name"),
]
