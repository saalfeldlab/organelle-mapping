import math
from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, Union

import corditea
import gunpowder as gp
from pydantic import BaseModel, Field
from pydantic_core import core_schema


class AugmentationConfig(BaseModel, ABC):
    name: str
    model_config = {"discriminator": "name"}

    @abstractmethod
    def instantiate(self):
        pass


class ArrayKeyField:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.str_schema(strict=True)
        )

    @staticmethod
    def _validate(value):
        return gp.ArrayKey(value)


class CoordinateField:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.tuple_schema(
                [core_schema.int_schema()] * 3, min_length=3, max_length=3
            ),
        )

    @staticmethod
    def _validate(value):
        return gp.Coordinate(value)


class IntensityAugmentConfig(AugmentationConfig):
    name: Literal["intensity_augment"]
    array: ArrayKeyField
    scale_min: float = Field(
        0.75, description="Minimum scale for intensity augmentation."
    )
    scale_max: float = Field(
        1.5, description="Maximum scale for intensity augmentation."
    )
    shift_min: float = Field(
        -0.15, description="Minimum shift for intensity augmentation."
    )
    shift_max: float = Field(
        0.15, description="Maximum shift for intensity augmentation."
    )
    z_section_wise: bool = Field(
        False, description="Perform augmentation z-section-wise."
    )
    clip: bool = True
    p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying intensity augmentation.",
    )

    def instantiate(self):
        return gp.IntensityAugment(
            self.array,
            self.scale_min,
            self.scale_max,
            self.shift_min,
            self.shift_max,
            z_section_wise=self.z_section_wise,
            clip=self.clip,
            p=self.p,
        )


class GammaAugmentConfig(AugmentationConfig):
    name: Literal["gamma_augment"]
    arrays: list[ArrayKeyField]
    gamma_min: float = Field(0.75, description="Minimum gamma value for augmentation.")
    gamma_max: float = Field(
        4 / 3.0, description="Maximum gamma value for augmentation."
    )

    def instantiate(self):
        return corditea.GammaAugment(self.arrays, self.gamma_min, self.gamma_max)


class SimpleAugmentConfig(AugmentationConfig):
    name: Literal["simple_augment"]
    mirror_only: list[int] | None = None
    transpose_only: list[int] | None = None
    mirror_probs: list[float] | None = None
    transpose_probs: dict[tuple[int], float | list[float]] | None = None
    p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying simple augmentation.",
    )

    def instantiate(self):
        return gp.SimpleAugment(
            mirror_only=self.mirror_only,
            transpose_only=self.transpose_only,
            mirror_probs=self.mirror_probs,
            transpose_probs=self.transpose_probs,
            p=self.p,
        )


class ElasticAugmentConfig(AugmentationConfig):
    name: Literal["corditea_elastic_augment"]
    control_point_spacing: CoordinateField = Field(
        default=(25, 25, 25),
        description="Default control point spacing for elastic augmentation.",
    )
    control_point_displacement_sigma: CoordinateField = Field(default=(24, 24, 24))
    rotation_interval: tuple[float] = (0, math.pi / 2.0)
    subsample: int = 8
    augmentation_probability: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Probability of applying elastic augmentation.",
    )
    uniform_3d_rotation: bool = True

    def instaniate(self):
        return corditea.ElasticAugment(
            control_point_spacing=self.control_point_spacing,
            control_point_displacement_sigma=self.control_point_displacement_sigma,
            rotation_interval=self.rotation_interval,
            subsample=self.subsample,
            augmentation_probability=self.augmentation_probability,
            uniform_3d_rotation=self.uniform_3d_rotation,
        )
class IntensityScaleShiftConfig(AugmentationConfig):
    name: Literal["intensity_scale_shift"]
    array: ArrayKeyField
    scale: float
    shift: float

    def instantiate(self):
        return gp.IntensityScaleShift(
            self.array,
            self.scale, 
            self.shift
            )

class GaussianNoiseAugmentConfig(AugmentationConfig):
    name: Literal["gaussian_noise_augment"]
    array: ArrayKeyField
    clip: bool = True
    var_range: tuple[float] = (0, 0.01)
    noise_prob: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying Gaussian noise augmentation.",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for skimage random noise.",
    )

    def instantiate(self):
        return corditea.GaussianNoiseAugment(
            self.array,
            clip=self.clip,
            noise_prob=self.noise_prob,
            var_range=self.var_range,
            **self.kwargs,  # Pass additional arguments
        )


Augmentation = Annotated[
    Union[
        IntensityAugmentConfig,
        GammaAugmentConfig,
        SimpleAugmentConfig,
        ElasticAugmentConfig,
        IntensityScaleShiftConfig,
        GaussianNoiseAugmentConfig,
    ],
    Field(discriminator="name"),
]


class AugmentationPipeline(BaseModel):
    augmentations: list[Augmentation]
