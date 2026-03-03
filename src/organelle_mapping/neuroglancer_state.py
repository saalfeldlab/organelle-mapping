"""Generate Neuroglancer JSON states for viewing predictions.

This module provides programmatic generation of Neuroglancer JSON states
that can be loaded directly into Neuroglancer or URL-encoded for sharing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from organelle_mapping.config.inference import InferenceConfig

# LSD channel count constants
LSD_FULL_CHANNELS = 10  # Full LSD: offsets (3) + variance (3) + pearson (3) + mass (1)
LSD_RGB_CHANNELS = 3  # RGB subset (e.g., just offsets)

# Shader for LSD channels (10 channels: offsets, variance, pearson, mass)
LSD_SHADER = """#uicontrol invlerp normalized(range=[0, 255])
#uicontrol int channelSet slider(min=0, max=9, step=3, default=0)

// channelSet = first channel of set:
//   0: offsets (ch 0-2)
//   3: variance (ch 3-5)
//   6: pearson (ch 6-8)
//   9: mass (ch 9)

void main() {
  float r, g, b, value;

  if (channelSet < 3) {
    r = normalized(getDataValue(0));
    g = normalized(getDataValue(1));
    b = normalized(getDataValue(2));
    emitRGB(vec3(r, g, b));
  } else if (channelSet < 6) {
    r = normalized(getDataValue(3));
    g = normalized(getDataValue(4));
    b = normalized(getDataValue(5));
    emitRGB(vec3(r, g, b));
  } else if (channelSet < 9) {
    r = normalized(getDataValue(6));
    g = normalized(getDataValue(7));
    b = normalized(getDataValue(8));
    emitRGB(vec3(r, g, b));
  } else {
    value = normalized(getDataValue(9));
    emitGrayscale(value);
  }
}
"""

# Shader for single-channel grayscale data
GRAYSCALE_SHADER = """#uicontrol invlerp normalized(range=[0, 255])

void main() {
  emitGrayscale(normalized());
}
"""

# Shader for 3-channel RGB data (e.g., LSD offsets/variance/pearson)
RGB_SHADER = """#uicontrol invlerp normalized(range=[0, 255])

void main() {
  float r = normalized(getDataValue(0));
  float g = normalized(getDataValue(1));
  float b = normalized(getDataValue(2));
  emitRGB(vec3(r, g, b));
}
"""

# Shader for raw EM data
RAW_SHADER = """#uicontrol invlerp normalized

void main() {
  emitGrayscale(normalized());
}
"""


@dataclass
class LayerConfig:
    """Configuration for a Neuroglancer layer."""

    name: str
    source: str | dict[str, Any]
    shader: str
    shader_controls: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    opacity: float = 1.0
    blend: str = "default"


def build_source_url(
    container_path: str,
    dataset_path: str,
    fileglancer_url: str | None = None,
) -> str:
    """Build a neuroglancer-compatible source URL.

    Args:
        container_path: Local path to the zarr container
        dataset_path: Path within the container (e.g., recon-1/em/fibsem-uint8/s1)
        fileglancer_url: Complete fileglancer URL for any directory in the path.
            The path after the token is matched against the full local path
            to determine what portion to replace.
            Example: https://fileglancer.int.janelia.org/fc/files/TOKEN/predictions/data.zarr
            The "predictions/data.zarr" is found in the full path and
            everything up to (and including) that match is replaced with the URL.

    Returns:
        Neuroglancer source URL
    """
    if fileglancer_url:
        fg_url = fileglancer_url.rstrip("/")

        # Extract the path portion after the token
        # URL format: https://fileglancer.int.janelia.org/files/TOKEN/path/to/dir
        # (also handles legacy /fc/files/ format)
        # We want the "path/to/dir" part (after the token)
        if "/files/" in fg_url:
            after_files = fg_url.split("/files/", 1)[1]
            # Skip the token (first component after /files/)
            parts = after_files.split("/", 1)
            fg_path = parts[1] if len(parts) > 1 else ""
        else:
            # Fallback: use everything after the last slash
            fg_path = fg_url.rsplit("/", 1)[-1]

        # Build the full local path (container + dataset)
        full_local = container_path.strip("/")
        if dataset_path:
            full_local = f"{full_local}/{dataset_path.strip('/')}"

        # Find fg_path in full_local and get everything after the match
        # Use rfind (last match) as heuristic for ambiguous cases
        remaining_path = ""
        if fg_path:
            idx = full_local.rfind(fg_path)
            if idx != -1:
                # Everything after the match
                after_match = full_local[idx + len(fg_path) :]
                remaining_path = after_match.strip("/")

        # Build final URL: fileglancer URL + remaining path
        full_path = fg_url
        if remaining_path:
            full_path = f"{full_path}/{remaining_path}"

        # Fileglancer URL format: https://fileglancer.../path|zarr2:
        url = f"{full_path}|zarr2:"
    else:
        # Local file URL
        full_path = container_path
        if dataset_path:
            full_path = f"{full_path}/{dataset_path}"
        url = f"zarr://{full_path}"

    return url


def build_multichannel_source(
    container_path: str,
    dataset_path: str,
    fileglancer_url: str | None = None,
    voxel_size: tuple[float, float, float] = (8.0, 8.0, 8.0),
) -> dict[str, Any]:
    """Build a source object for multi-channel data with channel dimension transform.

    Args:
        container_path: Path to the zarr container
        dataset_path: Path within the container
        fileglancer_url: Optional fileglancer URL (see build_source_url for details)
        voxel_size: Voxel size in nm (z, y, x)

    Returns:
        Source dict with channel dimension transform
    """
    base_url = build_source_url(container_path, dataset_path, fileglancer_url)

    return {
        "url": base_url,
        "transform": {
            "outputDimensions": {
                "c^": [1, ""],
                "z": [voxel_size[0], "nm"],
                "y": [voxel_size[1], "nm"],
                "x": [voxel_size[2], "nm"],
            }
        },
    }


def create_raw_layer(
    container_path: str,
    dataset_path: str,
    fileglancer_url: str | None = None,
) -> LayerConfig:
    """Create a layer config for raw EM data."""
    url = build_source_url(container_path, dataset_path, fileglancer_url)

    return LayerConfig(
        name="raw",
        source=url,
        shader=RAW_SHADER,
        shader_controls={
            "normalized": {"range": [0, 255]},
        },
        visible=True,
        opacity=1.0,
    )


def create_lsd_layer(
    container_path: str,
    dataset_path: str,
    name: str = "organelle_lsd",
    fileglancer_url: str | None = None,
    *,
    visible: bool = True,
    voxel_size: tuple[float, float, float] = (8.0, 8.0, 8.0),
) -> LayerConfig:
    """Create a layer config for LSD predictions (10 channels)."""
    source = build_multichannel_source(
        container_path,
        dataset_path,
        fileglancer_url=fileglancer_url,
        voxel_size=voxel_size,
    )

    return LayerConfig(
        name=name,
        source=source,
        shader=LSD_SHADER,
        shader_controls={
            "normalized": {"range": [0, 255]},
            "channelSet": 0,
        },
        visible=visible,
    )


def create_grayscale_layer(
    container_path: str,
    dataset_path: str,
    name: str,
    fileglancer_url: str | None = None,
    *,
    visible: bool = True,
) -> LayerConfig:
    """Create a layer config for single-channel grayscale data."""
    url = build_source_url(
        container_path,
        dataset_path,
        fileglancer_url,
    )

    return LayerConfig(
        name=name,
        source=url,
        shader=GRAYSCALE_SHADER,
        shader_controls={
            "normalized": {"range": [0, 255]},
        },
        visible=visible,
        blend="additive",
    )


def create_rgb_layer(
    container_path: str,
    dataset_path: str,
    name: str,
    fileglancer_url: str | None = None,
    *,
    visible: bool = True,
    voxel_size: tuple[float, float, float] = (8.0, 8.0, 8.0),
) -> LayerConfig:
    """Create a layer config for 3-channel RGB data."""
    source = build_multichannel_source(
        container_path,
        dataset_path,
        fileglancer_url=fileglancer_url,
        voxel_size=voxel_size,
    )

    return LayerConfig(
        name=name,
        source=source,
        shader=RGB_SHADER,
        shader_controls={
            "normalized": {"range": [0, 255]},
        },
        visible=visible,
    )


def create_multichannel_layer(
    container_path: str,
    dataset_path: str,
    name: str,
    fileglancer_url: str | None = None,
    *,
    visible: bool = True,
    voxel_size: tuple[float, float, float] = (8.0, 8.0, 8.0),
) -> LayerConfig:
    """Create a layer config for multi-channel data with channel slicer."""
    source = build_multichannel_source(
        container_path,
        dataset_path,
        fileglancer_url=fileglancer_url,
        voxel_size=voxel_size,
    )

    return LayerConfig(
        name=name,
        source=source,
        shader=GRAYSCALE_SHADER,
        shader_controls={
            "normalized": {"range": [0, 255]},
        },
        visible=visible,
    )


def layer_to_dict(layer: LayerConfig) -> dict[str, Any]:
    """Convert a LayerConfig to Neuroglancer JSON layer format."""
    layer_dict = {
        "type": "image",
        "source": layer.source,
        "shader": layer.shader,
        "shaderControls": layer.shader_controls,
        "visible": layer.visible,
        "opacity": layer.opacity,
        "name": layer.name,
    }

    if layer.blend != "default":
        layer_dict["blend"] = layer.blend

    return layer_dict


@dataclass
class NeuroglancerState:
    """Builder for Neuroglancer JSON states."""

    layers: list[LayerConfig] = field(default_factory=list)

    def add_layer(self, layer: LayerConfig) -> "NeuroglancerState":
        """Add a layer to the state."""
        self.layers.append(layer)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to Neuroglancer JSON state dict."""
        return {
            "layers": [layer_to_dict(layer) for layer in self.layers],
            "layout": "4panel-alt",
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def state_from_inference_config(
    inference_config_path: str | Path,
    raw_fileglancer_url: str | None = None,
    prediction_fileglancer_url: str | None = None,
) -> NeuroglancerState:
    """Create a Neuroglancer state from an inference config file.

    Args:
        inference_config_path: Path to inference.yaml
        raw_fileglancer_url: Optional fileglancer base URL for raw data
            (e.g., https://fileglancer.int.janelia.org/fc/files/TOKEN)
        prediction_fileglancer_url: Optional fileglancer base URL for predictions

    Returns:
        NeuroglancerState ready for export
    """
    config_path = Path(inference_config_path)
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    config = InferenceConfig.model_validate(
        raw_config,
        context={"base_dir": str(config_path.parent)},
    )

    # Build channel index to transform type mapping from run_config targets
    channel_idx = 0
    channel_to_transform_type: dict[int, str] = {}
    for target in config.targets:
        for transform in target.transforms:
            for _ in range(transform.num_channels):
                channel_to_transform_type[channel_idx] = transform.type
                channel_idx += 1

    def parse_channels(channels_str: str) -> tuple[int, int]:
        """Parse channel string to (start, count)."""
        if "-" in channels_str:
            start, end = map(int, channels_str.split("-"))
            return start, end - start + 1
        else:
            return int(channels_str), 1

    def get_transform_type(start_channel: int) -> str:
        """Get transform type for a channel index."""
        return channel_to_transform_type.get(start_channel, "binary")

    # Build state
    state = NeuroglancerState()
    voxel_size = config.input_data.voxel_size

    # Add raw layer
    raw_layer = create_raw_layer(
        config.input_data.container,
        config.input_data.dataset,
        fileglancer_url=raw_fileglancer_url,
    )
    state.add_layer(raw_layer)

    # Add prediction layers based on transform type and channel count
    for i, out in enumerate(config.output_data.outputs):
        start_channel, num_channels = parse_channels(out.channels)
        transform_type = get_transform_type(start_channel)
        dataset_path = f"{config.output_data.dataset}/{out.name}"
        visible = i == 0  # First layer visible

        if transform_type == "lsd":
            # LSD: shader depends on channel count
            if num_channels == LSD_FULL_CHANNELS:
                layer = create_lsd_layer(
                    config.output_data.container,
                    dataset_path,
                    name=out.name,
                    fileglancer_url=prediction_fileglancer_url,
                    visible=visible,
                    voxel_size=voxel_size,
                )
            elif num_channels == LSD_RGB_CHANNELS:
                layer = create_rgb_layer(
                    config.output_data.container,
                    dataset_path,
                    name=out.name,
                    fileglancer_url=prediction_fileglancer_url,
                    visible=visible,
                    voxel_size=voxel_size,
                )
            elif num_channels == 1:
                layer = create_grayscale_layer(
                    config.output_data.container,
                    dataset_path,
                    name=out.name,
                    fileglancer_url=prediction_fileglancer_url,
                    visible=visible,
                )
            else:
                # Fallback: multichannel with slicer
                layer = create_multichannel_layer(
                    config.output_data.container,
                    dataset_path,
                    name=out.name,
                    fileglancer_url=prediction_fileglancer_url,
                    visible=visible,
                    voxel_size=voxel_size,
                )
        else:
            # Binary and other transforms: grayscale
            layer = create_grayscale_layer(
                config.output_data.container,
                dataset_path,
                name=out.name,
                fileglancer_url=prediction_fileglancer_url,
                visible=visible,
            )
        state.add_layer(layer)

    return state
