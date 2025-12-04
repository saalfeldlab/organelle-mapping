# Neuroglancer Shaders

This directory contains GLSL shaders for visualizing training outputs in Neuroglancer.

## lsd_rgb_viewer.glsl

Visualizes LSD (Local Shape Descriptor) and binary segmentation channels as RGB.

### Features
- **Channel selector**: Slider to switch between different channels
- **Brightness control**: Adjustable brightness for better visibility
- **Multi-task support**: View both LSD components and binary segmentations
- **RGB for first two LSD sets**: Channels 0-2 and 3-5 shown as RGB
- **Grayscale for remaining channels**: Individual channels shown in grayscale

### Channel Mapping
- **0**: LSD mean offsets (0-2) as RGB - directional vectors from voxel to object center
- **1**: LSD covariance (3-5) as RGB - first three components of shape covariance matrix
- **2**: LSD covariance (6-8) as RGB - last three components of shape covariance matrix
- **3**: LSD size channel (9) in grayscale - object size/extent information
- **4-12**: Binary segmentation channels (10-18) in grayscale - individual organelle predictions

### Setup

1. **Rechunk snapshot zarr files** (required for ^ notation):
   ```bash
   pixi run python scripts/rechunk_snapshots.py /path/to/snapshots
   ```

2. **Configure neuroglancer layer**:
   - Open the layer's **Source** tab
   - Add a tick `'` after `d0` to make it `d0'` (sliceable batch dimension)
   - Add a caret `^` after `d1` to make it `d1^` (channel dimension for shader access)

3. **Apply shader**: Copy contents of `lsd_rgb_viewer.glsl` into the layer's **Shader** tab

### Usage Tips
- Start with channelSet=0 to view mean offset LSDs (most interpretable)
- Increase brightness for dim/early training outputs
- For ground truth targets, values typically start at 0 (not -1)
- Binary channels show individual label predictions
