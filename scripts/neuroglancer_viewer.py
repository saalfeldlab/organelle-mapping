#!/usr/bin/env python
"""Interactive Neuroglancer viewer with keyboard shortcuts for channel group selection.

This script starts a local Neuroglancer server with multi-channel layers for
raw, targets, output, and mask arrays. Use number keys to switch between
channel groups.

Usage:
    python neuroglancer_viewer.py --url <zarr_url>
    python neuroglancer_viewer.py --url <zarr_url> --channels 19

Keyboard shortcuts:
    Channel selection:
        1: Channels 0-2 (LSD set 1, RGB)
        2: Channels 3-5 (LSD set 2, RGB)
        3: Channels 6-8 (LSD set 3, RGB)
        4: Channel 9 (grayscale)
        5: Channel 10 (grayscale)
        6: Channel 11 (grayscale)
        7: Channel 12 (grayscale)
        8: Channel 13 (grayscale)
        9: Channel 14 (grayscale)
        0: Channel 15 (grayscale)
        -: Channel 16 (grayscale)
        =: Channel 17 (grayscale)
        q: Channel 18 (grayscale)

    Array visibility toggle:
        g: Toggle raw (grayscale/EM)
        t: Toggle targets
        o: Toggle output
        m: Toggle mask
"""

import logging

import click
import neuroglancer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RGB colors for channel groups
CHANNEL_COLORS = ["#ff0000", "#00ff00", "#0000ff"]

# Base shader for all channels
CHANNEL_SHADER = """#uicontrol invlerp contrast
#uicontrol vec3 color color
void main() {
  float contrast_value = contrast();
  if (VOLUME_RENDERING) {
    emitRGBA(vec4(color * contrast_value, contrast_value));
  }
  else {
    emitRGB(color * contrast_value);
  }
}
"""


def create_channel_layer(source_url: str, array_name: str, channel: int, color: str) -> neuroglancer.ImageLayer:
    """Create a layer configuration for a single channel."""
    return neuroglancer.ImageLayer(
        source=f"zarr://{source_url}/{array_name}",
        local_dimensions=neuroglancer.CoordinateSpace(
            names=["c'"],
            units=[""],
            scales=[1]
        ),
        local_position=[channel],
        opacity=1,
        blend="additive",
        shader=CHANNEL_SHADER,
        shader_controls={
            "contrast": neuroglancer.InvlerpParameters(
                range=[0, 1],
                window=[0, 1]
            ),
            "color": color
        },
    )


def create_raw_layer(source_url: str) -> neuroglancer.ImageLayer:
    """Create a single-channel raw layer."""
    return neuroglancer.ImageLayer(
        source=f"zarr://{source_url}/raw",
        opacity=1,
        shader="""#uicontrol invlerp contrast
void main() {
  float value = contrast();
  emitGrayscale(value);
}
""",
        shader_controls={
            "contrast": neuroglancer.InvlerpParameters(
                range=[-1, 1],
                window=[-1, 1]
            ),
        },
    )


def setup_viewer(base_url: str, num_channels: int, arrays: list[str]) -> neuroglancer.Viewer:
    """Set up the Neuroglancer viewer with all layers and keybindings."""

    viewer = neuroglancer.Viewer()

    # Add all layers
    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units=["m", "m", "m"],
            scales=[8e-9, 8e-9, 8e-9]
        )
        s.position = [96, 96, 96]
        s.cross_section_scale = 0.3
        s.projection_scale = 128

        # Add raw as a single layer (always visible)
        s.layers["raw"] = create_raw_layer(base_url)

        # Generate multi-channel layers for other arrays (excluding raw)
        multi_channel_arrays = [a for a in arrays if a != "raw"]
        for array_name in multi_channel_arrays:
            for channel in range(num_channels):
                # Determine color: RGB for LSD channels 0-8, white for binary channels 9+
                if channel < 9:
                    color_idx = channel % 3
                    color = CHANNEL_COLORS[color_idx]
                else:
                    color = "#ffffff"  # white for grayscale channels

                layer_name = f"{array_name} c{channel}"
                layer = create_channel_layer(base_url, array_name, channel, color)

                s.layers[layer_name] = layer
                # Start with all layers archived (hidden)
                s.layers[layer_name].archived = True

    # Define channel groups: first 3 are RGB triplets (LSD), rest are individual grayscale
    # Group format: (start_channel, end_channel, description)
    channel_groups = [
        (0, 3, "LSD 0-2"),      # RGB
        (3, 6, "LSD 3-5"),      # RGB
        (6, 9, "LSD 6-8"),      # RGB
        (9, 10, "ch 9"),        # grayscale
        (10, 11, "ch 10"),      # grayscale
        (11, 12, "ch 11"),      # grayscale
        (12, 13, "ch 12"),      # grayscale
        (13, 14, "ch 13"),      # grayscale
        (14, 15, "ch 14"),      # grayscale
        (15, 16, "ch 15"),      # grayscale
        (16, 17, "ch 16"),      # grayscale
        (17, 18, "ch 17"),      # grayscale
        (18, 19, "ch 18"),      # grayscale
    ]

    # Track array visibility state (needs to be defined before switch_to_group)
    array_visibility = {
        'raw': True,
        'targets': True,
        'output': True,
        'mask': True
    }

    # Function to switch to a channel group
    def switch_to_group(group_idx: int):
        """Switch visibility to a specific channel group."""
        if group_idx >= len(channel_groups):
            return

        start_channel, end_channel, desc = channel_groups[group_idx]

        with viewer.txn() as s:
            for array_name in multi_channel_arrays:
                for channel in range(num_channels):
                    layer_name = f"{array_name} c{channel}"
                    in_group = start_channel <= channel < end_channel

                    if in_group:
                        # Respect array visibility setting
                        s.layers[layer_name].visible = array_visibility.get(array_name, True)
                        s.layers[layer_name].archived = False
                    else:
                        s.layers[layer_name].visible = False
                        s.layers[layer_name].archived = True

        logger.info(f"Switched to group {group_idx + 1}: {desc}")

    # Key bindings for each group
    key_bindings = [
        'digit1', 'digit2', 'digit3', 'digit4', 'digit5',
        'digit6', 'digit7', 'digit8', 'digit9', 'digit0',
        'minus', 'equal', 'keyq'
    ]

    # Register actions for each channel group
    for i, key in enumerate(key_bindings):
        if i >= len(channel_groups):
            break

        # Create action
        action_name = f'channel-group-{i}'
        # Use default argument to capture current value of i
        viewer.actions.add(action_name, lambda s, idx=i: switch_to_group(idx))

        # Bind to key
        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer[key] = action_name

    # Start with first group visible
    switch_to_group(0)

    # Function to toggle array visibility
    def toggle_array(array_name: str):
        """Toggle visibility of all layers for an array."""
        array_visibility[array_name] = not array_visibility[array_name]
        visible = array_visibility[array_name]

        with viewer.txn() as s:
            if array_name == 'raw':
                # Raw is a single layer
                s.layers['raw'].visible = visible
            else:
                # Multi-channel arrays: toggle visibility of non-archived layers
                for channel in range(num_channels):
                    layer_name = f"{array_name} c{channel}"
                    if layer_name in s.layers:
                        # Only toggle visibility if not archived (i.e., in current group)
                        if not s.layers[layer_name].archived:
                            s.layers[layer_name].visible = visible

        status = "visible" if visible else "hidden"
        logger.info(f"Toggled {array_name}: {status}")

    # Array toggle key bindings
    array_keys = {
        'keyg': 'raw',
        'keyt': 'targets',
        'keyo': 'output',
        'keym': 'mask'
    }

    for key, array_name in array_keys.items():
        action_name = f'toggle-{array_name}'
        viewer.actions.add(action_name, lambda s, arr=array_name: toggle_array(arr))
        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer[key] = action_name

    # Add status message with instructions
    with viewer.config_state.txn() as s:
        s.status_messages['instructions'] = (
            'Channel: 1-3 LSD, 4-9,0,-,=,BS grayscale | Array: g/t/o/m toggle'
        )

    return viewer


@click.command()
@click.option('--url', required=True,
              help='Base zarr URL (e.g., https://fileglancer.int.janelia.org/fc/files/ABC123/00000011.zarr)')
@click.option('--channels', '-c', default=19, type=int,
              help='Number of channels in the data (default: 19)')
@click.option('--arrays', '-a', default='raw,targets,output,mask',
              help='Comma-separated list of array names (default: raw,targets,output,mask)')
@click.option('--bind-address', default='127.0.0.1',
              help='Address to bind the server to (default: 127.0.0.1)')
@click.option('--port', default=0, type=int,
              help='Port to bind to (default: 0 for auto-select)')
def main(url: str, channels: int, arrays: str, bind_address: str, port: int):
    """Start an interactive Neuroglancer viewer with channel group keybindings.

    Creates separate layers for each channel with RGB coloring and additive blending.
    Use number keys 1-9 to switch between channel groups.

    Examples:
        # View a snapshot
        python neuroglancer_viewer.py --url https://fileglancer.int.janelia.org/fc/files/ABC123/00000011.zarr

        # Specify number of channels
        python neuroglancer_viewer.py --url <url> --channels 19

        # Only view targets and output
        python neuroglancer_viewer.py --url <url> --arrays targets,output
    """
    # Parse arrays
    array_list = [a.strip() for a in arrays.split(',')]

    # Strip trailing slash from base URL
    base_url = url.rstrip('/')

    # Configure neuroglancer server
    neuroglancer.set_server_bind_address(bind_address, port)

    logger.info(f"Setting up viewer for {channels} channels")
    logger.info(f"Arrays: {array_list}")
    logger.info(f"URL: {base_url}")

    # Set up viewer
    viewer = setup_viewer(base_url, channels, array_list)

    # Print viewer URL
    print(f"\nNeuroglancer viewer URL: {viewer}")
    print(f"\nKeyboard shortcuts:")
    num_groups = (channels + 2) // 3
    for i in range(min(num_groups, 9)):
        start = i * 3
        end = min(start + 3, channels) - 1
        print(f"  {i + 1}: Channels {start}-{end}")
    print(f"\nPress Ctrl+C to stop the server.\n")

    # Keep the script running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
