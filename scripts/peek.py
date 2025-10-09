import yaml
import sys
from organelle_mapping.model import load_eval_model
from funlib.geometry.coordinate import Coordinate

print("Starting peek.py script...", file=sys.stderr)

network_config_path = "architecture.yaml"
checkpoint_path = "model_checkpoint_470000"

print(f"Looking for config at: {network_config_path}", file=sys.stderr)
print(f"Looking for checkpoint at: {checkpoint_path}", file=sys.stderr)

try:
    # Load and parse the architecture config
    from pydantic import TypeAdapter
    from organelle_mapping.config import Architecture
    
    with open(network_config_path) as network_config_file:
        config_dict = yaml.safe_load(network_config_file)
    
    # Convert dict to proper Architecture config object
    network_config = TypeAdapter(Architecture).validate_python(config_dict)
    print("Successfully loaded network config", file=sys.stderr)
except Exception as e:
    print(f"Error loading config: {e}", file=sys.stderr)
    raise

try:
    model = load_eval_model(network_config, checkpoint_path)
    print("Successfully loaded model", file=sys.stderr)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    raise

# Define voxel sizes
input_voxel_size = (8, 8, 8)
output_voxel_size = input_voxel_size

# Calculate read and write shapes
read_shape = Coordinate(network_config.input_shape) * Coordinate(input_voxel_size)
write_shape = Coordinate(network_config.output_shape) * Coordinate(output_voxel_size)

# Define output channels and block shape
output_channels = network_config.out_channels
block_shape = (*network_config.output_shape, output_channels)

print(f"Setup complete:", file=sys.stderr)
print(f"  Input voxel size: {input_voxel_size}", file=sys.stderr)
print(f"  Output voxel size: {output_voxel_size}", file=sys.stderr)
print(f"  Read shape: {read_shape}", file=sys.stderr)
print(f"  Write shape: {write_shape}", file=sys.stderr)
print(f"  Output channels: {output_channels}", file=sys.stderr)
print(f"  Block shape: {block_shape}", file=sys.stderr)
