import logging

import yaml
from funlib.geometry.coordinate import Coordinate

from organelle_mapping.model import load_eval_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Starting peek.py script...")

network_config_path = "architecture.yaml"
checkpoint_path = "model_checkpoint_470000"

logger.info(f"Looking for config at: {network_config_path}")
logger.info(f"Looking for checkpoint at: {checkpoint_path}")

try:
    # Load and parse the architecture config
    from pydantic import TypeAdapter

    from organelle_mapping.config import Architecture

    with open(network_config_path) as network_config_file:
        config_dict = yaml.safe_load(network_config_file)

    # Convert dict to proper Architecture config object
    network_config = TypeAdapter(Architecture).validate_python(config_dict)
    logger.info("Successfully loaded network config")
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise

try:
    model = load_eval_model(network_config, checkpoint_path)
    logger.info("Successfully loaded model")
except Exception as e:
    logger.error(f"Error loading model: {e}")
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

logger.info("Setup complete:")
logger.info(f"  Input voxel size: {input_voxel_size}")
logger.info(f"  Output voxel size: {output_voxel_size}")
logger.info(f"  Read shape: {read_shape}")
logger.info(f"  Write shape: {write_shape}")
logger.info(f"  Output channels: {output_channels}")
logger.info(f"  Block shape: {block_shape}")
