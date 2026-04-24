import copy
import logging
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import click
import torch
import yaml
from pydantic import TypeAdapter, ValidationError

from organelle_mapping.config import CheckpointEditConfig, RunConfig
from organelle_mapping.utils import setup_package_logger

logger = logging.getLogger(__name__)


def transfer_checkpoint_weights(
    checkpoint_path: Path,
    source_descriptors: Sequence[str],
    target_descriptors: Sequence[str],
    channel_mapping: Optional[dict[str, Optional[str]]] = None,
    heads_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Transfer weights from source checkpoint to match target channel configuration.

    Args:
        checkpoint_path: Path to source checkpoint
        source_descriptors: Channel descriptors in the source model
        target_descriptors: Channel descriptors for the target model
        channel_mapping: Optional explicit mapping from target to source descriptors.
            If None, auto-matches by descriptor name.
        heads_keys: Keys in state_dict that contain the output heads.
            If None, assumes last two keys are the heads (weight and bias).

    Returns:
        Modified checkpoint dict with transferred weights
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    weights = checkpoint["model_state_dict"]
    optimizer_state = checkpoint.get("optimizer_state_dict", None)
    weights_new = copy.deepcopy(weights)
    optimizer_state_new = copy.deepcopy(optimizer_state)

    # Build effective channel mapping
    if channel_mapping is None:
        # Auto-match by descriptor name
        channel_mapping = {}
        for target_desc in target_descriptors:
            if target_desc in source_descriptors:
                channel_mapping[target_desc] = target_desc
            else:
                channel_mapping[target_desc] = None  # Will be initialized fresh

    if heads_keys is None:
        # Assuming last two keys are heads (e.g., 'final_conv.weight', 'final_conv.bias')
        heads_keys = list(weights.keys())[-2:]
        logger.warning(f"Heads keys not defined. Assuming last two keys of saved weights: {heads_keys}")

    if optimizer_state is not None:
        optimizer_heads_keys = [list(weights.keys()).index(hk) for hk in heads_keys]
        buffers = {}
        for hk, ohk in zip(heads_keys, optimizer_heads_keys):
            buffers[ohk] = []
            for buffer_name, buffer_value in optimizer_state["state"][ohk].items():
                if not isinstance(buffer_value, torch.Tensor):
                    msg = f"Expected optimizer state '{buffer_name}' to be a tensor, got {type(buffer_value)}"
                    raise TypeError(msg)
                if buffer_value.ndim == weights[hk].ndim:
                    if buffer_value.size() != weights[hk].size():
                        msg = (
                            f"Assumptions about how model weights map to optimizer state do not hold for key {hk}."
                            " Can't transfer optimizer state."
                        )
                        raise ValueError(msg)
                    else:
                        buffers[ohk].append(buffer_name)
    else:
        optimizer_heads_keys = [None] * len(heads_keys)

    for key, optimizer_key in zip(heads_keys, optimizer_heads_keys):
        if key in weights:
            logger.info(f"Transferring weights for key: {key}")

            # Define init function based on parameter type
            if weights[key].ndim == 1:  # Bias
                init_fn = partial(torch.nn.init.constant_, val=0.01)
            else:  # Conv weights
                init_fn = partial(torch.nn.init.kaiming_normal_, mode="fan_out", nonlinearity="relu")

            weights_new[key] = match_head_weights(
                weights[key], source_descriptors, target_descriptors, channel_mapping, init_fn=init_fn
            )
            if optimizer_key is not None:
                for buffer_name in buffers[optimizer_key]:
                    logger.info(f"Transferring optimizer buffer '{buffer_name}' for head parameter {key}")
                    optimizer_state_new["state"][optimizer_key][buffer_name] = match_head_weights(
                        optimizer_state["state"][optimizer_key][buffer_name],
                        source_descriptors,
                        target_descriptors,
                        channel_mapping,
                        init_fn=partial(torch.nn.init.constant_, val=0.0),
                    )

    checkpoint_new = copy.deepcopy(checkpoint)
    checkpoint_new["model_state_dict"] = weights_new
    if optimizer_state_new is not None:
        checkpoint_new["optimizer_state_dict"] = optimizer_state_new
    return checkpoint_new


def match_head_weights(
    source_params: torch.Tensor,
    source_descriptors: Sequence[str],
    target_descriptors: Sequence[str],
    channel_mapping: dict[str, Optional[str]],
    init_fn: Optional[callable] = None,
) -> torch.Tensor:
    """Match parameters from source channels to target channels using descriptors.

    Args:
        source_params: Tensor of shape [num_source_channels, ...]
        source_descriptors: Channel descriptors in source model
        target_descriptors: Channel descriptors for target model
        channel_mapping: Mapping from target descriptors to source descriptors
        init_fn: Initialization function for unmatched channels. Required if any
                 target channels are not mapped.

    Returns:
        Tensor of shape [num_target_channels, ...] with matched weights
    """
    # Create tensor for target weights with same shape except first dimension
    target_shape = list(source_params.shape)
    target_shape[0] = len(target_descriptors)
    target_weights = torch.zeros(target_shape, dtype=source_params.dtype)

    for target_idx, target_desc in enumerate(target_descriptors):
        source_desc = channel_mapping.get(target_desc)

        if source_desc is not None and source_desc in source_descriptors:
            source_idx = source_descriptors.index(source_desc)
            target_weights[target_idx] = source_params[source_idx]
            logger.debug(f"Matched {target_desc}: source[{source_idx}] ({source_desc}) -> target[{target_idx}]")
        else:
            if init_fn is None:
                msg = f"Target channel '{target_desc}' not mapped and no init_fn provided"
                raise ValueError(msg)
            # Unsqueeze, apply init, then squeeze back
            init_fn(target_weights[target_idx].unsqueeze(0))
            target_weights[target_idx] = target_weights[target_idx].squeeze(0)

            # Get init function name for better logging
            init_name = getattr(
                init_fn,
                "__name__",
                getattr(init_fn, "func", init_fn).__name__ if hasattr(init_fn, "func") else str(init_fn),
            )
            logger.warning(f"Target channel '{target_desc}' not mapped to source, initializing with {init_name}")

    return target_weights


def create_transfer_checkpoint(
    finetuning_config: CheckpointEditConfig, output_checkpoint: Optional[Path] = None
) -> str:
    """Create a new checkpoint with transferred weights.

    Args:
        finetuning_config: CheckpointEditConfig object with source info and transfer settings.
            When channel_mapping is None, creates a symlink (assumes identical channels).
            When channel_mapping is provided, remaps channels according to the mapping.
        output_checkpoint: Path for output checkpoint. If None, uses source checkpoint name.

    Returns:
        Path to the prepared checkpoint
    """
    if output_checkpoint is None:
        output_checkpoint = Path(finetuning_config.source_checkpoint.name)
    if output_checkpoint.exists():
        logger.info(f"Output checkpoint {output_checkpoint} already exists, skipping transfer.")
        return str(output_checkpoint)

    # No channel mapping means assume identical channels - just symlink
    if finetuning_config.channel_mapping is None:
        output_checkpoint.symlink_to(finetuning_config.source_checkpoint)
        logger.info(f"No channel mapping provided. Created symlink to {finetuning_config.source_checkpoint}")
        return str(output_checkpoint)

    # Load source config to get descriptors
    with open(finetuning_config.source_experiment) as f:
        source_run_config = TypeAdapter(RunConfig).validate_python(
            yaml.safe_load(f), context={"base_dir": finetuning_config.source_experiment.parent}
        )
    source_descriptors = source_run_config.channel_descriptors

    # Target descriptors are the keys of the channel mapping
    target_descriptors = list(finetuning_config.channel_mapping.keys())

    logger.info(f"Source channels: {source_descriptors}")
    logger.info(f"Target channels: {target_descriptors}")

    # Determine heads_keys: finetuning config > architecture config > fallback to None
    heads_keys = finetuning_config.heads_keys or source_run_config.architecture.output_head_keys

    checkpoint_new = transfer_checkpoint_weights(
        finetuning_config.source_checkpoint,
        source_descriptors,
        target_descriptors,
        channel_mapping=finetuning_config.channel_mapping,
        heads_keys=heads_keys,
    )
    # Save transferred checkpoint
    torch.save(checkpoint_new, output_checkpoint)
    logger.info(f"Saved transferred checkpoint to {output_checkpoint}")

    return str(output_checkpoint)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to finetuning.yaml config file. If provided, other options are ignored.",
)
@click.option(
    "--source-checkpoint", type=click.Path(exists=True, path_type=Path), help="Path to source checkpoint file"
)
@click.option(
    "--source-experiment",
    type=click.Path(exists=True, path_type=Path),
    help="Path to source experiment run.yaml config file",
)
@click.option(
    "--target-descriptors",
    multiple=True,
    help="Target channel descriptors for the output model (e.g., 'mito_binary', 'er_lsd_0'). "
    "Can be specified multiple times. If not provided, creates symlink (assumes identical channels).",
)
@click.option("--heads-keys", multiple=True, help="State dict keys for output heads. Can be specified multiple times.")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output checkpoint path.")
@click.option(
    "--log-level",
    "log_levels",
    multiple=True,
    default=("INFO",),
    help=(
        "Logging level. Use 'LEVEL' (e.g. 'DEBUG') to set the organelle_mapping logger, "
        "or '<logger>.<LEVEL>' (e.g. 'gunpowder.DEBUG', 'lsd_lite.ERROR') for other loggers. "
        "May be passed multiple times."
    ),
)
def main(
    config: Optional[Path],
    source_checkpoint: Optional[Path],
    source_experiment: Optional[Path],
    target_descriptors: tuple[str, ...],
    heads_keys: tuple[str, ...],
    output: Path,
    log_levels: tuple[str, ...],
):
    """Transfer weights between model checkpoints with different label configurations."""
    setup_package_logger(log_levels)

    try:
        if config:
            with open(config) as f:
                finetuning_config = TypeAdapter(CheckpointEditConfig).validate_python(
                    yaml.safe_load(f), context={"base_dir": config.parent}
                )
        else:
            # Construct from CLI arguments
            if not source_checkpoint or not source_experiment:
                msg = "--source-checkpoint and --source-experiment are required when not using --config"
                raise click.ClickException(msg)

            heads_keys_list = list(heads_keys) if heads_keys else None

            # Build channel_mapping from target_descriptors if provided (auto-match by name)
            channel_mapping = None
            if target_descriptors:
                channel_mapping = {desc: desc for desc in target_descriptors}

            finetuning_config = CheckpointEditConfig(
                source_checkpoint=source_checkpoint,
                source_experiment=source_experiment,
                channel_mapping=channel_mapping,
                heads_keys=heads_keys_list,
            )

        # Call the function with explicit output path
        output_path = create_transfer_checkpoint(finetuning_config, output)
        logger.info(f"Weight transfer complete. Output: {output_path}")

    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        msg = f"Invalid configuration: {e}"
        raise click.ClickException(msg) from e
    except Exception as e:
        logger.error(f"Weight transfer failed: {e}")
        msg = f"Weight transfer failed: {e}"
        raise click.ClickException(msg) from e


if __name__ == "__main__":
    main()
