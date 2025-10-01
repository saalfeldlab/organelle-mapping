import copy
import logging
from pathlib import Path
from typing import Optional, Sequence
from functools import partial

import click
import torch
import yaml
from pydantic import TypeAdapter, ValidationError
from organelle_mapping.config import RunConfig, CheckpointEditConfig
from organelle_mapping.utils import setup_package_logger

logger = logging.getLogger(__name__)


def transfer_checkpoint_weights(
    checkpoint_path: Path,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    heads_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Transfer weights from source checkpoint to match target label configuration.

    Args:
        checkpoint_path: Path to source checkpoint
        source_labels: Labels in the source model
        target_labels: Labels for the target model
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
                    raise TypeError(f"Expected optimizer state '{buffer_name}' to be a tensor, got {type(buffer_value)}")
                if buffer_value.ndim == weights[hk].ndim:
                    if buffer_value.size() != weights[hk].size():
                        raise ValueError(f"Assumptions about how model weights map to optimizer state do not hold for key {hk}. Can't transfer optimizer state.")
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
                init_fn = partial(torch.nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')

            weights_new[key] = match_head_weights(
                weights[key],
                source_labels,
                target_labels,
                init_fn=init_fn
            )
            if optimizer_key is not None:
                for buffer_name in buffers[optimizer_key]:
                    logger.info(f"Transferring optimizer buffer '{buffer_name}' for head parameter {key}")
                    optimizer_state_new["state"][optimizer_key][buffer_name] = match_head_weights(
                        optimizer_state["state"][optimizer_key][buffer_name],
                        source_labels,
                        target_labels,
                        init_fn=partial(torch.nn.init.constant_, val=0.0)
                    )

    checkpoint_new = copy.deepcopy(checkpoint)
    checkpoint_new["model_state_dict"] = weights_new
    if optimizer_state_new is not None:
        checkpoint_new["optimizer_state_dict"] = optimizer_state_new
    return checkpoint_new


def match_head_weights(
    source_params: torch.Tensor,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    init_fn: Optional[callable] = None,
) -> torch.Tensor:
    """Match parameters from source labels to target labels.

    Args:
        source_params: Tensor of shape [num_source_labels, ...]
        source_labels: Labels in source model
        target_labels: Labels for target model
        init_fn: Initialization function for unmatched labels. Required if any
                 target labels are not in source labels.

    Returns:
        Tensor of shape [num_target_labels, ...] with matched weights
    """
    # Create tensor for target weights with same shape except first dimension
    target_shape = list(source_params.shape)
    target_shape[0] = len(target_labels)
    target_weights = torch.zeros(target_shape, dtype=source_params.dtype)

    for target_idx, target_label in enumerate(target_labels):
        if target_label in source_labels:
            source_idx = source_labels.index(target_label)
            target_weights[target_idx] = source_params[source_idx]
            logger.debug(f"Matched {target_label}: source[{source_idx}] -> target[{target_idx}]")
        else:
            if init_fn is None:
                msg = f"Target label '{target_label}' not found in source labels and no init_fn provided"
                raise ValueError(msg)
            # Unsqueeze, apply init, then squeeze back
            init_fn(target_weights[target_idx].unsqueeze(0))
            target_weights[target_idx] = target_weights[target_idx].squeeze(0)

            # Get init function name for better logging
            init_name = getattr(init_fn, '__name__', getattr(init_fn, 'func', init_fn).__name__ if hasattr(init_fn, 'func') else str(init_fn))
            logger.warning(f"Target label '{target_label}' not found in source labels, initializing with {init_name}")

    return target_weights


def create_transfer_checkpoint(finetuning_config: CheckpointEditConfig, output_checkpoint: Optional[Path] = None) -> str:
    """Create a new checkpoint with transferred weights.

    Args:
        finetuning_config: CheckpointEditConfig object with source info and transfer settings

    Returns:
        Path to the prepared checkpoint
    """
    if output_checkpoint is None:
        output_checkpoint = Path(finetuning_config.source_checkpoint.name)
    if output_checkpoint.exists():
        logger.info(f"Output checkpoint {output_checkpoint} already exists, skipping transfer.")
        return str(output_checkpoint)
    if finetuning_config.target_labels is None:
        output_checkpoint.symlink_to(finetuning_config.source_checkpoint)
        logger.info(f"Created symlink to {finetuning_config.source_checkpoint}")
        return str(output_checkpoint)

    # Load source run.yaml to get labels for transfer
    with open(finetuning_config.source_experiment) as f:
        source_run_config = TypeAdapter(RunConfig).validate_python(
            yaml.safe_load(f),
            context={'base_dir': finetuning_config.source_experiment.parent}
        )

    # Transfer weights
    source_labels = source_run_config.labels
    logger.info(f"Transferring weights from {source_labels} to {finetuning_config.target_labels}")

    # Determine heads_keys: finetuning config > architecture config > fallback to None
    heads_keys = finetuning_config.heads_keys or source_run_config.architecture.output_head_keys

    checkpoint_new = transfer_checkpoint_weights(
        finetuning_config.source_checkpoint,
        source_labels,
        finetuning_config.target_labels,
        heads_keys=heads_keys
    )
    # Save transferred checkpoint
    torch.save(checkpoint_new, output_checkpoint)
    logger.info(f"Saved transferred checkpoint to {output_checkpoint}")

    return str(output_checkpoint)



@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to finetuning.yaml config file. If provided, other options are ignored."
)
@click.option(
    "--source-checkpoint",
    type=click.Path(exists=True, path_type=Path),
    help="Path to source checkpoint file"
)
@click.option(
    "--source-experiment",
    type=click.Path(exists=True, path_type=Path),
    help="Path to source experiment run.yaml config file"
)
@click.option(
    "--target-labels",
    multiple=True,
    help="Target labels for the output model. Can be specified multiple times."
)
@click.option(
    "--heads-keys",
    multiple=True,
    help="State dict keys for output heads. Can be specified multiple times."
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output checkpoint path."
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="Set the logging level."
)
def main(
    config: Optional[Path],
    source_checkpoint: Optional[Path],
    source_experiment: Optional[Path],
    target_labels: tuple[str, ...],
    heads_keys: tuple[str, ...],
    output: Path,
    log_level: str
):
    """Transfer weights between model checkpoints with different label configurations."""
    setup_package_logger(log_level)

    try:
        if config:
            with open(config) as f:
                finetuning_config = TypeAdapter(CheckpointEditConfig).validate_python(
                    yaml.safe_load(f),
                    context={'base_dir': config.parent}
                )
        else:
            # Construct from CLI arguments
            if not source_checkpoint or not source_experiment:
                raise click.ClickException("--source-checkpoint and --source-experiment are required when not using --config")

            target_labels_list = list(target_labels) if target_labels else None
            heads_keys_list = list(heads_keys) if heads_keys else None

            finetuning_config = CheckpointEditConfig(
                source_checkpoint=source_checkpoint,
                source_experiment=source_experiment,
                target_labels=target_labels_list,
                heads_keys=heads_keys_list
            )

        # Call the function with explicit output path
        output_path = create_transfer_checkpoint(finetuning_config, output)
        logger.info(f"Weight transfer complete. Output: {output_path}")

    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise click.ClickException(f"Invalid configuration: {e}")
    except Exception as e:
        logger.error(f"Weight transfer failed: {e}")
        raise click.ClickException(f"Weight transfer failed: {e}")


if __name__ == "__main__":
    main()