import logging

import funlib.learn.torch
import torch

logger = logging.getLogger(__name__)


class SACModel(torch.nn.Module):
    """Split-Apply-Concatenate model wrapper for multi-task learning.

    This implements a Split-Apply-Concatenate pattern where:
    1. Model output is split by channel ranges for each transform
    2. Different activations are applied to each transform's channels
    3. Results are concatenated back together

    Uses PyTorch's train()/eval() modes to switch between training and inference activations.

    Args:
        base_model: The underlying model to wrap
        targets: List of target configurations containing transforms
    """

    def __init__(self, base_model: torch.nn.Module, targets):
        super().__init__()
        self.base_model = base_model

        # Build channel ranges and activation specs for each transform
        self.transform_specs = []
        current_channel = 0

        for target in targets:
            for transform in target.transforms:
                start_ch = current_channel
                end_ch = current_channel + transform.num_channels

                # Instantiate both training and inference activations
                train_activation_class = getattr(torch.nn, transform.activation)
                train_activation = train_activation_class()

                inference_activation_class = getattr(torch.nn, transform.inference_activation)
                inference_activation = inference_activation_class()

                self.transform_specs.append({
                    'start': start_ch,
                    'end': end_ch,
                    'train_activation': train_activation,
                    'inference_activation': inference_activation,
                    'name': f"{target.name or target.type}_{transform.source}_{transform.type}"
                })

                current_channel = end_ch

        logger.info(f"SACModel initialized with {len(self.transform_specs)} transforms:")
        for spec in self.transform_specs:
            logger.info(f"  Channels {spec['start']}:{spec['end']} ({spec['name']})")
            logger.info(f"    Training: {spec['train_activation']}, Inference: {spec['inference_activation']}")

    def forward(self, x):
        """Apply base model then split-apply-concatenate activations.

        Uses self.training (set by model.train()/model.eval()) to select appropriate activations.
        """
        # Get raw model output
        output = self.base_model(x)

        # Apply activations to each channel range
        activated_chunks = []
        for spec in self.transform_specs:
            chunk = output[:, spec['start']:spec['end']]

            # Select activation based on training mode
            activation = spec['train_activation'] if self.training else spec['inference_activation']
            activated_chunk = activation(chunk)
            activated_chunks.append(activated_chunk)

        # Concatenate back together
        return torch.cat(activated_chunks, dim=1)


def create_sac_model(architecture, targets) -> torch.nn.Module:
    """Create a model with Split-Apply-Concatenate multi-task activations.

    Args:
        architecture: Architecture configuration with instantiate() method
        targets: List of target configurations containing transforms

    Returns:
        Model wrapped with SACModel
    """
    base_model = architecture.instantiate()
    model = SACModel(base_model, targets)
    return model


def load_eval_model(architecture, targets, checkpoint_path: str):
    """Load a trained model for evaluation with multi-task activations.

    Args:
        architecture: Architecture configuration with instantiate() method
        targets: List of target configurations containing transforms
        checkpoint_path: Path to the checkpoint file

    Returns:
        Model loaded with trained weights and proper activations
    """
    # Create model with multi-task activations
    model = create_sac_model(architecture, targets)

    # Load checkpoint
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Load state dict into the base model (not the wrapper)
    model.base_model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model



class StandardUnet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_fmaps=16,
        fmap_inc_factor=6,
        downsample_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
        padding="valid"
    ):
        super().__init__()
        if downsample_factors is None:
            downsample_factors = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
        if kernel_size_down is None:
            kernel_size_level = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            kernel_size_down = [
                kernel_size_level,
            ] * (len(downsample_factors) + 1)
        if kernel_size_up is None:
            kernel_size_level = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            kernel_size_level = [
                kernel_size_level,
            ] * len(downsample_factors)

        self.unet_backbone = funlib.learn.torch.models.UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            constant_upsample=True,
            padding=padding
        )

        self.final_conv = torch.nn.Conv3d(num_fmaps, out_channels, (1, 1, 1), padding="valid")

    def forward(self, raw):
        x = self.unet_backbone(raw)
        return self.final_conv(x)
