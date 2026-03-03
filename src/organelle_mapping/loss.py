from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from organelle_mapping.config.target import TargetConfig


class MaskedMultiLabelBCEwithLogits(torch.nn.BCEWithLogitsLoss):
    """Masked multi-label binary cross-entropy loss with logits."""

    def __init__(self, pos_weight, spatial_dims=3):
        if pos_weight is not None:
            pos_weight = torch.Tensor(pos_weight)[(...,) + (None,) * spatial_dims]
        super().__init__(reduction="none", pos_weight=pos_weight)
        self.spatial_dims = spatial_dims

    def forward(self, output, target, mask):
        bce = torch.sum(super().forward(output, target) * mask)
        mask_sum = torch.sum(mask)
        if mask_sum == 0:
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)
        bce /= mask_sum
        return bce


class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Masked binary cross-entropy loss with logits for single labels."""

    def __init__(self, pos_weight=None):
        super().__init__(reduction="none", pos_weight=pos_weight)

    def forward(self, output, target, mask):
        bce = torch.sum(super().forward(output, target) * mask)
        mask_sum = torch.sum(mask)
        if mask_sum == 0:
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)
        bce /= mask_sum
        return bce


class MaskedMSELoss(torch.nn.MSELoss):
    """Masked mean squared error loss."""

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, output, target, mask):
        mse = torch.sum(super().forward(output, target) * mask)
        mask_sum = torch.sum(mask)
        if mask_sum == 0:
            return torch.tensor(0.0, device=output.device, dtype=output.dtype)
        mse /= mask_sum
        return mse


class CombinedLoss(torch.nn.Module):
    """Combined loss function that handles targets with different loss functions."""

    def __init__(self, targets: Sequence["TargetConfig"]):
        super().__init__()
        self.targets = targets
        self.loss_functions = []
        self.channel_ranges = []

        # Create loss functions and calculate channel ranges
        current_channel = 0
        for target in targets:
            # Create loss function for this target
            loss_fn = target.create_loss_function()
            self.loss_functions.append(loss_fn)

            # Calculate channel range for this target
            target_channels = sum(tt.num_channels for tt in target.transforms)
            start_channel = current_channel
            end_channel = current_channel + target_channels
            self.channel_ranges.append((start_channel, end_channel))
            current_channel = end_channel

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute multi-task loss.

        Args:
            output: Model predictions [batch, channels, ...]
            target: Ground truth [batch, channels, ...]
            mask: Mask tensor [batch, channels, ...]

        Returns:
            Combined loss scalar
        """
        total_loss = 0.0

        for target_config, loss_fn, (start_ch, end_ch) in zip(
            self.targets, self.loss_functions, self.channel_ranges
        ):
            # Extract channels for this target
            output_slice = output[:, start_ch:end_ch]
            target_slice = target[:, start_ch:end_ch]
            mask_slice = mask[:, start_ch:end_ch]

            # Compute loss for this target
            loss = loss_fn(output_slice, target_slice, mask_slice)

            # Weight the loss
            weighted_loss = target_config.weight * loss
            total_loss += weighted_loss

        return total_loss
