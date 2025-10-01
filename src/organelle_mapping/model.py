import logging

import funlib.learn.torch
import torch
import tems

logger = logging.getLogger(__name__)


def load_eval_model(architecture_config, checkpoint_path):
    model_backbone = architecture_config.instantiate()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model



class MaskedMultiLabelBCEwithLogits(torch.nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight, spatial_dims=3):
        pos_weight = torch.Tensor(pos_weight)[(...,) + (None,) * spatial_dims]
        self.loss_fn = super().__init__(reduction="none", pos_weight=pos_weight)
        self.spatial_dims = spatial_dims

    def forward(self, output, target, mask):
        bce = torch.sum(super().forward(output, target) * mask)
        bce /= torch.sum(mask)
        return bce

class TemsUnet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fmaps_down=None,
        fmaps_up=None,
        downsample_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
        residual=True,
        upsample_mode="trilinear",
        padding="valid",
        activation=torch.nn.ReLU,
        final_activation=torch.nn.Identity,
    ):
        super().__init__()
        if downsample_factors is None:
            downsample_factors = [(2,2,2), (2,2,2), (2,2,2)]
        if kernel_size_down is None:
            kernel_size_level = [(3,3,3), (3,3,3), (3,3,3)]
            kernel_size_down = [kernel_size_level,] * (len(downsample_factors) + 1)
        if kernel_size_up is None:
            kernel_size_level = [(3,3,3), (3,3,3), (3,3,3)]
            kernel_size_up = [
                kernel_size_level,
            ] * len(downsample_factors)
        if fmaps_down is None:
            fmaps_down = list(16 * 6**lvl for lvl in range(len(downsample_factors)+1))
        if fmaps_up is None:
            fmaps_up = fmaps_down
        levels = []
        for lvl in range(len(downsample_factors)):
            if lvl == 0:
                in_ch = in_channels
            else:
                in_ch = fmaps_down[lvl-1]
            if lvl == len(downsample_factors) - 1:
                fmaps_below = fmaps_down[lvl+1]
            else:
                fmaps_below = fmaps_up[lvl+1]

            levels.append((
                tems.ConvPass(
                    dims=3, 
                    in_channels=in_ch, 
                    out_channels=fmaps_down[lvl],
                    kernel_sizes = kernel_size_down[lvl],
                    residual=residual,
                    activation=activation,
                ),
                tems.Downsample(
                    dims=3, 
                    downsample_factor=downsample_factors[lvl]
                    ),
                tems.Upsample(
                    dims=3, 
                    scale_factor=downsample_factors[lvl], 
                    mode=upsample_mode
                    ),
                tems.ConvPass(
                    dims=3,
                    in_channels=fmaps_down[lvl] + fmaps_below,
                    out_channels=fmaps_up[lvl],
                    kernel_sizes=kernel_size_up[lvl],
                    activation=activation
                )
            ))
        lvl= len(downsample_factors)
        bottleneck = tems.ConvPass(
            dims=3,
            in_channels=fmaps_down[lvl-1],
            out_channels=fmaps_down[lvl],
            kernel_sizes=kernel_size_down[lvl],
            activation=activation
        )
        
        self.unet_backbone = tems.UNet(dims=3,
                                       bottleneck=bottleneck,
                                       levels=levels)
        self.final_conv = torch.nn.Conv3d(fmaps_up[0], out_channels, (1,1,1))
        self.final_activation = final_activation()
    def forward(self, x):
        x = self.unet_backbone(x)
        x = self.final_conv(x)
        return self.final_activation(x)
        
        
                
        
    
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
            kernel_size_up = [
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
