import gunpowder as gp
import numpy as np
from .lsd_utils import get_local_shape_descriptors
from .aff_utils import Affs

class LSDAffinities(gp.BatchFilter):

    def __init__(self, array,affinities=None, sigma=2.0, voxel_size = None):
        self.array = array
        self.affinities = affinities
        self.sigma = sigma
        self.voxel_size = voxel_size
        self.affinities = affinities
        if self.affinities is not None:
            self.aff_fn = Affs(
                neighborhood=self.affinities,
            )
        else:
            self.aff_fn = None

    def setup(self):
        self.spec[self.array].dtype = np.float32

    def process(self, batch, request):

        data = batch[self.array].data
        data = data.astype(np.uint8)
        lsd_data = get_local_shape_descriptors(
        data,
        sigma = self.sigma,
        voxel_size = self.voxel_size
        ).astype(np.float32)
        if self.aff_fn is not None:
            affs_data = self.aff_fn(data).astype(np.float32)
            lsd_data = np.concatenate([affs_data,lsd_data], axis=0)
        batch[self.array].data = lsd_data
        # batch[self.array].dtype = np.float32