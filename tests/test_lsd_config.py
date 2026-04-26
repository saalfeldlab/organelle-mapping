import pytest
from pydantic import ValidationError

from organelle_mapping.config.transform import LSDConfig


def test_default_backend_is_lsd_lite():
    cfg = LSDConfig(source="mito")
    assert cfg.backend == "lsd-lite"


def test_jax_backend_accepts_default_downsample():
    cfg = LSDConfig(source="mito", backend="lsd-jax")
    assert cfg.backend == "lsd-jax"
    assert cfg.downsample == 1


def test_jax_backend_rejects_downsample_gt_one():
    with pytest.raises(ValidationError, match="downsample"):
        LSDConfig(source="mito", backend="lsd-jax", downsample=2)


def test_lite_backend_allows_downsample_gt_one():
    cfg = LSDConfig(source="mito", backend="lsd-lite", downsample=2)
    assert cfg.downsample == 2
