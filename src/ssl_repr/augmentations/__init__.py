"""Augmentation strategies for self-supervised learning."""

from ssl_repr.augmentations.tabular import (
    TabularAugmenter,
    cutmix_tabular,
    feature_corruption,
    gaussian_noise,
)

__all__ = [
    "TabularAugmenter",
    "cutmix_tabular",
    "feature_corruption",
    "gaussian_noise",
]
