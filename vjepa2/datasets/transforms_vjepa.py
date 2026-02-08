# Copyright (c) Meta Platforms, Inc. and affiliates.
# V-JEPA transform for optimized video dataset: random resize crop, flip, normalize.
# Expects input tensor [C, T, H, W] in [0, 1] range.

import torch

# Use vjepa2's existing video transforms
import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing


def _tensor_normalize_inplace(tensor, mean, std):
    """Normalize tensor (C, T, H, W) in place. mean/std in 0-1 range, shape (C,) or (1,1,1,1)."""
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, dtype=torch.float32, device=tensor.device)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std, dtype=torch.float32, device=tensor.device)
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


class DataAugmentationForVJEPA:
    """
    Data augmentation for V-JEPA: random resize crop, horizontal flip, normalize.
    Input: tensor [C, T, H, W] in [0, 1].
    Output: tensor [C, T, H, W] normalized.
    """

    def __init__(
        self,
        crop_size=224,
        random_resize_aspect_ratio=(0.75, 1.35),
        random_resize_scale=(0.3, 1.0),
        random_horizontal_flip=True,
        reprob=0.0,
        motion_shift=False,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
    ):
        self.crop_size = crop_size
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.random_horizontal_flip = random_horizontal_flip
        self.motion_shift = motion_shift
        self.mean = torch.tensor(normalize_mean, dtype=torch.float32).view(1, 1, 1, 1)
        self.std = torch.tensor(normalize_std, dtype=torch.float32).view(1, 1, 1, 1)
        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift
            if motion_shift
            else video_transforms.random_resized_crop
        )
        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob, mode="pixel", max_count=1, num_splits=1, device="cpu"
        )

    def __call__(self, video):
        """
        Args:
            video: tensor [C, T, H, W], float in [0, 1]
        Returns:
            tensor [C, T, H, W], normalized
        """
        if not torch.is_tensor(video):
            video = torch.tensor(video, dtype=torch.float32)
        video = video.float()
        if video.max() > 1.0:
            video = video / 255.0
        # video: [C, T, H, W]
        video = self.spatial_transform(
            images=video,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            video, _ = video_transforms.horizontal_flip(0.5, video)
        video = _tensor_normalize_inplace(video, self.mean, self.std)
        if self.reprob > 0:
            video = video.permute(1, 0, 2, 3)  # T C H W for erase
            video = self.erase_transform(video)
            video = video.permute(1, 0, 2, 3)  # C T H W
        return video
