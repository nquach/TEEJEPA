# Copyright (c) Meta Platforms, Inc. and affiliates.
# Optimized video dataset for V-JEPA using LitData StreamingDataset.
# Returns (buffer, label, clip_indices) for MaskCollator compatibility.

import os
import random
import warnings

import numpy as np
import torch
from litdata import StreamingDataset
from litdata.streaming.cache import Dir

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')


def _safe_makedir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class OptimizedVideoDatasetVJEPA(StreamingDataset):
    """
    LitData StreamingDataset that yields samples in V-JEPA format for MaskCollator.
    Each sample is (buffer, label, clip_indices) where buffer = [tensor [C, T, H, W]].
    """

    def __init__(
        self,
        data_dir,
        frames_to_sample=16,
        temporal_stride=1,
        num_frames=None,
        subset_ratio=None,
        seed=None,
        transform=None,
        cache_dir=None,
        max_cache_size='50GB',
        drop_last=False,
        storage_options=None,
    ):
        if cache_dir:
            _safe_makedir(cache_dir)
        try:
            if subset_ratio is not None:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir),
                    transform=None,
                    subsample=subset_ratio,
                    drop_last=drop_last,
                    max_cache_size=max_cache_size,
                    storage_options=storage_options,
                )
            else:
                super().__init__(
                    input_dir=Dir(path=cache_dir, url=data_dir),
                    transform=None,
                    drop_last=drop_last,
                    max_cache_size=max_cache_size,
                    storage_options=storage_options,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize StreamingDataset from {data_dir}. Error: {e}"
            )
        self.frames_to_sample = frames_to_sample
        self.temporal_stride = temporal_stride
        self.num_frames = num_frames or (frames_to_sample // temporal_stride)
        self.custom_transform = transform
        if seed is not None:
            random.seed(seed)
        print(f"OptimizedVideoDatasetVJEPA: loaded from {data_dir}, num_frames={self.num_frames}")

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        video = data['video']
        # Support both numpy and tensor from litdata
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        video = video.float()
        # Assume video shape (T, H, W, C) or (T, C, H, W)
        if video.shape[-1] == 3:
            # (T, H, W, C)
            pass
        else:
            # (T, C, H, W) -> (T, H, W, C)
            video = video.permute(0, 2, 3, 1)
        num_frames = video.shape[0]
        max_start = num_frames - self.frames_to_sample
        if max_start <= 0:
            start_frame = 0
        else:
            start_frame = random.randint(0, max_start)
        sampled = video[start_frame : start_frame + self.frames_to_sample]
        sampled = sampled[:: self.temporal_stride]
        if sampled.max() > 1.0:
            sampled = sampled / 255.0
        # (T, H, W, C) -> (C, T, H, W)
        video_tensor = sampled.permute(3, 0, 1, 2)
        if self.custom_transform is not None:
            video_tensor = self.custom_transform(video_tensor)
        n = video_tensor.shape[1]
        buffer = [video_tensor]
        label = 0
        clip_indices = [np.arange(n, dtype=np.int32)]
        return buffer, label, clip_indices
