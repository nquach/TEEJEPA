# Copyright (c) Meta Platforms, Inc. and affiliates.
# PyTorch Lightning DataModule for V-JEPA pretraining with litdata optimized dataset.

import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .optimized_video_dataset_vjepa import OptimizedVideoDatasetVJEPA
from .transforms_vjepa import DataAugmentationForVJEPA
from src.masks.multiseq_multiblock3d import MaskCollator


def _safe_makedir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class VideoDataModuleVJEPA(pl.LightningDataModule):
    """
    Lightning DataModule for V-JEPA pretraining using litdata optimized video dataset.
    Builds OptimizedVideoDatasetVJEPA, MaskCollator, and provides train_dataloader().
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config['data']
        self.model_config = config.get('model', config.get('data', {}))
        self.training_config = config['training']
        self.mask_config = config.get('mask', [])
        self.data_aug_config = config.get('data_aug', {})

        self.train_dataset = None
        self.mask_collator = None
        self.transform = None

    def setup(self, stage=None):
        if stage != 'fit' and stage is not None:
            return
        data_cfg = self.data_config
        model_cfg = self.model_config
        train_cfg = self.training_config
        crop_size = model_cfg.get('crop_size', data_cfg.get('crop_size', 224))
        patch_size = model_cfg.get('patch_size', data_cfg.get('patch_size', 16))
        tubelet_size = data_cfg.get('tubelet_size', 2)
        dataset_fpcs = data_cfg.get('dataset_fpcs', [16])

        normalize_mean = data_cfg.get('normalize_mean', [0.485, 0.456, 0.406])
        normalize_std = data_cfg.get('normalize_std', [0.229, 0.224, 0.225])
        self.transform = DataAugmentationForVJEPA(
            crop_size=crop_size,
            random_resize_aspect_ratio=tuple(
                self.data_aug_config.get('random_resize_aspect_ratio', [0.75, 1.35])
            ),
            random_resize_scale=tuple(
                self.data_aug_config.get('random_resize_scale', [0.3, 1.0])
            ),
            random_horizontal_flip=True,
            reprob=self.data_aug_config.get('reprob', 0.0),
            motion_shift=self.data_aug_config.get('motion_shift', False),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        train_optimized_dir = data_cfg.get('train_optimized_dir')
        if not train_optimized_dir:
            raise ValueError(
                "data.train_optimized_dir is required for VideoDataModuleVJEPA."
            )
        cache_dir = data_cfg.get('cache_dir', './output/cache')
        _safe_makedir(cache_dir)

        storage_options = None
        if data_cfg.get('cloud_type') == 's3_public':
            try:
                import botocore
                storage_options = {
                    "config": botocore.config.Config(
                        retries={"max_attempts": 1000, "mode": "adaptive"},
                        signature_version=botocore.UNSIGNED,
                    )
                }
            except ImportError:
                pass

        self.train_dataset = OptimizedVideoDatasetVJEPA(
            data_dir=train_optimized_dir,
            frames_to_sample=data_cfg.get('frames_to_sample', 16),
            temporal_stride=data_cfg.get('temporal_stride', 1),
            num_frames=dataset_fpcs[0] if dataset_fpcs else 16,
            subset_ratio=data_cfg.get('subset_ratio'),
            seed=train_cfg.get('seed', 0),
            transform=self.transform,
            cache_dir=cache_dir,
            max_cache_size=data_cfg.get('max_cache_size', '50GB'),
            drop_last=True,
            storage_options=storage_options,
        )

        crop_size_tuple = (crop_size, crop_size)
        patch_size_tuple = (patch_size, patch_size)
        self.mask_collator = MaskCollator(
            cfgs_mask=self.mask_config,
            dataset_fpcs=dataset_fpcs,
            crop_size=crop_size_tuple,
            patch_size=patch_size_tuple,
            tubelet_size=tubelet_size,
        )
        print(f"VideoDataModuleVJEPA: train dataset size = {len(self.train_dataset)}")

    def train_dataloader(self):
        if self.train_dataset is None or self.mask_collator is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        batch_size = self.training_config.get('batch_size', 24)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.training_config.get('num_workers', 8),
            pin_memory=self.training_config.get('pin_memory', True),
            collate_fn=self.mask_collator,
            drop_last=True,
        )

    def get_mask_collator(self):
        """Return the MaskCollator so the Lightning module or callbacks can call .step()."""
        return self.mask_collator
