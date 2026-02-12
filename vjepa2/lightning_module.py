# Copyright (c) Meta Platforms, Inc. and affiliates.
# PyTorch Lightning module for V-JEPA pretraining with schedule-free optimizer and EMA target.

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from app.vjepa.utils import init_video_model
from src.masks.utils import apply_masks
from optimizers.schedule_free_optimizer import create_schedule_free_optimizer

logger = logging.getLogger(__name__)


class VJEPALightningModule(pl.LightningModule):
    """
    Lightning module for V-JEPA pretraining: encoder, predictor, target encoder (EMA),
    schedule-free optimizer, and optional mask_collator.step() via callback.
    """

    def __init__(self, config, mask_collator=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.mask_collator = mask_collator

        data_cfg = config['data']
        model_cfg = config['model']
        mask_cfgs = config.get('mask', [])
        opt_cfg = config.get('optimizer', {})
        opt_training_cfg = config.get('optimization', {})
        loss_cfg = config.get('loss', {})

        dataset_fpcs = data_cfg.get('dataset_fpcs', [16])
        max_num_frames = max(dataset_fpcs)
        crop_size = model_cfg.get('crop_size', data_cfg.get('crop_size', 224))
        patch_size = model_cfg.get('patch_size', 16)
        tubelet_size = data_cfg.get('tubelet_size', 2)
        num_mask_tokens = int(len(mask_cfgs) * len(dataset_fpcs))

        encoder, predictor = init_video_model(
            device=torch.device('cpu'),
            patch_size=patch_size,
            max_num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            model_name=model_cfg.get('model_name', 'vit_base'),
            crop_size=crop_size,
            pred_depth=model_cfg.get('pred_depth', 6),
            pred_num_heads=model_cfg.get('pred_num_heads'),
            pred_embed_dim=model_cfg.get('pred_embed_dim', 384),
            uniform_power=model_cfg.get('uniform_power', False),
            use_mask_tokens=model_cfg.get('use_mask_tokens', False),
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=model_cfg.get('zero_init_mask_tokens', True),
            use_sdpa=model_cfg.get('use_sdpa', False),
            use_rope=model_cfg.get('use_rope', False),
            use_silu=model_cfg.get('use_silu', False),
            use_pred_silu=model_cfg.get('use_pred_silu', False),
            wide_silu=model_cfg.get('wide_silu', True),
            use_activation_checkpointing=model_cfg.get('use_activation_checkpointing', False),
        )
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.loss_exp = float(loss_cfg.get('loss_exp', 1.0))
        self.ema = opt_training_cfg.get('ema', [0.99925, 0.99925])
        self.ipe_scale = float(opt_training_cfg.get('ipe_scale', 1.25))
        self.opt_config = opt_cfg
        self._optimizer = None
        # For Option B: log train_loss_epoch every N steps as running avg so save_top_k gets multiple values
        self._save_every_n_steps = config.get('checkpoint', {}).get('save_every_n_steps')
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(t) for t in x)
        return x

    def training_step(self, batch, batch_idx):
        # batch: list of (udata, masks_enc, masks_pred) per fpc
        all_clips, all_masks_enc, all_masks_pred = [], [], []
        for fpc_sample in batch:
            udata, masks_enc, masks_pred = fpc_sample
            all_clips.append(self._to_device(udata[0][0]))
            all_masks_enc.append([self._to_device(m) for m in masks_enc])
            all_masks_pred.append([self._to_device(m) for m in masks_pred])

        with torch.no_grad():
            h = self.target_encoder(all_clips)
            h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
        h_masked = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, all_masks_pred)]

        z = self.encoder(all_clips, all_masks_enc)
        z = self.predictor(z, all_masks_enc, all_masks_pred)

        loss, n = 0.0, 0
        for zi, hi in zip(z, h_masked):
            for zij, hij in zip(zi, hi):
                loss = loss + torch.mean(torch.abs(zij - hij) ** self.loss_exp) / self.loss_exp
                n += 1
        loss = loss / n

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Log running average every save_every_n_steps so ModelCheckpoint (save_top_k) sees multiple values
        if self._save_every_n_steps is not None and self._save_every_n_steps > 0:
            loss_val = loss.detach().float().item()
            self._train_loss_sum += loss_val
            self._train_loss_count += 1
            if self.global_step > 0 and self.global_step % self._save_every_n_steps == 0:
                running_avg = self._train_loss_sum / self._train_loss_count
                self.log(
                    'train_loss_epoch',
                    running_avg,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
                self._train_loss_sum = 0.0
                self._train_loss_count = 0

        return loss

    def on_after_optimizer_step(self, optimizer):
        # EMA update: target = m * target + (1 - m) * encoder
        step = self.global_step
        total_steps = getattr(
            self.trainer,
            'estimated_stepping_batches',
            None,
        ) or getattr(self.trainer, 'num_training_batches', None)
        if total_steps is None:
            train_cfg = self.config.get('training', {})
            max_epochs = train_cfg.get('max_epochs', 800)
            total_steps = 1000 * max_epochs
        total_steps = int(total_steps * self.ipe_scale)
        total_steps = max(total_steps, 1)
        m = self.ema[0] + step * (self.ema[1] - self.ema[0]) / total_steps
        m = min(max(m, 0.0), 1.0)
        with torch.no_grad():
            params_k = list(self.target_encoder.parameters())
            params_q = list(self.encoder.parameters())
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1.0 - m)

        if self.mask_collator is not None:
            self.mask_collator.step()

    def on_before_optimizer_step(self, optimizer):
        # Schedule-free requires the optimizer to be in train mode before step()
        # Unwrap to the actual optimizer that will be stepped (e.g. Lightning/AMP wrappers)
        actual = optimizer
        while hasattr(actual, 'optimizer'):
            actual = actual.optimizer
        if hasattr(actual, 'train'):
            actual.train()
        # Also ensure our stored reference is in train mode (same object in typical case)
        if self._optimizer is not None and hasattr(self._optimizer, 'train'):
            self._optimizer.train()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Schedule-free: call .train() immediately before step() (required by schedule-free lib)
        actual = optimizer
        while hasattr(actual, 'optimizer'):
            actual = actual.optimizer
        if hasattr(actual, 'train'):
            actual.train()
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def configure_optimizers(self):
        opt_cfg = self.opt_config
        lr = opt_cfg.get('lr', 5.25e-4)
        if isinstance(lr, (list, tuple)):
            lr = float(lr[0]) if lr else 5.25e-4
        else:
            lr = float(lr)
        wd = opt_cfg.get('weight_decay', 0.04)
        if isinstance(wd, (list, tuple)):
            wd = float(wd[0]) if wd else 0.04
        else:
            wd = float(wd)
        betas = opt_cfg.get('betas', [0.9, 0.999])
        if isinstance(betas, (list, tuple)):
            betas = tuple(float(b) for b in betas)
        else:
            betas = (0.9, 0.999)
        eps = opt_cfg.get('eps', 1e-8)
        if isinstance(eps, (list, tuple)):
            eps = float(eps[0]) if eps else 1e-8
        else:
            eps = float(eps)
        warmup_steps = opt_cfg.get('warmup_steps', 0)
        if isinstance(warmup_steps, (list, tuple)):
            warmup_steps = int(warmup_steps[0]) if warmup_steps else 0
        else:
            warmup_steps = int(warmup_steps)

        # Only encoder and predictor are trained
        trainable = nn.ModuleList([self.encoder, self.predictor])
        optimizer = create_schedule_free_optimizer(
            trainable,
            optimizer_type=opt_cfg.get('type', 'adamw'),
            lr=lr,
            weight_decay=wd,
            betas=betas,
            eps=eps,
            warmup_steps=warmup_steps,
        )
        self._optimizer = optimizer
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Save encoder/predictor/target_encoder at top level for eval and resume compatibility
        # (same keys as original vjepa2 app/vjepa/train.py save_checkpoint)
        checkpoint['encoder'] = self.encoder.state_dict()
        checkpoint['predictor'] = self.predictor.state_dict()
        checkpoint['target_encoder'] = self.target_encoder.state_dict()
        if self._optimizer is not None:
            checkpoint['opt'] = self._optimizer.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        if 'predictor' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor'], strict=True)
        if 'target_encoder' in checkpoint:
            self.target_encoder.load_state_dict(checkpoint['target_encoder'], strict=True)
