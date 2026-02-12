"""
V-JEPA Pretraining with PyTorch Lightning.

Loads YAML config, sets up litdata-optimized dataset, Lightning module (encoder + predictor
+ target encoder, schedule-free optimizer), and runs Trainer.fit().
"""

import os
import warnings

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning, module='torchvision')

from datasets import VideoDataModuleVJEPA
from lightning_module import VJEPALightningModule


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V-JEPA pretraining with PyTorch Lightning')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    seed = config.get('training', {}).get('seed', 0)
    pl.seed_everything(seed, workers=True)

    print("Creating data module...")
    data_module = VideoDataModuleVJEPA(config)
    data_module.setup('fit')
    print(f"Training dataset size: {len(data_module.train_dataset)}")

    # Compute default save_every_n_steps = steps_per_epoch_per_gpu if not set in config
    checkpoint_config = config.get('checkpoint', {})
    training_config = config.get('training', {})
    save_every_n_steps = checkpoint_config.get('save_every_n_steps')
    if save_every_n_steps is None:
        try:
            dataset_size = len(data_module.train_dataset)
            batch_size = training_config.get('batch_size', 24)
            num_gpus = max(1, torch.cuda.device_count())  # Use at least 1 to avoid division by zero
            if dataset_size > 0 and batch_size > 0:
                steps_per_epoch = dataset_size // batch_size
                steps_per_epoch_per_gpu = steps_per_epoch // num_gpus
                if steps_per_epoch_per_gpu > 0:
                    save_every_n_steps = steps_per_epoch_per_gpu
                    config['checkpoint']['save_every_n_steps'] = save_every_n_steps
                    print(f"save_every_n_steps not set in config, defaulting to steps_per_epoch_per_gpu={steps_per_epoch_per_gpu} (dataset_size={dataset_size}, batch_size={batch_size}, num_gpus={num_gpus})")
                else:
                    print(f"Warning: steps_per_epoch_per_gpu is 0 (steps_per_epoch={steps_per_epoch}, num_gpus={num_gpus}), skipping step-based checkpointing")
            else:
                print(f"Warning: Cannot compute steps_per_epoch (dataset_size={dataset_size}, batch_size={batch_size}), skipping step-based checkpointing")
        except (TypeError, AttributeError) as e:
            print(f"Warning: Dataset does not support len() or error accessing size: {e}. Skipping step-based checkpointing default.")

    mask_collator = data_module.get_mask_collator()
    print("Creating Lightning model...")
    model = VJEPALightningModule(config, mask_collator=mask_collator)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    checkpoint_config = config.get('checkpoint', {})
    checkpoint_enabled = checkpoint_config.get('enable', True)
    checkpoint_callback = None
    if checkpoint_enabled:
        ckpt_dir = checkpoint_config.get('dir', './output/checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        prefix = checkpoint_config.get('prefix', 'vjepa_')
        save_top_k = checkpoint_config.get('save_top_k', 1)
        strict_top_k = checkpoint_config.get('strict_top_k', False)
        save_last = False if strict_top_k else checkpoint_config.get('save_last', True)
        every_n_epochs = checkpoint_config.get('save_ckpt_freq')
        # save_every_n_steps already computed above (or read from config if set)
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_config.get('monitor', 'train_loss_epoch'),
            dirpath=ckpt_dir,
            filename=prefix + "-{epoch:02d}-{step:06d}-{train_loss_epoch:.4f}",
            mode='min',
            save_top_k=save_top_k,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=save_every_n_steps,
            save_on_train_epoch_end=True,
            save_last=save_last,
        )
        print(f"Checkpoint: dir={ckpt_dir}, prefix={prefix}, save_top_k={save_top_k}, save_last={save_last}, save_every_n_steps={save_every_n_steps}")

    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', './output/logs')
    os.makedirs(log_dir, exist_ok=True)
    logger_name = checkpoint_config.get('prefix', 'vjepa') if checkpoint_enabled else 'vjepa'
    logger = TensorBoardLogger(save_dir=log_dir, name=logger_name)

    callbacks = [c for c in [checkpoint_callback] if c is not None]

    # training_config already accessed above for save_every_n_steps computation
    dtype = (training_config.get('dtype') or 'float32').lower()
    if dtype == 'bfloat16':
        precision = 'bf16-mixed'
    elif dtype in ('float16', 'fp16', '16'):
        precision = '16-mixed'
    else:
        precision = '32'

    strategy = training_config.get('strategy', 'ddp')
    if torch.cuda.device_count() <= 1:
        strategy = 'auto'

    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 800),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy=strategy,
        callbacks=callbacks if callbacks else None,
        logger=logger,
        log_every_n_steps=logging_config.get('log_freq', 10),
        precision=precision,
        gradient_clip_val=training_config.get('gradient_clip_val', 0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
    )

    print("Starting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    print("Training completed.")
    if checkpoint_callback is not None:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
