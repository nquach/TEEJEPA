"""
Schedule-Free Optimizer Integration

This module provides integration for schedule-free optimizers (AdamWScheduleFree and
RAdamScheduleFree) from the schedule-free package for use with PyTorch Lightning.
"""

try:
    from schedulefree import AdamWScheduleFree, RAdamScheduleFree
    SCHEDULE_FREE_AVAILABLE = True
    __all__ = [
        'AdamWScheduleFree',
        'RAdamScheduleFree',
        'SCHEDULE_FREE_AVAILABLE',
        'create_schedule_free_optimizer',
    ]
except ImportError:
    SCHEDULE_FREE_AVAILABLE = False
    AdamWScheduleFree = None
    RAdamScheduleFree = None
    print("Warning: schedule-free package not found. Install with: pip install schedule-free")
    __all__ = ['SCHEDULE_FREE_AVAILABLE', 'create_schedule_free_optimizer']


def create_schedule_free_optimizer(
    model,
    optimizer_type='adamw',
    lr=1.5e-4,
    weight_decay=0.05,
    betas=(0.9, 0.95),
    eps=1e-8,
    warmup_steps=0
):
    """
    Create a schedule-free optimizer for the model.

    Supported optimizers:
    - 'adamw': AdamWScheduleFree (default) - supports warmup_steps
    - 'radam': RAdamScheduleFree

    Args:
        model (torch.nn.Module): Model to optimize (encoder + predictor parameters)
        optimizer_type (str): Type of optimizer - 'adamw' or 'radam'
        lr (float): Learning rate
        weight_decay (float): Weight decay coefficient
        betas (tuple): Beta parameters for Adam-based optimizers
        eps (float): Epsilon for numerical stability
        warmup_steps (int): Number of warmup steps for AdamWScheduleFree (only for 'adamw')

    Returns:
        torch.optim.Optimizer: Schedule-free optimizer instance
    """
    if not SCHEDULE_FREE_AVAILABLE:
        raise ImportError(
            "schedule-free package is required but not installed. "
            "Install with: pip install schedule-free"
        )

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

    parameters = list(model.parameters())
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adamw':
        optimizer = AdamWScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            warmup_steps=warmup_steps
        )
        if warmup_steps > 0:
            print(f"AdamWScheduleFree initialized with {warmup_steps} warmup steps")
    elif optimizer_type == 'radam':
        if warmup_steps > 0:
            print(
                "Warning: warmup_steps specified but RAdamScheduleFree "
                "does not support warmup. Ignoring warmup_steps parameter."
            )
        optimizer = RAdamScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    else:
        raise ValueError(
            f"Unsupported optimizer_type: {optimizer_type}. "
            "Must be 'adamw' or 'radam'"
        )

    return optimizer
