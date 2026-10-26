import torch
import torch.nn as nn 
import torch.cuda.amp as amp


from src.core import register
import src.misc.dist as dist 


__all__ = ['GradScaler']


@register
class GradScaler(amp.grad_scaler.GradScaler):
    def __init__(self, init_scale=65536.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
