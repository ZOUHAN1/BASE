"""Reusable fixed-shape CUDA Graph runner for deployed paired detectors."""

import torch


class CudaGraphRunner:
    """Capture model forward once and replay it with new same-shape CUDA inputs.

    The returned tensors are graph-owned buffers and are overwritten by the next
    replay. Consumers that retain an output must clone it first.
    """

    @torch.inference_mode()
    def __init__(self, model, example_image, example_mask, amp=True, warmups=100):
        if model.training:
            raise RuntimeError('CUDA Graph capture requires an eval-mode model')
        if not example_image.is_cuda or not example_mask.is_cuda:
            raise ValueError('CUDA Graph inputs must already be on CUDA')
        self.model = model
        self.amp = bool(amp)
        self.image = torch.empty_like(example_image)
        self.mask = torch.empty_like(example_mask)
        self.image.copy_(example_image)
        self.mask.copy_(example_mask)

        for _ in range(warmups):
            self._forward()
        torch.cuda.synchronize(example_image.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._forward()
        for _ in range(20):
            self.graph.replay()
        torch.cuda.synchronize(example_image.device)

    def _forward(self):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
            return self.model(self.image, pixel_valid=self.mask)

    @torch.inference_mode()
    def __call__(self, image, mask):
        if image.shape != self.image.shape or image.dtype != self.image.dtype:
            raise ValueError('Image shape/dtype differs from the captured input')
        if mask.shape != self.mask.shape or mask.dtype != self.mask.dtype:
            raise ValueError('Mask shape/dtype differs from the captured input')
        self.image.copy_(image)
        self.mask.copy_(mask)
        self.graph.replay()
        return self.output
