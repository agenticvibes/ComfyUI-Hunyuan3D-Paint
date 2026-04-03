"""Hybrid UNet wrapper: PyTorch interface, MLX computation.

Replaces the PyTorch UNet's forward method with a wrapper that:
1. Converts PyTorch MPS tensors → numpy → MLX arrays
2. Runs the MLX UNet forward pass
3. Converts MLX output → numpy → PyTorch tensor

This is a drop-in replacement for the existing UNet2p5DConditionModel
within the diffusers HunyuanPaintPipeline.
"""

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import mlx.core as mx

from .unet import HunyuanUNet2p5D


class HybridMLXUNet:
    """Wraps MLX UNet to match the PyTorch UNet2p5DConditionModel interface.

    After calling `patch_pipeline(pipeline)`, the diffusers pipeline
    will use MLX for all UNet forward passes while keeping everything
    else (VAE, text encoder, scheduler) in PyTorch.
    """

    def __init__(self, model_path: str, weights_path: Optional[str] = None):
        """Load the MLX UNet from converted weights.

        Args:
            model_path: Path to hunyuan3d-paintpbr-v2-1/unet directory
            weights_path: Path to directory containing unet.npz (pre-converted MLX weights)
        """
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(model_path), "mlx_weights")

        unet_npz = os.path.join(weights_path, "unet.npz")
        if not os.path.exists(unet_npz):
            raise FileNotFoundError(
                f"[MLX] Pre-converted MLX UNet weights not found: {unet_npz}\n"
                "Download from https://huggingface.co/AgenticVibes/hunyuan3d-2.1-mlx "
                "and place in ComfyUI/models/diffusers/hunyuan3d-mlx-weights/\n"
                "Or convert manually: python -m hy3dpaint.mlx.convert_weights --model-path <paint-model-path>"
            )

        # Create MLX UNet
        self.mlx_unet = HunyuanUNet2p5D(
            pbr_settings=["albedo", "mr"],
            cross_attention_dim=1024,
            out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=(2, 2, 2, 2),
            transformer_layers_per_block=(1, 1, 1, 1),
            num_attention_heads=(5, 10, 20, 20),
            norm_num_groups=32,
        )

        # Load weights
        print("[MLX] Loading UNet weights...")
        t0 = time.time()
        raw = dict(np.load(unet_npz, allow_pickle=True))
        self.mlx_unet.load_weights([(k, mx.array(v)) for k, v in raw.items()])
        print(f"[MLX] Loaded {len(raw)} weights in {time.time()-t0:.1f}s")
        del raw

        self._cache = {}
        self._call_count = 0

    def _to_np(self, t):
        """Convert PyTorch tensor to numpy."""
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return t

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        *args,
        return_dict=False,
        **kwargs,
    ):
        """Drop-in replacement for UNet2p5DConditionModel.forward().

        Accepts PyTorch tensors, runs MLX, returns PyTorch tensors.
        """
        self._call_count += 1
        device = sample.device
        dtype = sample.dtype

        # Extract kwargs that the MLX UNet needs
        embeds_normal = kwargs.get("embeds_normal")
        embeds_position = kwargs.get("embeds_position")
        ref_latents = kwargs.get("ref_latents")
        dino_hidden_states = kwargs.get("dino_hidden_states")
        num_in_batch = kwargs.get("num_in_batch", 6)
        mva_scale = kwargs.get("mva_scale", 1.0)
        ref_scale = kwargs.get("ref_scale", 1.0)

        # Scales can be float or tensor — convert to float or mx.array
        def _scale_to_mlx(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return float(v.item())
                # Per-batch tensor: convert to mx.array [B]
                return mx.array(v.detach().cpu().float().numpy())
            return float(v)

        mva_scale = _scale_to_mlx(mva_scale)
        ref_scale = _scale_to_mlx(ref_scale)

        # Convert to numpy
        sample_np = self._to_np(sample)
        enc_np = self._to_np(encoder_hidden_states)
        t_val = float(timestep) if isinstance(timestep, (int, float)) else float(timestep.item()) if timestep.dim() == 0 else float(timestep[0].item())

        # Convert to MLX
        t0 = time.time()
        output = self.mlx_unet(
            mx.array(sample_np),
            mx.array(t_val),
            mx.array(enc_np),
            embeds_normal=mx.array(self._to_np(embeds_normal)) if embeds_normal is not None else None,
            embeds_position=mx.array(self._to_np(embeds_position)) if embeds_position is not None else None,
            ref_latents=mx.array(self._to_np(ref_latents)) if ref_latents is not None else None,
            dino_hidden_states=mx.array(self._to_np(dino_hidden_states)) if dino_hidden_states is not None else None,
            mva_scale=mva_scale,
            ref_scale=ref_scale,
            num_in_batch=num_in_batch,
            cache=self._cache,
        )
        mx.eval(output)
        elapsed = time.time() - t0

        if self._call_count <= 5 or self._call_count % 5 == 0:
            print(f"[MLX] UNet step {self._call_count}: {elapsed:.2f}s")

        # Convert back to PyTorch
        output_np = np.array(output)
        output_pt = torch.from_numpy(output_np).to(dtype=dtype, device=device)

        if return_dict:
            return {"sample": output_pt}
        return (output_pt,)

    @staticmethod
    def patch_pipeline(pipeline, model_path: str, weights_path: Optional[str] = None):
        """Patch an existing diffusers HunyuanPaintPipeline to use MLX UNet.

        Args:
            pipeline: The loaded HunyuanPaintPipeline instance
            model_path: Path to the model directory (containing unet/)
            weights_path: Optional path to pre-converted MLX weights

        Returns:
            The HybridMLXUNet instance (for reference)
        """
        unet_path = os.path.join(model_path, "unet")
        hybrid = HybridMLXUNet(unet_path, weights_path)

        # Save original forward and replace
        original_forward = pipeline.unet.forward
        pipeline.unet._original_forward = original_forward
        pipeline.unet.forward = hybrid.forward

        # Also patch __call__ since diffusers sometimes uses it directly
        pipeline.unet._mlx_hybrid = hybrid

        print("[MLX] Pipeline patched — UNet forward will use MLX")
        return hybrid
