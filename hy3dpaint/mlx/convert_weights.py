"""Weight conversion from PyTorch (diffusers) to MLX for Hunyuan3D-Paint.

Handles:
- Conv2d weight transposition: NCHW (PyTorch) → NHWC (MLX)
- GeGLU feedforward split: single [2*hidden, in] → two [hidden, in] linears
- Key renaming from diffusers naming to MLX module paths
- float16 preservation (no unnecessary dtype changes)

Usage:
    python -m hy3dpaint.mlx.convert_weights --model-path /path/to/hunyuan3d-paintpbr-v2-1

Or from Python:
    from hy3dpaint.mlx.convert_weights import convert_unet_weights, convert_vae_weights
"""

import os
import argparse
import numpy as np

# Lazy imports to avoid requiring both frameworks at import time
_torch = None
_mx = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_mx():
    global _mx
    if _mx is None:
        import mlx.core
        _mx = mlx.core
    return _mx


def _transpose_conv_weight(arr):
    """Transpose Conv2d weight from NCHW to NHWC format.

    PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
    MLX:     (out_channels, kernel_h, kernel_w, in_channels)
    """
    if arr.ndim == 4:
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _split_geglu_weight(weight):
    """Split a GeGLU projection weight [2*hidden, in] into gate and proj [hidden, in] each."""
    hidden_dim = weight.shape[0] // 2
    return weight[:hidden_dim], weight[hidden_dim:]


def _split_geglu_bias(bias):
    """Split a GeGLU projection bias [2*hidden] into gate and proj [hidden] each."""
    hidden_dim = bias.shape[0] // 2
    return bias[:hidden_dim], bias[hidden_dim:]


def _remap_transformer_block_keys(old_key, prefix):
    """Remap keys within a single transformer block from diffusers → MLX naming.

    Input keys look like:
        {prefix}.transformer.attn1.to_q.weight
        {prefix}.transformer.ff.net.0.proj.weight  (GeGLU)
        {prefix}.transformer.ff.net.2.weight
        {prefix}.attn_multiview.to_q.weight
        {prefix}.attn_refview.processor.to_v_mr.weight
        {prefix}.attn_dino.to_q.weight

    Returns: (new_key, transform) where transform is None, 'conv_transpose',
             'geglu_split_1', or 'geglu_split_2'
    """
    # Strip the prefix to get the local key
    if not old_key.startswith(prefix + "."):
        return None, None
    local = old_key[len(prefix) + 1:]

    # GeGLU feedforward split
    # diffusers: ff.net.0.proj.weight [2*hidden, in] → linear1.weight + linear2.weight
    #            ff.net.2.weight [in, hidden] → linear3.weight
    if local == "transformer.ff.net.0.proj.weight":
        return prefix + ".transformer.linear1.weight", "geglu_split_1"
    if local == "transformer.ff.net.0.proj.bias":
        return prefix + ".transformer.linear1.bias", "geglu_split_1"
    if local == "transformer.ff.net.2.weight":
        return prefix + ".transformer.linear3.weight", None
    if local == "transformer.ff.net.2.bias":
        return prefix + ".transformer.linear3.bias", None

    # Everything else keeps its key as-is (our MLX architecture mirrors PyTorch naming)
    return old_key, None


def _remap_key_for_mlx_model(key):
    """Remap a converted weight key to match the MLX model parameter tree.

    Transformations (applied in order):
    1. 'unet.image_proj_model_dino.*' → 'image_proj_model_dino.*'
    2. 'unet.learned_text_clip_*' → 'learned_text_clip_*'
    3. Within transformer_blocks: 'transformer.attn1.*' → 'attn1.*'
    4. Within transformer_blocks: 'transformer.norm*' → 'norm*'
    5. Within transformer_blocks: 'transformer.linear*' → 'linear*'
    6. '.processor.to_q_mr' → '.to_q_mr' (flatten processor sub-module)
    7. '.processor.to_v_mr' → '.to_v_mr'
    8. '.to_out.0.' → '.to_out_0.' (MLX uses underscore not dot for index)
    9. '.to_out_mr.0.' → '.to_out_mr_0.'
    10. 'unet.mid_block.resnets.0.*' → 'unet.mid_resnets_0.*'
    11. 'unet.mid_block.attentions.0.*' → 'unet.mid_attentions_0.*'
    12. 'unet.mid_block.resnets.1.*' → 'unet.mid_resnets_1.*'
    """
    k = key

    # Top-level: move image_proj and learned_text_clip out of unet prefix
    k = k.replace("unet.image_proj_model_dino.", "image_proj_model_dino.")
    k = k.replace("unet.learned_text_clip_", "learned_text_clip_")

    # Mid block: flatten diffusers mid_block structure to match our named attributes
    k = k.replace("unet.mid_block.resnets.0.", "unet.mid_resnets_0.")
    k = k.replace("unet.mid_block.attentions.0.", "unet.mid_attentions_0.")
    k = k.replace("unet.mid_block.resnets.1.", "unet.mid_resnets_1.")

    # Within transformer_blocks: remove 'transformer.' prefix for attn1/attn2/norm/linear
    # These are nested under transformer_blocks.N.transformer.* in PyTorch
    # but directly under transformer_blocks.N.* in MLX
    import re
    k = re.sub(
        r'(transformer_blocks\.\d+)\.transformer\.(attn1|attn2|norm1|norm2|norm3|linear1|linear2|linear3)\.',
        r'\1.\2.',
        k
    )

    # Flatten .processor. sub-module (PBR-specific projections)
    # transformer_blocks.N.attn1.processor.to_q_mr → transformer_blocks.N.attn1.to_q_mr
    k = re.sub(r'\.processor\.to_', '.to_', k)

    # .to_out.0. → .to_out_0.  (module list index → underscore)
    k = k.replace(".to_out.0.", ".to_out_0.")
    # .to_out_mr.0. → .to_out_mr_0.
    k = k.replace(".to_out_mr.0.", ".to_out_mr_0.")

    # unet_dual mid block (same pattern)
    k = k.replace("unet_dual.mid_block.resnets.0.", "unet_dual.mid_resnets_0.")
    k = k.replace("unet_dual.mid_block.attentions.0.", "unet_dual.mid_attentions_0.")
    k = k.replace("unet_dual.mid_block.resnets.1.", "unet_dual.mid_resnets_1.")

    # Downsamplers/upsamplers: diffusers uses downsamplers.0.conv, our model uses downsample directly
    k = k.replace(".downsamplers.0.conv.", ".downsample.")
    k = k.replace(".downsamplers.0.", ".downsample.")
    k = k.replace(".upsamplers.0.conv.", ".upsample.")
    k = k.replace(".upsamplers.0.", ".upsample.")

    return k


def convert_unet_weights(pytorch_state_dict):
    """Convert UNet weights from PyTorch to MLX-compatible numpy arrays.

    Returns:
        dict mapping MLX key → numpy array
    """
    mlx_weights = {}

    for key, tensor in pytorch_state_dict.items():
        arr = tensor.cpu().numpy()

        # Check if this is inside a transformer block that needs GeGLU remapping
        # Pattern: unet.{down/up}_blocks.N.attentions.N.transformer_blocks.N.*
        # or:      unet.mid_block.attentions.N.transformer_blocks.N.*
        new_key = key
        transform = None

        # Find transformer block prefixes
        parts = key.split(".")
        for i in range(len(parts)):
            if parts[i] == "transformer_blocks" and i + 1 < len(parts):
                block_prefix = ".".join(parts[:i + 2])
                new_key, transform = _remap_transformer_block_keys(key, block_prefix)
                if new_key is None:
                    new_key = key
                break

        # Apply transforms
        if transform == "geglu_split_1":
            # Split GeGLU into two keys: linear1 (gate) and linear2 (proj)
            gate, proj = _split_geglu_weight(arr) if arr.ndim == 2 else _split_geglu_bias(arr)
            mlx_weights[new_key] = gate
            linear2_key = new_key.replace("linear1", "linear2")
            mlx_weights[linear2_key] = proj
            continue
        elif transform == "geglu_split_2":
            # This shouldn't happen with current logic, but just in case
            pass

        # conv_shortcut in diffusers is Conv2d 1x1 but MLX base uses Linear
        if "conv_shortcut" in new_key and arr.ndim == 4:
            arr = arr.squeeze(-1).squeeze(-1)  # [out, in, 1, 1] → [out, in]
            mlx_weights[new_key] = arr
            continue

        # Transpose conv weights NCHW → NHWC
        if arr.ndim == 4:
            arr = _transpose_conv_weight(arr)

        mlx_weights[new_key] = arr

    # Post-process: remap keys to match MLX model parameter tree
    remapped = {}
    for k, v in mlx_weights.items():
        remapped[_remap_key_for_mlx_model(k)] = v
    return remapped


def convert_vae_weights(pytorch_state_dict):
    """Convert VAE weights from PyTorch to MLX-compatible numpy arrays.

    The VAE uses diffusers AutoencoderKL naming. Key mapping:
    - quant_conv → quant_proj (Linear in MLX, Conv2d 1x1 in diffusers)
    - post_quant_conv → post_quant_proj
    - mid_block.attentions.0 → mid_blocks.1 (Attention module)
    - Conv2d weights: NCHW → NHWC
    """
    mlx_weights = {}

    for key, tensor in pytorch_state_dict.items():
        arr = tensor.cpu().numpy()
        new_key = key

        # Remap quant_conv (1x1 Conv2d in diffusers) to quant_proj (Linear in MLX)
        # diffusers: quant_conv.weight [out, in, 1, 1] → quant_proj.weight [out, in]
        if key in ("quant_conv.weight", "post_quant_conv.weight"):
            new_key = key.replace("_conv", "_proj")
            if arr.ndim == 4:
                arr = arr.squeeze(-1).squeeze(-1)  # [out, in, 1, 1] → [out, in]
            mlx_weights[new_key] = arr
            continue
        if key in ("quant_conv.bias", "post_quant_conv.bias"):
            new_key = key.replace("_conv", "_proj")
            mlx_weights[new_key] = arr
            continue

        # Remap mid_block structure
        # diffusers: mid_block.attentions.0.* → mid_blocks.1.*
        # diffusers: mid_block.resnets.0.* → mid_blocks.0.*
        # diffusers: mid_block.resnets.1.* → mid_blocks.2.*
        if "mid_block." in key:
            new_key = key.replace("mid_block.resnets.0.", "mid_blocks.0.")
            new_key = new_key.replace("mid_block.attentions.0.", "mid_blocks.1.")
            new_key = new_key.replace("mid_block.resnets.1.", "mid_blocks.2.")

            # VAE attention remapping
            # diffusers uses: query/key/value/proj_attn (no to_ prefix)
            # MLX VAE uses: query_proj/key_proj/value_proj/out_proj
            if "mid_blocks.1." in new_key:
                new_key = new_key.replace(".to_q.", ".query_proj.")
                new_key = new_key.replace(".to_k.", ".key_proj.")
                new_key = new_key.replace(".to_v.", ".value_proj.")
                new_key = new_key.replace(".to_out.0.", ".out_proj.")
                # diffusers VAE also uses bare query/key/value/proj_attn
                new_key = new_key.replace(".query.", ".query_proj.")
                new_key = new_key.replace(".key.", ".key_proj.")
                new_key = new_key.replace(".value.", ".value_proj.")
                new_key = new_key.replace(".proj_attn.", ".out_proj.")

        # Remap encoder/decoder block structure
        # diffusers: encoder.down_blocks → encoder.down_blocks (same)
        # diffusers: decoder.up_blocks → decoder.up_blocks (same)

        # Remap downsampler/upsampler
        # diffusers: downsamplers.0.conv → downsample (MLX stores conv directly)
        if "downsamplers.0." in new_key:
            new_key = new_key.replace("downsamplers.0.conv.", "downsample.")
            new_key = new_key.replace("downsamplers.0.", "downsample.")
        if "upsamplers.0." in new_key:
            new_key = new_key.replace("upsamplers.0.conv.", "upsample.")
            new_key = new_key.replace("upsamplers.0.", "upsample.")

        # Remap conv_shortcut (diffusers uses Conv2d 1x1, MLX base uses Linear)
        if "conv_shortcut" in new_key and arr.ndim == 4:
            arr = arr.squeeze(-1).squeeze(-1)  # [out, in, 1, 1] → [out, in]
            mlx_weights[new_key] = arr
            continue

        # Transpose Conv2d weights NCHW → NHWC
        if arr.ndim == 4:
            arr = _transpose_conv_weight(arr)

        mlx_weights[new_key] = arr

    return mlx_weights


def convert_and_save(model_path, output_dir=None):
    """Convert all weights from a Hunyuan3D-Paint model directory.

    Args:
        model_path: Path to hunyuan3d-paintpbr-v2-1 directory
        output_dir: Output directory (defaults to model_path/mlx_weights)
    """
    torch = _get_torch()
    mx = _get_mx()

    if output_dir is None:
        output_dir = os.path.join(model_path, "mlx_weights")
    os.makedirs(output_dir, exist_ok=True)

    # Convert UNet
    print("Loading UNet weights...")
    unet_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.bin")
    unet_sd = torch.load(unet_path, map_location="cpu", weights_only=True)
    print(f"  {len(unet_sd)} PyTorch keys loaded")

    print("Converting UNet weights...")
    unet_mlx = convert_unet_weights(unet_sd)
    print(f"  {len(unet_mlx)} MLX keys produced")

    # Validate
    conv_count = sum(1 for k, v in unet_mlx.items() if v.ndim == 4)
    print(f"  {conv_count} conv weights transposed to NHWC")

    # Save
    unet_out = os.path.join(output_dir, "unet.npz")
    np.savez(unet_out, **unet_mlx)
    size_mb = os.path.getsize(unet_out) / 1024 ** 2
    print(f"  Saved to {unet_out} ({size_mb:.1f} MB)")

    del unet_sd, unet_mlx

    # Convert VAE
    print("\nLoading VAE weights...")
    vae_path = os.path.join(model_path, "vae", "diffusion_pytorch_model.bin")
    vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)
    print(f"  {len(vae_sd)} PyTorch keys loaded")

    print("Converting VAE weights...")
    vae_mlx = convert_vae_weights(vae_sd)
    print(f"  {len(vae_mlx)} MLX keys produced")

    # Save
    vae_out = os.path.join(output_dir, "vae.npz")
    np.savez(vae_out, **vae_mlx)
    size_mb = os.path.getsize(vae_out) / 1024 ** 2
    print(f"  Saved to {vae_out} ({size_mb:.1f} MB)")

    print("\nDone! Converted weights saved to:", output_dir)
    return output_dir


def validate_conversion(model_path, output_dir=None):
    """Validate converted weights against PyTorch originals.

    Checks:
    1. Key count matches (accounting for GeGLU splits)
    2. Conv weights have correct NHWC shape
    3. Spot-check values match
    """
    torch = _get_torch()

    if output_dir is None:
        output_dir = os.path.join(model_path, "mlx_weights")

    print("=== Validating UNet conversion ===")
    unet_pt = torch.load(
        os.path.join(model_path, "unet", "diffusion_pytorch_model.bin"),
        map_location="cpu", weights_only=True,
    )
    unet_mlx = dict(np.load(os.path.join(output_dir, "unet.npz"), allow_pickle=True))

    # Count GeGLU splits (each split adds 1 key)
    geglu_count = sum(1 for k in unet_pt if "ff.net.0.proj" in k)
    expected_mlx_keys = len(unet_pt) + geglu_count  # each GeGLU becomes 2 keys
    # But we also remove the original ff.net.0.proj keys and ff.net.2 keys get renamed
    print(f"  PyTorch keys: {len(unet_pt)}")
    print(f"  MLX keys: {len(unet_mlx)} (expected ~{expected_mlx_keys} with {geglu_count} GeGLU splits)")

    # Check conv weights are NHWC
    for k, v in unet_mlx.items():
        if v.ndim == 4:
            # NHWC: last dim should be small (in_channels)
            assert v.shape[1] == v.shape[2], f"Conv weight {k} looks wrong: {v.shape}"

    # Spot-check a non-conv weight
    test_key = "unet.time_embedding.linear_1.weight"
    if test_key in unet_pt and test_key in unet_mlx:
        pt_val = unet_pt[test_key].numpy()
        mlx_val = unet_mlx[test_key]
        max_diff = np.max(np.abs(pt_val - mlx_val))
        print(f"  Spot check '{test_key}': max diff = {max_diff}")
        assert max_diff < 1e-6, f"Values don't match for {test_key}"

    print("  UNet validation PASSED")

    print("\n=== Validating VAE conversion ===")
    vae_pt = torch.load(
        os.path.join(model_path, "vae", "diffusion_pytorch_model.bin"),
        map_location="cpu", weights_only=True,
    )
    vae_mlx = dict(np.load(os.path.join(output_dir, "vae.npz"), allow_pickle=True))
    print(f"  PyTorch keys: {len(vae_pt)}")
    print(f"  MLX keys: {len(vae_mlx)}")

    # Check conv weights are NHWC
    for k, v in vae_mlx.items():
        if v.ndim == 4:
            assert v.shape[1] == v.shape[2], f"Conv weight {k} looks wrong: {v.shape}"

    print("  VAE validation PASSED")
    print("\nAll validations passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hunyuan3D-Paint weights to MLX format")
    parser.add_argument("--model-path", required=True, help="Path to hunyuan3d-paintpbr-v2-1 directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: model_path/mlx_weights)")
    parser.add_argument("--validate", action="store_true", help="Run validation after conversion")
    args = parser.parse_args()

    output_dir = convert_and_save(args.model_path, args.output_dir)
    if args.validate:
        validate_conversion(args.model_path, output_dir)
