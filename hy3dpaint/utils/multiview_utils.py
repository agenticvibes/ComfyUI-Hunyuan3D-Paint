# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from ..hunyuanpaintpbr.pipeline import HunyuanPaintPipeline


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        model_path = config.paint_model_path
        if model_path is None:
            raise ValueError(
                "[Hunyuan3D] paint_model_path is required. "
                "Place the 'hunyuan3d-paintpbr-v2-1' directory in ComfyUI/models/diffusers/ "
                "and select it in the node."
            )
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"[Hunyuan3D] Paint model directory not found: {model_path}\n"
                "Download from https://huggingface.co/tencent/Hunyuan3D-2.1 "
                "and place in ComfyUI/models/diffusers/"
            )

        # Monkey-patch F.scaled_dot_product_attention on MPS to use chunked attention (avoids O(n²) OOM)
        if torch.device(self.device).type == "mps":
            from ..hunyuanpaintpbr.unet.attn_processor import _chunked_scaled_dot_product_attention
            _original_sdpa = torch.nn.functional.scaled_dot_product_attention

            # Only use chunked attention for large sequences that would OOM on MPS
            _MPS_CHUNK_THRESHOLD = 8192

            def _mps_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                if query.device.type == "mps" and query.shape[2] > _MPS_CHUNK_THRESHOLD:
                    return _chunked_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, chunk_size=1024)
                return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            torch.nn.functional.scaled_dot_product_attention = _mps_sdpa

        # Ensure local patched UNet modules are in the model directory
        # (required for MPS support and diffusers compatibility)
        import shutil
        local_unet_dir = os.path.join(os.path.dirname(__file__), "..", "hunyuanpaintpbr", "unet")
        model_unet_dir = os.path.join(model_path, "unet")
        for module_file in ["attn_processor.py", "modules.py"]:
            local_file = os.path.join(local_unet_dir, module_file)
            model_file = os.path.join(model_unet_dir, module_file)
            if os.path.exists(local_file) and os.path.exists(model_unet_dir):
                shutil.copy2(local_file, model_file)

        pipeline = HunyuanPaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )

        # Use DDIM scheduler matching training config (v_prediction + zero-SNR)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=False)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        if torch.device(self.device).type == "cuda":
            pipeline.enable_model_cpu_offload()
        self.pipeline = pipeline.to(self.device)

        # Optionally patch UNet with MLX backend for faster inference on Apple Silicon
        diffusion_backend = getattr(config, 'diffusion_backend', 'pytorch')
        if diffusion_backend == "mlx" and torch.device(self.device).type == "mps":
            mlx_weights_path = getattr(config, 'mlx_weights_path', None)
            from ..mlx.hybrid_unet import HybridMLXUNet
            self._mlx_hybrid = HybridMLXUNet.patch_pipeline(
                self.pipeline, model_path, weights_path=mlx_weights_path
            )
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from ..hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input, num_steps=num_steps, guidance_scale=guidance_scale, seed=seed
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, num_steps=10, guidance_scale=3.0, seed=0):
        self.seed_everything(seed)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        
        if not isinstance(input_images, List):
            input_images = [input_images]
            
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
            
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 10,
            "UniPCMultistepScheduler": 10,
            "DDIMScheduler": 10,
            "ShiftSNRScheduler": 10,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=num_steps,
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
