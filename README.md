# ComfyUI-Hunyuan3d-Paint

ComfyUI nodes for generating PBR textures on 3D meshes using Tencent's Hunyuan3D-2.1 multiview diffusion model. Supports **MPS** (Mac) and **MLX** (Apple Silicon, ~3-5x faster) backends. Powered by Tencent Hunyuan.

This is the **texture painting** half of the Hunyuan3D pipeline. For mesh generation, see [ComfyUI-Hunyuan3d-Shape](https://github.com/agenticvibes/ComfyUI-Hunyuan3d-Shape).

## Nodes

| Node | Description |
|---|---|
| **MultiViews Generator** | Generate multi-view PBR textures (albedo + metallic-roughness) |
| **Bake MultiViews** | Bake multi-view images into UV texture maps |
| **InPaint** | Inpaint baked textures and export final GLB with PBR materials |
| **Camera Config** | Configure camera angles and weights for multi-view rendering |
| **Use MultiViews** | Create pipeline from pre-generated multiview images |
| **Use MultiViews From MetaData** | Load multiviews from exported metadata JSON |
| **MultiViews Generator With MetaData** | Generate multiviews and save with metadata |
| **Bake MultiViews With MetaData** | Bake textures, inpaint, and save with metadata |
| **HighPoly to LowPoly** | Create LOD meshes from high-poly using saved metadata |
| **MultiViews Generator Batch** | Batch texture generation for folders of meshes |

## Typical Workflow

```
TRIMESH → Camera Config → MultiViews Generator → Bake MultiViews → InPaint → GLB
```

Input meshes can come from [ComfyUI-Hunyuan3d-Shape](https://github.com/agenticvibes/ComfyUI-Hunyuan3d-Shape), [Trellis](https://github.com/microsoft/TRELLIS), or any source that outputs a `TRIMESH`. Use [ComfyUI-MeshTools](https://github.com/agenticvibes/ComfyUI-MeshTools) for UV unwrapping and mesh post-processing.

## Installation

### 1. Clone or copy into ComfyUI

```
ComfyUI/custom_nodes/ComfyUI-Hunyuan3d-Paint/
```

### 2. Install dependencies

The `install.py` runs automatically when ComfyUI loads the node pack. It installs:
- Python dependencies from `requirements.txt`
- C++ rasterizer extensions (precompiled wheels or source build)
- MLX on macOS Apple Silicon (optional)

To install manually:

```bash
pip install -r requirements.txt

# C++ extensions (required for texture baking)
cd hy3dpaint/custom_rasterizer && pip install --no-build-isolation .
cd hy3dpaint/DifferentiableRenderer && pip install --no-build-isolation .

# Optional: MLX for Apple Silicon acceleration
pip install mlx
```

### 3. Download models

You need the **paint model** plus **DINOv2** for image conditioning. Choose either PyTorch OR MLX weights for the paint model — you don't need both.

#### Required: Paint model (choose one)

**Option A — PyTorch (MPS/CUDA):**

Download `hunyuan3d-paintpbr-v2-1/` from [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) and place in:

```
ComfyUI/models/diffusers/
└── hunyuan3d-paintpbr-v2-1/      ← entire directory (~3.7 GB)
    ├── unet/
    ├── vae/
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── feature_extractor/
    ├── image_encoder/
    └── model_index.json
```

**Option B — MLX (Apple Silicon, ~3-5x faster):**

Download from [AgenticVibes/hunyuan3d-2.1-mlx](https://huggingface.co/AgenticVibes/hunyuan3d-2.1-mlx) and place in:

```
ComfyUI/models/diffusers/
└── hunyuan3d-mlx-weights/        ← (~4 GB)
    ├── unet.npz
    └── vae.npz
```

> Note: MLX backend still requires the PyTorch paint model directory for pipeline config files. Place both directories in `models/diffusers/`.

#### Required: DINOv2 image encoder

Download [facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant) and place in:

```
ComfyUI/models/clip_vision/
└── dinov2-giant/                  ← (~4.5 GB)
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

If not found locally, the model will be downloaded from HuggingFace automatically on first use.

#### Optional: RealESRGAN upscaler

For texture upscaling, download [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) and place in:

```
ComfyUI/models/upscale_models/
└── RealESRGAN_x4plus.pth         ← (~65 MB)
```

### Complete model directory structure

```
ComfyUI/models/
├── diffusers/
│   ├── hunyuan3d-paintpbr-v2-1/      (from tencent/Hunyuan3D-2.1)
│   └── hunyuan3d-mlx-weights/        (from AgenticVibes/hunyuan3d-2.1-mlx — Mac only)
├── clip_vision/
│   └── dinov2-giant/                  (from facebook/dinov2-giant)
└── upscale_models/
    └── RealESRGAN_x4plus.pth         (from xinntao/Real-ESRGAN — optional)
```

## Platform Support

| Platform | Backend | Notes |
|---|---|---|
| **CUDA** (Windows/Linux) | PyTorch | Full support, C++ rasterizer runs natively |
| **MPS** (macOS Apple Silicon) | PyTorch | Full support, chunked attention for large sequences, CPU fallback for rasterizer |
| **MLX** (macOS Apple Silicon) | MLX | ~3-5x faster UNet inference, select `mlx` in the `diffusion_backend` dropdown |

## License

This project contains code from Tencent's Hunyuan3D-2.1 and original MPS/MLX additions.

- **Tencent code**: [Tencent Hunyuan 3D 2.1 Community License Agreement](LICENSE) — non-commercial, territory restricted (excludes EU/UK/South Korea)
- **MPS/MLX additions**: [MIT License](LICENSE-MIT)
- **Apple MLX base code** (`hy3dpaint/mlx/base/`): MIT License — Copyright © 2023 Apple Inc.

**Important restrictions:**
- **Non-commercial use only** (Tencent license)
- **Territory restricted** — excludes European Union, United Kingdom, and South Korea
- See [LICENSE](LICENSE) for full terms

## Acknowledgements

- [Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — Hunyuan3D-2.1 model and original code
- [visualbruno](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1) — Original ComfyUI wrapper
- [kijai](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper) — Original Hunyuan3D v2.0 wrapper
- [Apple](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion) — MLX stable diffusion base code
