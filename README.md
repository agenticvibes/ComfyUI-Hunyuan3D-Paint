# ComfyUI-Hunyuan3D-Paint

ComfyUI nodes for generating PBR textures on 3D meshes using Tencent's Hunyuan3D-2.1 multiview diffusion model. Powered by Tencent Hunyuan.

Supports three backends:
- **CUDA** — Windows/Linux with NVIDIA GPU
- **MPS** — macOS Apple Silicon via PyTorch
- **MLX** — macOS Apple Silicon via Apple MLX (~3-5x faster than MPS)

On Mac, the node UI lets you choose between `pytorch` (MPS) and `mlx` backends depending on what's installed. MLX is installed automatically by `install.py` on Apple Silicon.

This is the **texture painting** package. For mesh generation, see [ComfyUI-Hunyuan3D-Shape](https://github.com/agenticvibes/ComfyUI-Hunyuan3D-Shape). For generic mesh tools, see [ComfyUI-MeshTools](https://github.com/agenticvibes/ComfyUI-MeshTools).

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

Input meshes can come from [ComfyUI-Hunyuan3D-Shape](https://github.com/agenticvibes/ComfyUI-Hunyuan3D-Shape), [Trellis](https://github.com/microsoft/TRELLIS), or any source that outputs a `TRIMESH`. Use [ComfyUI-MeshTools](https://github.com/agenticvibes/ComfyUI-MeshTools) for UV unwrapping and mesh post-processing.

## Installation

### 1. Clone into ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/agenticvibes/ComfyUI-Hunyuan3D-Paint.git
```

### 2. Install dependencies

The `install.py` runs automatically when ComfyUI loads the node pack. It installs:
- Python dependencies from `requirements.txt`
- C++ rasterizer extensions (precompiled wheels or source build)
- MLX on macOS Apple Silicon (for the faster MLX backend)

To install manually instead:

```bash
pip install -r requirements.txt

# C++ extensions (required for texture baking)
cd hy3dpaint/custom_rasterizer && pip install --no-build-isolation .
cd hy3dpaint/DifferentiableRenderer && pip install --no-build-isolation .

# MLX for Apple Silicon (optional when installing manually — enables the mlx backend)
pip install mlx
```

### 3. Download models

Which models you need depends on your platform:

| Model | CUDA (Windows/Linux) | MPS (Mac PyTorch) | MLX (Mac fast) |
|---|---|---|---|
| `hunyuan3d-paintpbr-v2-1/` | Required | Required | Required (for config files) |
| `hunyuan3d-mlx-weights/` | Not needed | Not needed | Required |
| `dinov2-giant/` | Required | Required | Required |
| `RealESRGAN_x4plus.pth` | Optional | Optional | Optional |

#### Paint model

**For CUDA or MPS (PyTorch):** Download `hunyuan3d-paintpbr-v2-1/` from [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1):

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

**For MLX (Apple Silicon):** Download from [AgenticVibes/hunyuan3d-2.1-mlx](https://huggingface.co/AgenticVibes/hunyuan3d-2.1-mlx):

```
ComfyUI/models/diffusers/
└── hunyuan3d-mlx-weights/        ← (~4 GB)
    ├── unet.npz
    └── vae.npz
```

> **Note:** The MLX backend still requires the PyTorch paint model directory for pipeline config files (scheduler, tokenizer, etc.). If using MLX, place both directories in `models/diffusers/`.

#### DINOv2 image encoder

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

| Platform | Backend | How to select | Notes |
|---|---|---|---|
| **CUDA** (Windows/Linux) | PyTorch | `pytorch` (default) | C++ rasterizer runs natively on GPU |
| **MPS** (macOS Apple Silicon) | PyTorch | `pytorch` (default) | Chunked attention for large sequences, CPU fallback for rasterizer |
| **MLX** (macOS Apple Silicon) | MLX | `mlx` in dropdown | ~3-5x faster UNet inference. Requires MLX installed + MLX weights downloaded |

On Mac, the `diffusion_backend` dropdown in the MultiViews Generator node shows available backends. `mlx` only appears when the `mlx` package is installed.

## License

This project contains code from Tencent's Hunyuan3D-2.1 and original MPS/MLX additions.

- **Tencent code**: [Tencent Hunyuan 3D 2.1 Community License Agreement](LICENSE) — non-commercial, territory restricted (excludes EU/UK/South Korea)
- **MPS/MLX port**: [MIT License](LICENSE-MIT) — Copyright (c) 2025 agenticvibes
- **Apple MLX base code** (`hy3dpaint/mlx/base/`): MIT License — Copyright (c) 2023 Apple Inc.

**Important restrictions:**
- **Non-commercial use only** (Tencent license)
- **Territory restricted** — excludes European Union, United Kingdom, and South Korea
- See [LICENSE](LICENSE) for full terms

## Acknowledgements

- [Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — Hunyuan3D-2.1 model and original code
- [visualbruno](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1) — Original ComfyUI wrapper
- [kijai](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper) — Original Hunyuan3D v2.0 wrapper
- [Apple](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion) — MLX stable diffusion base code
