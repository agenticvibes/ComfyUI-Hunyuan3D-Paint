# ComfyUI-Hunyuan3D-Paint Node Reference

## MultiViews Generator

The core texture generation node. Renders the mesh from multiple camera angles, then uses multiview diffusion to generate PBR textures (albedo + metallic-roughness) that are consistent across all views.

Supports three backends: CUDA (Windows/Linux), MPS (Mac PyTorch), and MLX (Mac Apple Silicon, ~3-5x faster).

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trimesh` | TRIMESH | - | 3D mesh to paint textures on. Should have UV coordinates (use MeshTools UV Unwrap first, or enable `unwrap_mesh`). |
| `camera_config` | HY3D21CAMERA | - | Camera angles and weights from the Camera Config node. Determines which viewpoints are rendered and how they're weighted during baking. |
| `paint_model` | ENUM | - | Paint model directory from `ComfyUI/models/diffusers/`. Select the `hunyuan3d-paintpbr-v2-1` directory. |
| `view_size` | INT | `512` | Resolution of each rendered view in pixels. **512:** standard quality, faster. **1024:** higher quality views, more VRAM. The diffusion model processes each view at this resolution. |
| `image` | IMAGE | - | Reference image that guides texture style, color, and material appearance. The diffusion model uses this to understand what the object should look like. |
| `steps` | INT | `10` | Number of diffusion denoising steps for texture generation. Unlike shape generation (which needs 50), texture diffusion converges quickly. **10:** good default. **15-20:** slightly better quality. **5:** fast preview. |
| `guidance_scale` | FLOAT | `3.0` | How closely the generated textures follow the reference image. **Low (1-2):** more creative, may not match reference colors. **3.0:** good balance. **High (5+):** very strict, may produce artifacts. |
| `texture_size` | INT | `1024` | Final output texture resolution in pixels. **512:** low-res preview. **1024:** standard quality. **2048:** high detail. **4096:** maximum detail, high memory. |
| `unwrap_mesh` | BOOLEAN | `True` | Automatically UV unwrap the mesh using xatlas before texturing. Disable if your mesh already has good UV coordinates from an earlier step or external tool. |
| `seed` | INT | `0` | Random seed for reproducible texture generation. Same seed + same inputs = same textures. |
| `diffusion_backend` | ENUM | `"pytorch"` | **pytorch:** standard backend, works on CUDA and MPS. **mlx:** Apple MLX backend, ~3-5x faster on Apple Silicon. Only shown when MLX is installed. Requires MLX weights downloaded separately. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `pipeline` | HY3DPIPELINE | The paint pipeline with loaded mesh and renderer. Pass to Bake MultiViews. |
| `albedo` | IMAGE | Multi-view albedo (color) images — one per camera angle. |
| `mr` | IMAGE | Multi-view metallic-roughness images — one per camera angle. |
| `positions` | IMAGE | Rendered position maps for each view (used internally for baking). |
| `normals` | IMAGE | Rendered normal maps for each view. |
| `camera_config` | HY3D21CAMERA | Pass-through of camera configuration. |
| `metadata` | HY3D21METADATA | Generation metadata for saving/reloading. |

---

## Bake MultiViews

Project the multi-view texture images back onto the 3D mesh's UV space. This takes the per-view albedo and metallic-roughness images and combines them into a single UV texture map, weighted by the camera configuration.

Areas not visible from any camera angle will appear as black patches in the mask outputs — these are filled by the InPaint node.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pipeline` | HY3DPIPELINE | - | Paint pipeline from the MultiViews Generator, containing the mesh and renderer. |
| `camera_config` | HY3D21CAMERA | - | Camera configuration matching the views that were generated. |
| `albedo` | IMAGE | - | Multi-view albedo images from the MultiViews Generator. |
| `mr` | IMAGE | - | Multi-view metallic-roughness images from the MultiViews Generator. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `pipeline` | HY3DPIPELINE | Pass-through pipeline for the InPaint node. |
| `albedo` | NPARRAY | Baked albedo texture as numpy array. |
| `albedo_mask` | NPARRAY | Mask showing painted (white) vs unpainted (black) regions of albedo. |
| `mr` | NPARRAY | Baked metallic-roughness texture as numpy array. |
| `mr_mask` | NPARRAY | Mask showing painted vs unpainted regions of MR texture. |
| `albedo_texture` | IMAGE | Preview of the baked albedo as a ComfyUI image. |
| `mr_texture` | IMAGE | Preview of the baked MR texture as a ComfyUI image. |

---

## InPaint

Fill unpainted regions of the baked textures and export the final textured mesh as a GLB file with PBR materials (albedo + metallic-roughness).

The inpainting uses the bake masks to identify regions not covered by any camera view and fills them with plausible texture content.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pipeline` | HY3DPIPELINE | - | Paint pipeline from the Bake node, containing mesh and renderer. |
| `albedo` | NPARRAY | - | Baked albedo texture from the Bake node. |
| `albedo_mask` | NPARRAY | - | Albedo coverage mask from the Bake node. Black regions will be inpainted. |
| `mr` | NPARRAY | - | Baked metallic-roughness texture from the Bake node. |
| `mr_mask` | NPARRAY | - | MR coverage mask from the Bake node. |
| `output_mesh_name` | STRING | - | Filename for the exported GLB (without extension). The file is saved in ComfyUI's output directory. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `albedo` | IMAGE | Final inpainted albedo texture. |
| `mr` | IMAGE | Final inpainted metallic-roughness texture. |
| `trimesh` | TRIMESH | The textured mesh object. |
| `output_glb_path` | STRING | Path to the exported GLB file with PBR materials. |

---

## Camera Config

Define camera viewpoints for multi-view rendering. Each viewpoint is defined by an azimuth (horizontal angle), elevation (vertical angle), and weight (importance during baking).

The default configuration covers 6 views: front, right, back, left, top, and bottom — with higher weight on the front view.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera_azimuths` | STRING | `"0, 90, 180, 270, 0, 180"` | Comma-separated horizontal rotation angles in degrees. **0** = front, **90** = right side, **180** = back, **270** = left side. The number of values determines how many views are rendered. |
| `camera_elevations` | STRING | `"0, 0, 0, 0, 90, -90"` | Comma-separated vertical angles in degrees. **0** = eye level, **90** = directly above (top-down), **-90** = directly below (bottom-up). Must have the same count as azimuths. |
| `view_weights` | STRING | `"1, 0.1, 0.5, 0.1, 0.05, 0.05"` | Comma-separated importance weights for texture baking. Higher weight = that view contributes more to the final texture. Typically the front view (matching the reference image) gets weight 1.0, and other views get lower weights. |
| `ortho_scale` | FLOAT | `1.0` | Orthographic camera scale. Controls how much of the object is visible in each view. **1.0** fits most objects. **Increase** if the object is getting cropped. **Decrease** to zoom in. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `camera_config` | HY3D21CAMERA | Camera configuration dict to connect to the MultiViews Generator. |

---

## Use MultiViews

Create a paint pipeline from pre-generated multi-view images (e.g., from a previous generation that was saved, or from an external tool). Skips the diffusion step entirely — just sets up the renderer for baking.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trimesh` | TRIMESH | - | 3D mesh to apply textures to. Must match the mesh used to generate the multi-views. |
| `camera_config` | HY3D21CAMERA | - | Camera configuration that matches the multi-view images. |
| `albedo` | IMAGE | - | Pre-generated albedo multi-view images. |
| `mr` | IMAGE | - | Pre-generated metallic-roughness multi-view images. |
| `view_size` | INT | `512` | Resolution matching the multi-view images. |
| `texture_size` | INT | `1024` | Output texture resolution for baking. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `pipeline` | HY3DPIPELINE | Paint pipeline ready for the Bake node. |
| `albedo` | IMAGE | Pass-through of input albedo images. |
| `mr` | IMAGE | Pass-through of input MR images. |
| `camera_config` | HY3D21CAMERA | Pass-through of camera config. |

---

## Use MultiViews From MetaData

Load multi-view images and camera configuration from a previously exported `meta_data.json` file. Useful for re-baking textures at different resolutions or applying saved multi-views to a different mesh.

If upscaled versions of the multi-views exist (from a previous batch run with upscaling), those are used automatically.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trimesh` | TRIMESH | - | 3D mesh to apply the saved textures to. |
| `metadata_file` | STRING | - | Full path to a `meta_data.json` file from a previous generation. |
| `view_size` | INT | `512` | View resolution matching the saved images. |
| `texture_size` | INT | `1024` | Output texture resolution for baking. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `pipeline` | HY3DPIPELINE | Paint pipeline ready for the Bake node. |
| `albedo` | IMAGE | Loaded albedo multi-view images. |
| `mr` | IMAGE | Loaded MR multi-view images. |
| `camera_config` | HY3D21CAMERA | Camera configuration from the metadata. |

---

## MultiViews Generator With MetaData

Same as MultiViews Generator, but also saves the generated multi-view images and metadata to disk. This enables re-using the generation results later without re-running the diffusion model.

**Inputs:**

Same as MultiViews Generator, plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_name` | STRING | - | Base filename for saving. Creates a folder with this name containing the multi-view images and `meta_data.json`. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `pipeline` | HY3DPIPELINE | Paint pipeline for baking. |
| `albedo` | IMAGE | Generated albedo multi-views. |
| `mr` | IMAGE | Generated MR multi-views. |
| `metadata` | HY3D21METADATA | Metadata object for the Bake With MetaData node. |
| `positions` | IMAGE | Position maps. |
| `normals` | IMAGE | Normal maps. |

---

## Bake MultiViews With MetaData

Bakes textures, inpaints, and exports the final GLB — all in one node. Also saves metadata for later re-use. Combines the Bake + InPaint steps with metadata export.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pipeline` | HY3DPIPELINE | - | Paint pipeline from the MultiViews Generator. |
| `albedo` | IMAGE | - | Multi-view albedo images. |
| `mr` | IMAGE | - | Multi-view MR images. |
| `metadata` | HY3D21METADATA | - | Metadata from the MultiViews Generator With MetaData node. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `albedo` | IMAGE | Final inpainted albedo texture. |
| `mr` | IMAGE | Final inpainted MR texture. |
| `trimesh` | TRIMESH | The textured mesh. |
| `output_glb_path` | STRING | Path to the exported GLB with PBR materials. |

---

## HighPoly to LowPoly Bake

Create multiple Level-of-Detail (LOD) meshes from a high-poly textured mesh using saved metadata. For each target face count, the mesh is decimated, UV-unwrapped, and textures are re-baked from the saved multi-views.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metadata_file` | STRING | - | Path to `meta_data.json` from a previous MetaData generation. Must reference a valid mesh file and multi-view images. |
| `view_size` | INT | `512` | View resolution matching the saved multi-views. |
| `texture_size` | INT | `1024` | Texture resolution for the low-poly meshes. |
| `target_face_nums` | STRING | `"20000,10000,5000"` | Comma-separated face counts for each LOD level. A separate mesh is generated for each value. Example: `"20000,10000,5000"` creates three LODs. |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `output_lowpoly_path` | STRING | Path to the folder containing generated LOD meshes. |

---

## MultiViews Generator Batch

Process folders of images and meshes to generate textures in batch. Each image is matched to a mesh by filename, and the full texture pipeline (multiview generation → bake → inpaint → GLB export) runs automatically.

**Inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_folder` | STRING | - | Where to save textured GLB files and optional metadata. |
| `camera_config` | HY3D21CAMERA | - | Camera configuration for all meshes in the batch. |
| `paint_model` | ENUM | - | Paint model directory from `ComfyUI/models/diffusers/`. |
| `view_size` | INT | `512` | Resolution for rendered views. |
| `steps` | INT | `10` | Diffusion steps per mesh. |
| `guidance_scale` | FLOAT | `3.0` | Reference image adherence strength. |
| `texture_size` | INT | `1024` | Output texture resolution. |
| `unwrap_mesh` | BOOLEAN | `True` | UV unwrap each mesh before texturing. |
| `seed` | INT | `0` | Base random seed. |
| `generate_random_seed` | BOOLEAN | `True` | Use a unique random seed per mesh for variety. |
| `remove_background` | BOOLEAN | `False` | Remove image backgrounds before texturing. Requires the `rembg` pip package. |
| `skip_generated_mesh` | BOOLEAN | `True` | Skip meshes that already have output in the folder. Enables resumable batches. |
| `upscale_multiviews` | ENUM | `"None"` | **None:** no upscaling. **CustomModel:** upscale multi-view images using the selected model before baking, for sharper textures. |
| `upscale_model_name` | ENUM | - | Upscale model from `ComfyUI/models/upscale_models/`. Only used when upscale_multiviews = CustomModel. |
| `export_multiviews` | BOOLEAN | `True` | Save individual multi-view images alongside each textured mesh. Useful for re-baking later. |
| `export_metadata` | BOOLEAN | `True` | Save a `meta_data.json` file with camera config and image references. Required for the HighPoly to LowPoly node. |
| `input_images_folder` | STRING | *(optional)* | Folder with reference images. Matched to meshes by filename (e.g., `chair.png` matches `chair.glb`). |
| `input_meshes_folder` | STRING | *(optional)* | Folder with input meshes (GLB or OBJ). |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `processed_meshes` | STRING | List of generated textured mesh paths. |
