from PIL import Image, ImageSequence, ImageOps
import torch
import shutil
import argparse
import copy
import os
import time
import re
import numpy as np
import gc
import json
import trimesh as Trimesh
from dataclasses import dataclass, field
from typing import Union, Optional, Tuple, List, Any
from pathlib import Path

#painting
from .hy3dpaint.DifferentiableRenderer.MeshRender import MeshRender
from .hy3dpaint.utils.multiview_utils import multiviewDiffusionNet
from .hy3dpaint.utils.pipeline_utils import ViewProcessor
from .hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap
from .hy3dpaint.convert_utils import create_glb_with_pbr_materials
from .hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

from .hy3dshape.hy3dshape.rembg import BackgroundRemover

from spandrel import ModelLoader, ImageModelDescriptor

import folder_paths
import node_helpers
import hashlib

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import comfy.utils

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
diffusions_dir = os.path.join(comfy_path, "models", "diffusers")


def _get_diffusers_models():
    """List directories in ComfyUI/models/diffusers/ for diffusers-style pipeline selection."""
    if not os.path.isdir(diffusions_dir):
        return []
    return [d for d in os.listdir(diffusions_dir) if os.path.isdir(os.path.join(diffusions_dir, d))]


def _resolve_paint_model_paths(paint_model_name=None):
    """Resolve all model paths needed by the paint pipeline.

    Returns dict with keys: paint_model_path, dino_model_path, realesrgan_model_path, mlx_weights_path
    """
    paths = {}

    # Paint model: diffusers-style directory
    if paint_model_name:
        paths["paint_model_path"] = os.path.join(diffusions_dir, paint_model_name)
    else:
        paths["paint_model_path"] = None

    # DINOv2: check clip_vision folder, fall back to HF repo ID
    dino_dirs = [d for d in (folder_paths.get_filename_list("clip_vision") if "clip_vision" in folder_paths.folder_names_and_paths else []) if "dinov2" in d.lower()]
    if dino_dirs:
        paths["dino_model_path"] = folder_paths.get_full_path("clip_vision", dino_dirs[0])
    else:
        # Fallback: check for dinov2-giant directory directly in clip_vision folders
        clip_dirs = folder_paths.get_folder_paths("clip_vision") if "clip_vision" in folder_paths.folder_names_and_paths else []
        found = False
        for clip_dir in clip_dirs:
            dino_path = os.path.join(clip_dir, "dinov2-giant")
            if os.path.isdir(dino_path):
                paths["dino_model_path"] = dino_path
                found = True
                break
        if not found:
            paths["dino_model_path"] = "facebook/dinov2-giant"  # HF fallback

    # RealESRGAN: check upscale_models folder
    try:
        paths["realesrgan_model_path"] = folder_paths.get_full_path("upscale_models", "RealESRGAN_x4plus.pth")
    except Exception:
        paths["realesrgan_model_path"] = None

    # MLX weights: look for hunyuan3d-mlx-weights directory in diffusers folder
    mlx_dir = os.path.join(diffusions_dir, "hunyuan3d-mlx-weights")
    if os.path.isdir(mlx_dir):
        paths["mlx_weights_path"] = mlx_dir
    else:
        paths["mlx_weights_path"] = None

    return paths

def _get_diffusion_backends():
    """Return available diffusion backends. MLX only shown on Apple Silicon with mlx installed."""
    backends = ["pytorch"]
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core
            backends.append("mlx")
        except ImportError:
            pass  # MLX not installed, only show pytorch
    return backends

def parse_string_to_int_list(number_string):
  """
  Parses a string containing comma-separated numbers into a list of integers.

  Args:
    number_string: A string containing comma-separated numbers (e.g., "20000,10000,5000").

  Returns:
    A list of integers parsed from the input string.
    Returns an empty list if the input string is empty or None.
  """
  if not number_string:
    return []

  try:
    # Split the string by comma and convert each part to an integer
    int_list = [int(num.strip()) for num in number_string.split(',')]
    return int_list
  except ValueError as e:
    print(f"Error converting string to integer: {e}. Please ensure all values are valid numbers.")
    return []

def hy3dpaintimages_to_tensor(images):
    tensors = []
    for pil_img in images:
        np_img = np.array(pil_img).astype(np.uint8)
        np_img = np_img / 255.0
        tensor_img = torch.from_numpy(np_img).float()
        tensors.append(tensor_img)
    tensors = torch.stack(tensors)
    return tensors

def get_picture_files(folder_path):
    """
    Retrieves all picture files (based on common extensions) from a given folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the picture files found.
    """
    picture_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    picture_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []

    for entry_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry_name)

        # Check if the entry is actually a file (and not a sub-directory)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry_name)
            if file_extension.lower().endswith(picture_extensions):
                picture_files.append(full_path)
    return picture_files

def get_mesh_files(folder_path, name_filter = None):
    """
    Retrieves all picture files (based on common extensions) from a given folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the picture files found.
    """
    mesh_extensions = ('.obj', '.glb')
    mesh_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []

    for entry_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry_name)

        # Check if the entry is actually a file (and not a sub-directory)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry_name)
            if file_extension.lower().endswith(mesh_extensions):
                if name_filter is None or name_filter.lower() in file_name.lower():
                    mesh_files.append(full_path)
    return mesh_files

def get_filename_without_extension_os_path(full_file_path):
    """
    Extracts the filename without its extension from a full file path using os.path.

    Args:
        full_file_path (str): The complete path to the file.

    Returns:
        str: The filename without its extension.
    """
    # 1. Get the base name (filename with extension)
    base_name = os.path.basename(full_file_path)

    # 2. Split the base name into root (filename without ext) and extension
    file_name_without_ext, _ = os.path.splitext(base_name)

    return file_name_without_ext

def _convert_texture_format(tex: Union[np.ndarray, torch.Tensor, Image.Image],
                          texture_size: Tuple[int, int], device: str, force_set: bool = False) -> torch.Tensor:
    """Unified texture format conversion logic."""
    if not force_set:
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):
            tex_np = tex.cpu().numpy()

            # 2. Handle potential batch dimension (B, C, H, W) or (B, H, W, C)
            if tex_np.ndim == 4:
                if tex_np.shape[0] == 1:
                    tex_np = tex_np.squeeze(0)
                else:
                    tex_np = tex_np[0]

            # 3. Handle data type and channel order for PIL
            if tex_np.ndim == 3:
                if tex_np.shape[0] in [1, 3, 4] and tex_np.shape[0] < tex_np.shape[1] and tex_np.shape[0] < tex_np.shape[2]:
                    tex_np = np.transpose(tex_np, (1, 2, 0))
                elif tex_np.shape[2] in [1, 3, 4] and tex_np.shape[0] > 4 and tex_np.shape[1] > 4:
                    pass
                else:
                    raise ValueError(f"Unsupported 3D tensor shape after squeezing batch and moving to CPU. "
                                     f"Expected (C, H, W) or (H, W, C) but got {tex_np.shape}")

                if tex_np.shape[2] == 1:
                    tex_np = tex_np.squeeze(2) # Remove the channel dimension

            elif tex_np.ndim == 2:
                pass
            else:
                raise ValueError(f"Unsupported tensor dimension after squeezing batch and moving to CPU: {tex_np.ndim} "
                                 f"with shape {tex_np.shape}. Expected 2D or 3D image data.")

            tex_np_uint8 = (tex_np * 255).astype(np.uint8)

            tex = Image.fromarray(tex_np_uint8)


        tex = tex.resize(texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        return torch.from_numpy(tex).float().to(device)
    else:
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        return tex.float().to(device)

def convert_ndarray_to_pil(texture):
    texture_size = len(texture)
    tex = _convert_texture_format(texture,(texture_size, texture_size), mm.get_torch_device())
    tex = tex.cpu().numpy()
    processed_texture = (tex * 255).astype(np.uint8)
    pil_texture = Image.fromarray(processed_texture)
    return pil_texture

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def convert_pil_images_to_tensor(images):
    tensor_array = []

    for image in images:
        tensor_array.append(pil2tensor(image))

    return tensor_array

def convert_tensor_images_to_pil(images):
    pil_array = []

    for image in images:
        pil_array.append(tensor2pil(image))

    return pil_array

@dataclass
class MetaData:
    camera_config: Optional[dict] = None
    albedos: Optional[List[str]] = None
    mrs: Optional[List[str]] = None
    albedos_upscaled: Optional[List[str]] = None
    mrs_upscaled: Optional[List[str]] = None
    mesh_file: Optional[str] = None


class Hy3DMultiViewsGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "paint_model": (_get_diffusers_models(), {"tooltip": "Paint model directory from ComfyUI/models/diffusers/ (e.g. hunyuan3d-paintpbr-v2-1)"}),
                "view_size": ("INT", {"default": 512, "min": 512, "max":1024, "step":256}),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of steps"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1, "max": 10, "step": 0.1, "tooltip": "Guidance scale"}),
                "texture_size": ("INT", {"default":1024,"min":512,"max":4096,"step":512}),
                "unwrap_mesh": ("BOOLEAN", {"default":True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "diffusion_backend": (_get_diffusion_backends(), {"default": "pytorch", "tooltip": "Diffusion backend: pytorch (default, proven) or mlx (experimental, ~5x faster on Apple Silicon)"}),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE","IMAGE","IMAGE","HY3D21CAMERA","HY3D21METADATA",)
    RETURN_NAMES = ("pipeline", "albedo","mr","positions","normals","camera_config", "metadata")
    FUNCTION = "genmultiviews"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Generate multi-view PBR textures from a mesh using multiview diffusion."

    def genmultiviews(self, trimesh, camera_config, paint_model, view_size, image, steps, guidance_scale, texture_size, unwrap_mesh, seed, diffusion_backend="pytorch"):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        seed = seed % (2**32)

        model_paths = _resolve_paint_model_paths(paint_model)
        conf = Hunyuan3DPaintConfig(
            view_size, camera_config["selected_camera_azims"], camera_config["selected_camera_elevs"],
            camera_config["selected_view_weights"], camera_config["ortho_scale"], texture_size, device=device,
            paint_model_path=model_paths["paint_model_path"],
            dino_model_path=model_paths["dino_model_path"],
            realesrgan_model_path=model_paths["realesrgan_model_path"],
            mlx_weights_path=model_paths.get("mlx_weights_path"),
        )
        conf.diffusion_backend = diffusion_backend
        paint_pipeline = Hunyuan3DPaintPipeline(conf)

        image = tensor2pil(image)

        temp_folder_path = os.path.join(comfy_path, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        temp_output_path = os.path.join(temp_folder_path, "textured_mesh.obj")

        albedo, mr, normal_maps, position_maps = paint_pipeline(mesh=trimesh, image_path=image, output_mesh_path=temp_output_path, num_steps=steps, guidance_scale=guidance_scale, unwrap=unwrap_mesh, seed=seed)

        albedo_tensor = hy3dpaintimages_to_tensor(albedo)
        mr_tensor = hy3dpaintimages_to_tensor(mr)
        normals_tensor = hy3dpaintimages_to_tensor(normal_maps)
        positions_tensor = hy3dpaintimages_to_tensor(position_maps)

        return (paint_pipeline, albedo_tensor, mr_tensor, positions_tensor, normals_tensor, camera_config,)

class Hy3DBakeMultiViews:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "camera_config": ("HY3D21CAMERA", ),
                "albedo": ("IMAGE", ),
                "mr": ("IMAGE", )
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("pipeline", "albedo", "albedo_mask", "mr", "mr_mask", "albedo_texture", "mr_texture",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Bake multi-view images into UV texture maps."

    def process(self, pipeline, camera_config, albedo, mr):
        albedo = convert_tensor_images_to_pil(albedo)
        mr = convert_tensor_images_to_pil(mr)

        texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(albedo,mr,camera_config["selected_camera_elevs"], camera_config["selected_camera_azims"], camera_config["selected_view_weights"])

        texture_pil = convert_ndarray_to_pil(texture)
        #mask_pil = convert_ndarray_to_pil(mask)
        texture_mr_pil = convert_ndarray_to_pil(texture_mr)
        #mask_mr_pil = convert_ndarray_to_pil(mask_mr)

        texture_tensor = pil2tensor(texture_pil)
        #mask_tensor = pil2tensor(mask_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)
        #mask_mr_tensor = pil2tensor(mask_mr_pil)

        return (pipeline, texture, mask, texture_mr, mask_mr, texture_tensor, texture_mr_tensor)

class Hy3DInPaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "albedo": ("NPARRAY", ),
                "albedo_mask": ("NPARRAY", ),
                "mr": ("NPARRAY", ),
                "mr_mask": ("NPARRAY",),
                "output_mesh_name": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","TRIMESH", "STRING",)
    RETURN_NAMES = ("albedo", "mr", "trimesh", "output_glb_path")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Inpaint baked textures and export final GLB with PBR materials."
    OUTPUT_NODE = True

    def process(self, pipeline, albedo, albedo_mask, mr, mr_mask, output_mesh_name):

        #albedo = tensor2pil(albedo)
        #albedo_mask = tensor2pil(albedo_mask)
        #mr = tensor2pil(mr)
        #mr_mask = tensor2pil(mr_mask)

        vertex_inpaint = True
        method = "NS"

        albedo, mr = pipeline.inpaint(albedo, albedo_mask, mr, mr_mask, vertex_inpaint, method)

        pipeline.set_texture_albedo(albedo)
        pipeline.set_texture_mr(mr)

        # Strip any existing extension from mesh name to avoid double extensions
        mesh_base_name = os.path.splitext(output_mesh_name)[0]

        temp_folder_path = os.path.join(comfy_path, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        output_mesh_path = os.path.join(temp_folder_path, f"{mesh_base_name}.obj")
        output_temp_path = pipeline.save_mesh(output_mesh_path)

        # Save to output with counter-based naming to avoid caching issues
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(mesh_base_name, folder_paths.get_output_directory())
        output_glb_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.glb")
        shutil.copyfile(output_temp_path, output_glb_path)

        trimesh = Trimesh.load(output_glb_path, force="mesh")

        texture_pil = convert_ndarray_to_pil(albedo)
        texture_mr_pil = convert_ndarray_to_pil(mr)
        texture_tensor = pil2tensor(texture_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)

        # Return relative path (subfolder/filename) matching ComfyUI convention
        output_glb_relative = os.path.join(subfolder, f"{filename}_{counter:05}_.glb") if subfolder else f"{filename}_{counter:05}_.glb"

        pipeline.clean_memory()

        del pipeline

        mm.soft_empty_cache()
        gc.collect()

        return (texture_tensor, texture_mr_tensor, trimesh, output_glb_relative)

class Hy3D21CameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 0, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "view_weights": ("STRING", {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False}),
                "ortho_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("HY3D21CAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Configure camera angles and weights for multi-view rendering."

    def process(self, camera_azimuths, camera_elevations, view_weights, ortho_scale):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(',')))
        weights_list = list(map(float, view_weights.replace(" ", "").split(',')))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list,
            "ortho_scale": ortho_scale,
            }

        return (camera_config,)

class Hy3D21UseMultiViews:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "albedo": ("IMAGE",),
                "mr": ("IMAGE",),
                "view_size": ("INT",{"default":512}),
                "texture_size": ("INT",{"default":1024}),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE","HY3D21CAMERA",)
    RETURN_NAMES = ("pipeline", "albedo","mr","camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Create a paint pipeline from pre-generated multi-view images."

    def process(self, trimesh, camera_config, albedo, mr, view_size, texture_size):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        conf = Hunyuan3DPaintConfig(view_size, camera_config["selected_camera_azims"], camera_config["selected_camera_elevs"], camera_config["selected_view_weights"], camera_config["ortho_scale"], texture_size, device=device)
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        paint_pipeline.load_mesh(trimesh)

        return (paint_pipeline, albedo, mr, camera_config)

class Hy3D21UseMultiViewsFromMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "metadata_file": ("STRING",),
                "view_size": ("INT",{"default":512}),
                "texture_size": ("INT",{"default":1024}),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE","HY3D21CAMERA",)
    RETURN_NAMES = ("pipeline", "albedo","mr","camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Load multi-view images and camera config from an exported metadata JSON."

    def process(self, trimesh, metadata_file, view_size, texture_size):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        with open(metadata_file, 'r') as fr:
            loaded_data = json.load(fr)
            loaded_metaData = MetaData()
            for key, value in loaded_data.items():
                setattr(loaded_metaData, key, value)

        conf = Hunyuan3DPaintConfig(view_size, loaded_metaData.camera_config["selected_camera_azims"], loaded_metaData.camera_config["selected_camera_elevs"], loaded_metaData.camera_config["selected_view_weights"], loaded_metaData.camera_config["ortho_scale"], texture_size, device=device)
        paint_pipeline = Hunyuan3DPaintPipeline(conf)

        paint_pipeline.load_mesh(trimesh)

        dir_name = os.path.dirname(metadata_file)

        albedos = []
        mrs = []

        if loaded_metaData.albedos_upscaled != None:
            print('Using upscaled pictures ...')
            for file in loaded_metaData.albedos_upscaled:
                albedo_file = os.path.join(dir_name,file)
                albedo = Image.open(albedo_file)
                albedos.append(albedo)

            for file in loaded_metaData.mrs_upscaled:
                mr_file = os.path.join(dir_name,file)
                mr = Image.open(mr_file)
                mrs.append(mr)
        else:
            print('Using non-upscaled pictures ...')
            for file in loaded_metaData.albedos:
                albedo_file = os.path.join(dir_name,file)
                albedo = Image.open(albedo_file)
                albedos.append(albedo)

            for file in loaded_metaData.mrs:
                mr_file = os.path.join(dir_name,file)
                mr = Image.open(mr_file)
                mrs.append(mr)

        albedos_tensor = convert_pil_images_to_tensor(albedos)
        mrs_tensor = convert_pil_images_to_tensor(mrs)

        return (paint_pipeline, albedos_tensor, mrs_tensor, loaded_metaData.camera_config)

class Hy3D21MultiViewsGeneratorWithMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "paint_model": (_get_diffusers_models(), {"tooltip": "Paint model directory from ComfyUI/models/diffusers/"}),
                "view_size": ("INT", {"default": 512, "min": 512, "max":1024, "step":256}),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of steps"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1, "max": 10, "step": 0.1, "tooltip": "Guidance scale"}),
                "texture_size": ("INT", {"default":1024,"min":512,"max":4096,"step":512}),
                "unwrap_mesh": ("BOOLEAN", {"default":True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "output_name":("STRING",),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE","HY3D21METADATA","IMAGE","IMAGE",)
    RETURN_NAMES = ("pipeline", "albedo","mr","metadata","positions","normals",)
    FUNCTION = "genmultiviews"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Generate multi-view textures and save results with metadata."

    def genmultiviews(self, trimesh, camera_config, paint_model, view_size, image, steps, guidance_scale, texture_size, unwrap_mesh, seed, output_name):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        seed = seed % (2**32)

        model_paths = _resolve_paint_model_paths(paint_model)
        conf = Hunyuan3DPaintConfig(
            view_size, camera_config["selected_camera_azims"], camera_config["selected_camera_elevs"],
            camera_config["selected_view_weights"], camera_config["ortho_scale"], texture_size, device=device,
            paint_model_path=model_paths["paint_model_path"],
            dino_model_path=model_paths["dino_model_path"],
            realesrgan_model_path=model_paths["realesrgan_model_path"],
            mlx_weights_path=model_paths.get("mlx_weights_path"),
        )
        paint_pipeline = Hunyuan3DPaintPipeline(conf)

        image = tensor2pil(image)

        temp_folder_path = os.path.join(comfy_path, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        temp_output_path = os.path.join(temp_folder_path, "textured_mesh.obj")

        albedo, mr, normal_maps, position_maps = paint_pipeline(mesh=trimesh, image_path=image, output_mesh_path=temp_output_path, num_steps=steps, guidance_scale=guidance_scale, unwrap=unwrap_mesh, seed=seed)

        albedo_tensor = hy3dpaintimages_to_tensor(albedo)
        mr_tensor = hy3dpaintimages_to_tensor(mr)
        normals_tensor = hy3dpaintimages_to_tensor(normal_maps)
        positions_tensor = hy3dpaintimages_to_tensor(position_maps)

        output_dir_path = os.path.join(comfy_path, "output", "3D", output_name)
        os.makedirs(output_dir_path, exist_ok=True)

        metadata = MetaData()
        metadata.mesh_file = output_name
        metadata.camera_config = camera_config
        metadata.albedos = []
        metadata.mrs = []

        print('Saving Albedo and MR views ...')
        for index, img in enumerate(albedo_tensor):
            output_file_path = os.path.join(output_dir_path,f'Albedo_{index}.png')
            pil_image = tensor2pil(img)
            pil_image.save(output_file_path)
            metadata.albedos.append(f'Albedo_{index}.png')

        for index, img in enumerate(mr_tensor):
            output_file_path = os.path.join(output_dir_path,f'MR_{index}.png')
            pil_image = tensor2pil(img)
            pil_image.save(output_file_path)
            metadata.mrs.append(f'MR_{index}.png')

        return (paint_pipeline, albedo_tensor, mr_tensor, metadata, positions_tensor, normals_tensor,)

class Hy3DBakeMultiViewsWithMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "albedo": ("IMAGE", ),
                "mr": ("IMAGE", ),
                "metadata": ("HY3D21METADATA",),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","TRIMESH", "STRING", )
    RETURN_NAMES = ("albedo", "mr", "trimesh", "output_glb_path", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Bake multi-view textures, inpaint, and save with metadata."

    def process(self, pipeline, albedo, mr, metadata):
        vertex_inpaint = True
        method = "NS"

        albedo = convert_tensor_images_to_pil(albedo)
        mr = convert_tensor_images_to_pil(mr)

        output_mesh_name = metadata.mesh_file
        output_dir_path = os.path.join(comfy_path, "output", "3D", output_mesh_name)

        #detect if images have been upscaled
        albedo1 = albedo[0]
        width, height = albedo1.size
        if width>pipeline.config.resolution:
            print('Upscaled images detected. Saving Upscaled images ...')
            metadata.albedos_upscaled = []
            metadata.mrs_upscaled = []

            for index, img in enumerate(albedo):
                output_file_path = os.path.join(output_dir_path,f'Albedo_Upscaled_{index}.png')
                img.save(output_file_path)
                metadata.albedos_upscaled.append(f'Albedo_Upscaled_{index}.png')

            for index, img in enumerate(mr):
                output_file_path = os.path.join(output_dir_path,f'MR_Upscaled_{index}.png')
                img.save(output_file_path)
                metadata.mrs_upscaled.append(f'MR_Upscaled_{index}.png')

        camera_config = metadata.camera_config
        texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(albedo,mr,camera_config["selected_camera_elevs"], camera_config["selected_camera_azims"], camera_config["selected_view_weights"])

        albedo, mr = pipeline.inpaint(texture, mask, texture_mr, mask_mr, vertex_inpaint, method)

        pipeline.set_texture_albedo(albedo)
        pipeline.set_texture_mr(mr)

        output_glb_path = os.path.join(output_dir_path,f'{output_mesh_name}.obj')

        pipeline.save_mesh(output_glb_path)

        output_glb_path = os.path.join(output_dir_path,f'{output_mesh_name}.glb')

        trimesh = Trimesh.load(output_glb_path)

        texture_pil = convert_ndarray_to_pil(albedo)
        texture_mr_pil = convert_ndarray_to_pil(mr)
        texture_tensor = pil2tensor(texture_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)

        pipeline.clean_memory()

        metadata.mesh_file = f'{output_mesh_name}.glb'

        output_metadata_path = os.path.join(output_dir_path,'meta_data.json')
        with open(output_metadata_path,'w') as fw:
            json.dump(metadata.__dict__, indent="\t", fp=fw)

        del pipeline

        mm.soft_empty_cache()
        gc.collect()

        return (texture_tensor, texture_mr_tensor, trimesh, output_glb_path)

class Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metadata_file": ("STRING",),
                "view_size": ("INT",{"default":512}),
                "texture_size": ("INT",{"default":1024}),
                "target_face_nums": ("STRING",{"default":"20000,10000,5000"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("output_lowpoly_path", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Create multiple LOD meshes from a high-poly mesh using saved metadata."
    OUTPUT_NODE = True

    def process(self, metadata_file, view_size, texture_size, target_face_nums):
        try:
            import meshlib.mrmeshpy as mrmeshpy
        except ImportError:
            raise ImportError("meshlib not found. Please install it using 'pip install meshlib'")

        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        output_lowpoly_path = ""

        vertex_inpaint = True
        method = "NS"

        with open(metadata_file, 'r') as fr:
            loaded_data = json.load(fr)
            loaded_metaData = MetaData()
            for key, value in loaded_data.items():
                setattr(loaded_metaData, key, value)

        list_of_faces = parse_string_to_int_list(target_face_nums)
        if len(list_of_faces)>0:
            input_dir = os.path.dirname(metadata_file)
            mesh_name = loaded_metaData.mesh_file.replace(".glb","").replace(".obj","")
            mesh_file_path = os.path.join(input_dir, loaded_metaData.mesh_file)

            if os.path.exists(mesh_file_path):
                conf = Hunyuan3DPaintConfig(view_size, loaded_metaData.camera_config["selected_camera_azims"], loaded_metaData.camera_config["selected_camera_elevs"], loaded_metaData.camera_config["selected_view_weights"], loaded_metaData.camera_config["ortho_scale"], texture_size, device=device)

                highpoly_mesh = Trimesh.load(mesh_file_path, force="mesh")
                highpoly_mesh = Trimesh.Trimesh(vertices=highpoly_mesh.vertices, faces=highpoly_mesh.faces) # Remove texture coordinates
                highpoly_faces_num = highpoly_mesh.faces.shape[0]

                albedos = []
                mrs = []

                if loaded_metaData.albedos_upscaled != None:
                    print('Using upscaled pictures ...')
                    for file in loaded_metaData.albedos_upscaled:
                        albedo_file = os.path.join(input_dir,file)
                        albedo = Image.open(albedo_file)
                        albedos.append(albedo)

                    for file in loaded_metaData.mrs_upscaled:
                        mr_file = os.path.join(input_dir,file)
                        mr = Image.open(mr_file)
                        mrs.append(mr)
                else:
                    print('Using non-upscaled pictures ...')
                    for file in loaded_metaData.albedos:
                        albedo_file = os.path.join(input_dir,file)
                        albedo = Image.open(albedo_file)
                        albedos.append(albedo)

                    for file in loaded_metaData.mrs:
                        mr_file = os.path.join(dir_name,file)
                        mr = Image.open(mr_file)
                        mrs.append(mr)

                output_lowpoly_path = os.path.join(input_dir, "LowPoly")

                for target_face_num in list_of_faces:
                    print('Processing {target_face_num} faces ...')
                    pipeline = Hunyuan3DPaintPipeline(conf)
                    output_dir_path = os.path.join(input_dir, "LowPoly", f"{target_face_num}")
                    os.makedirs(output_dir_path, exist_ok=True)

                    settings = mrmeshpy.DecimateSettings()
                    faces_to_delete = highpoly_faces_num - target_face_num
                    settings.maxDeletedFaces = faces_to_delete
                    settings.subdivideParts = 16
                    settings.packMesh = True

                    print(f'Decimating to {target_face_num} faces ...')
                    lowpoly_mesh = postprocessmesh(highpoly_mesh.vertices, highpoly_mesh.faces, settings)

                    print('UV Unwrapping ...')
                    lowpoly_mesh = mesh_uv_wrap(lowpoly_mesh)

                    pipeline.load_mesh(lowpoly_mesh)

                    camera_config = loaded_metaData.camera_config
                    texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(albedos,mrs,camera_config["selected_camera_elevs"], camera_config["selected_camera_azims"], camera_config["selected_view_weights"])

                    albedo, mr = pipeline.inpaint(texture, mask, texture_mr, mask_mr, vertex_inpaint, method)

                    pipeline.set_texture_albedo(albedo)
                    pipeline.set_texture_mr(mr)

                    output_glb_path = os.path.join(output_dir_path,f'{mesh_name}_{target_face_num}.obj')

                    pipeline.save_mesh(output_glb_path)

                    pipeline.clean_memory()

            else:
                print(f'Mesh file does not exist: {mesh_file_path}')
        else:
            print('target_face_nums is empty')

        return (output_lowpoly_path,)

class Hy3D21GenerateMultiViewsBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_folder": ("STRING",),
                "camera_config": ("HY3D21CAMERA",),
                "paint_model": (_get_diffusers_models(), {"tooltip": "Paint model directory from ComfyUI/models/diffusers/"}),
                "view_size": ("INT", {"default": 512, "min": 512, "max":1024, "step":256}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of steps"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1, "max": 10, "step": 0.1, "tooltip": "Guidance scale"}),
                "texture_size": ("INT", {"default":1024,"min":512,"max":4096,"step":512}),
                "unwrap_mesh": ("BOOLEAN", {"default":True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "generate_random_seed": ("BOOLEAN",{"default":True}),
                "remove_background": ("BOOLEAN",{"default":False}),
                "skip_generated_mesh": ("BOOLEAN",{"default":True}),
                "upscale_multiviews": (["None","CustomModel"],{"default":"None"}),
                "upscale_model_name": (folder_paths.get_filename_list("upscale_models"), ),
                "export_multiviews": ("BOOLEAN",{"default":True, "tooltip":"Multiviews can be used to apply texture to a low poly mesh"}),
                "export_metadata": ("BOOLEAN",{"default":True,"tooltip":"Exporta json file with camera config and multiviews"}),
            },
            "optional": {
                "input_images_folder": ("STRING",),
                "input_meshes_folder": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_meshes",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Process all meshes from a folder"
    OUTPUT_NODE = True

    def process(self, output_folder, camera_config, paint_model, view_size, steps, guidance_scale, texture_size, unwrap_mesh, seed, generate_random_seed, remove_background, skip_generated_mesh, upscale_multiviews, upscale_model_name, export_multiviews, export_metadata, input_images_folder = None, input_meshes_folder = None):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()
        rembg = BackgroundRemover()
        processed_meshes = []

        vertex_inpaint = True
        method = "NS"

        if input_images_folder != None and input_meshes_folder != None:
            files = get_picture_files(input_images_folder)
            nb_pictures = len(files)

            if nb_pictures>0:
                model_paths = _resolve_paint_model_paths(paint_model)
                conf = Hunyuan3DPaintConfig(
                    view_size, camera_config["selected_camera_azims"], camera_config["selected_camera_elevs"],
                    camera_config["selected_view_weights"], camera_config["ortho_scale"], texture_size, device=device,
                    paint_model_path=model_paths["paint_model_path"],
                    dino_model_path=model_paths["dino_model_path"],
                    realesrgan_model_path=model_paths["realesrgan_model_path"],
                )

                temp_folder_path = os.path.join(comfy_path, "temp")
                os.makedirs(temp_folder_path, exist_ok=True)
                temp_output_path = os.path.join(temp_folder_path, "textured_mesh.obj")

                pbar = ProgressBar(nb_pictures)
                for file in files:
                    image_name = get_filename_without_extension_os_path(file)
                    input_meshes = get_mesh_files(input_meshes_folder, image_name)
                    if len(input_meshes)>0:
                        if len(input_meshes)>1:
                            print(f'Warning: Multiple meshes found for input_image {image_name} -> Taking the first one')

                        output_file_name = get_filename_without_extension_os_path(file)
                        output_mesh_folder = os.path.join(output_folder, output_file_name)
                        output_glb_path = Path(output_mesh_folder, f'{output_file_name}.glb')

                        processMesh = True

                        if skip_generated_mesh and os.path.exists(output_glb_path):
                            processMesh = False

                        if processMesh:
                            os.makedirs(output_mesh_folder, exist_ok=True)

                            print(f'Processing {file} with {input_meshes[0]} ...')
                            metaData = MetaData()
                            metaData.camera_config = camera_config
                            image = Image.open(file)
                            if remove_background:
                                print('Removing background ...')
                                image = rembg(image)

                            if generate_random_seed:
                                seed = int.from_bytes(os.urandom(4), 'big')

                            trimesh = Trimesh.load(input_meshes[0])

                            paint_pipeline = Hunyuan3DPaintPipeline(conf)
                            albedo, mr, normal_maps, position_maps = paint_pipeline(mesh=trimesh, image_path=image, output_mesh_path=temp_output_path, num_steps=steps, guidance_scale=guidance_scale, unwrap=unwrap_mesh, seed=seed)

                            if export_multiviews:
                                metaData.albedos = []
                                metaData.mrs = []

                                for index, img in enumerate(albedo):
                                    image_output_path = os.path.join(output_mesh_folder, f'Albedo_{index}.png')
                                    img.save(image_output_path)
                                    metaData.albedos.append(f'Albedo_{index}.png')

                                for index, img in enumerate(mr):
                                    image_output_path = os.path.join(output_mesh_folder, f'MR_{index}.png')
                                    img.save(image_output_path)
                                    metaData.mrs.append(f'MR_{index}.png')


                            if upscale_multiviews == "CustomModel":
                                model_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_model_name)
                                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                                if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                                    sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
                                upscale_model = ModelLoader().load_from_state_dict(sd).eval()

                                if not isinstance(upscale_model, ImageModelDescriptor):
                                    print("Cannot Upscale: Upscale model must be a single-image model.")
                                    del upscale_model
                                    upscale_model = None
                                else:
                                    upscale_model.to(device)

                                if upscale_model != None:
                                    print('Upscaling Albedo ...')
                                    albedo_tensors = hy3dpaintimages_to_tensor(albedo)
                                    in_img = albedo_tensors.movedim(-1,-3).to(device)

                                    tile = 512
                                    overlap = 32

                                    oom = True
                                    while oom:
                                        try:
                                            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                                            pbar = comfy.utils.ProgressBar(steps)
                                            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                                            oom = False
                                        except mm.OOM_EXCEPTION as e:
                                            tile //= 2
                                            if tile < 128:
                                                raise e

                                    #upscale_model.to("cpu")
                                    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)

                                    albedo = convert_tensor_images_to_pil(s)

                                    if export_multiviews:
                                        metaData.albedos_upscaled = []
                                        for index, img in enumerate(albedo):
                                            image_output_path = os.path.join(output_mesh_folder, f'Albedo_Upscaled_{index}.png')
                                            img.save(image_output_path)
                                            metaData.albedos_upscaled.append(f'Albedo_Upscaled_{index}.png')

                                    print('Upscaling MR ...')
                                    mr_tensors = hy3dpaintimages_to_tensor(mr)
                                    in_img = mr_tensors.movedim(-1,-3).to(device)

                                    tile = 512
                                    overlap = 32

                                    oom = True
                                    while oom:
                                        try:
                                            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                                            pbar = comfy.utils.ProgressBar(steps)
                                            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                                            oom = False
                                        except mm.OOM_EXCEPTION as e:
                                            tile //= 2
                                            if tile < 128:
                                                raise e

                                    #upscale_model.to("cpu")
                                    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)

                                    mr = convert_tensor_images_to_pil(s)

                                    if export_multiviews:
                                        metaData.mrs_upscaled = []
                                        for index, img in enumerate(mr):
                                            image_output_path = os.path.join(output_mesh_folder, f'MR_Upscaled_{index}.png')
                                            img.save(image_output_path)
                                            metaData.mrs_upscaled.append(f'MR_Upscaled_{index}.png')

                                    del upscale_model

                            print('Baking MultiViews ...')
                            texture, mask, texture_mr, mask_mr = paint_pipeline.bake_from_multiview(albedo,mr,camera_config["selected_camera_elevs"], camera_config["selected_camera_azims"], camera_config["selected_view_weights"])

                            albedo, mr = paint_pipeline.inpaint(texture, mask, texture_mr, mask_mr, vertex_inpaint, method)
                            paint_pipeline.set_texture_albedo(albedo)
                            paint_pipeline.set_texture_mr(mr)

                            output_mesh_path = os.path.join(comfy_path, "temp", f"{output_file_name}.obj")
                            output_temp_path = paint_pipeline.save_mesh(output_mesh_path)
                            shutil.copyfile(output_temp_path, output_glb_path)
                            metaData.mesh_file = f'{output_file_name}.glb'

                            if export_metadata:
                                output_metadata_path = os.path.join(output_mesh_folder,'meta_data.json')
                                with open(output_metadata_path,'w') as fw:
                                    json.dump(metaData.__dict__, indent="\t", fp=fw)

                            processed_meshes.append(output_glb_path)

                            paint_pipeline.clean_memory()
                            del paint_pipeline

                            mm.soft_empty_cache()
                            gc.collect()
                        else:
                            print(f'Skipping {file}')
                    else:
                        print(f'Error: No mesh found for input image {image_name}')

                    pbar.update(1)

            else:
                print('No image found in input_images_folder')
        else:
            print('Nothing to process')

        mm.soft_empty_cache()
        gc.collect()

        return (processed_meshes, )


NODE_CLASS_MAPPINGS = {
    "Hy3DMultiViewsGenerator": Hy3DMultiViewsGenerator,
    "Hy3DBakeMultiViews": Hy3DBakeMultiViews,
    "Hy3DInPaint": Hy3DInPaint,
    "Hy3D21CameraConfig": Hy3D21CameraConfig,
    "Hy3D21UseMultiViews": Hy3D21UseMultiViews,
    "Hy3D21UseMultiViewsFromMetaData": Hy3D21UseMultiViewsFromMetaData,
    "Hy3D21MultiViewsGeneratorWithMetaData": Hy3D21MultiViewsGeneratorWithMetaData,
    "Hy3DBakeMultiViewsWithMetaData": Hy3DBakeMultiViewsWithMetaData,
    "Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData": Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData,
    "Hy3D21GenerateMultiViewsBatch": Hy3D21GenerateMultiViewsBatch,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DMultiViewsGenerator": "Hunyuan 3D 2.1 MultiViews Generator",
    "Hy3DBakeMultiViews": "Hunyuan 3D 2.1 Bake MultiViews",
    "Hy3DInPaint": "Hunyuan 3D 2.1 InPaint",
    "Hy3D21CameraConfig": "Hunyuan 3D 2.1 Camera Config",
    "Hy3D21UseMultiViews": "Hunyuan 3D 2.1 Use MultiViews",
    "Hy3D21UseMultiViewsFromMetaData": "Hunyuan 3D 2.1 Use MultiViews From MetaData",
    "Hy3D21MultiViewsGeneratorWithMetaData": "Hunyuan 3D 2.1 MultiViews Generator With MetaData",
    "Hy3DBakeMultiViewsWithMetaData": "Hunyuan 3D 2.1 Bake MultiViews With MetaData",
    "Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData": "Hunyuan 3D 2.1 HighPoly to LowPoly Bake MultiViews With MetaData",
    "Hy3D21GenerateMultiViewsBatch": "Hunyuan 3D 2.1 MultiViews Generator Batch",
    }
