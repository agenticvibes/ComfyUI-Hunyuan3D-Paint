import subprocess
import sys
import os
import platform
import glob


script_dir = os.path.dirname(os.path.abspath(__file__))


def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def get_matching_wheel(dist_dir, package_prefix):
    """Find a precompiled wheel matching the current platform and Python version."""
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        plat_tag = "win_amd64"
    elif system == "linux":
        if machine == "x86_64":
            plat_tag = "linux_x86_64"
        elif machine == "aarch64":
            plat_tag = "linux_aarch64"
        else:
            return None
    else:
        return None  # No precompiled wheels for macOS

    pattern = os.path.join(dist_dir, f"{package_prefix}*{py_version}*{plat_tag}*.whl")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def install_extension(name, module_name, dist_dir, source_dir):
    """Install a C++ extension: try precompiled wheel first, then build from source."""
    if is_package_installed(module_name):
        print(f"[Hunyuan3D-Paint] {name} already installed.")
        return True

    # Try precompiled wheel
    wheel = get_matching_wheel(dist_dir, module_name.replace("_kernel", ""))
    if wheel is None:
        wheel = get_matching_wheel(dist_dir, module_name)

    if wheel:
        print(f"[Hunyuan3D-Paint] Installing {name} from precompiled wheel...")
        try:
            pip_install(wheel)
            print(f"[Hunyuan3D-Paint] {name} installed successfully from wheel.")
            return True
        except subprocess.CalledProcessError:
            print(f"[Hunyuan3D-Paint] Wheel installation failed for {name}, trying source build...")

    # Build from source
    print(f"[Hunyuan3D-Paint] Building {name} from source...")
    try:
        pip_install("--no-build-isolation", source_dir)
        print(f"[Hunyuan3D-Paint] {name} built and installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Hunyuan3D-Paint] ERROR: Failed to build {name} from source: {e}")
        print(f"[Hunyuan3D-Paint] You can try manually: cd {source_dir} && pip install .")
        return False


# Install pip dependencies
print("[Hunyuan3D-Paint] Installing pip dependencies...")
requirements_path = os.path.join(script_dir, "requirements.txt")
try:
    pip_install("-r", requirements_path)
except subprocess.CalledProcessError as e:
    print(f"[Hunyuan3D-Paint] WARNING: Some pip dependencies failed to install: {e}")

# Install custom_rasterizer
install_extension(
    name="custom_rasterizer",
    module_name="custom_rasterizer_kernel",
    dist_dir=os.path.join(script_dir, "hy3dpaint", "custom_rasterizer", "dist"),
    source_dir=os.path.join(script_dir, "hy3dpaint", "custom_rasterizer"),
)

# Install mesh_inpaint_processor
install_extension(
    name="mesh_inpaint_processor",
    module_name="mesh_inpaint_processor",
    dist_dir=os.path.join(script_dir, "hy3dpaint", "DifferentiableRenderer", "dist"),
    source_dir=os.path.join(script_dir, "hy3dpaint", "DifferentiableRenderer"),
)

# On macOS Apple Silicon, install MLX for ~3-5x faster diffusion inference.
# The node UI will show 'mlx' as a backend option when MLX is installed.
# Without MLX, only the 'pytorch' (MPS) backend is available on Mac.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    if not is_package_installed("mlx"):
        print("[Hunyuan3D-Paint] Installing MLX for Apple Silicon acceleration...")
        try:
            pip_install("mlx")
            print("[Hunyuan3D-Paint] MLX installed. Select 'mlx' in diffusion_backend for ~3-5x faster texture generation.")
        except subprocess.CalledProcessError:
            print("[Hunyuan3D-Paint] MLX installation failed. PyTorch MPS backend will still work.")
    else:
        print("[Hunyuan3D-Paint] MLX already installed.")
