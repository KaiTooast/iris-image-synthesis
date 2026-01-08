from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from PIL import Image
import os
import time
import sys
import random
import numpy as np
import io
import json
from datetime import datetime
import asyncio
import threading
import gc
import re
import base64
import subprocess

from src.core.config import Config
from src.utils.logger import create_logger
from src.utils.file_manager import FileManager



BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets"

logger = create_logger("IRISWebServer")

PROMPTS_LOG_FILE = Config.DATA_DIR / "prompts_history.json"

# Real-ESRGAN import handling
REALESRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    logger.success("Real-ESRGAN libraries available")
except ImportError as e:
    logger.warning(f"Real-ESRGAN not available: {e}")
    logger.info("Will use Lanczos upscaling as fallback")

# SwinIR import handling (assuming it's a separate library or module)
SWINIR_AVAILABLE = False
swinir_model = None
try:
    # This is a placeholder, replace with actual SwinIR import if available
    # from swinir_package import SwinIRModel
    # swinir_model = SWINIRModel(...)
    logger.info("SwinIR model not found or not configured. Upscaling will fall back to Real-ESRGAN or Lanczos.")
except ImportError as e:
    logger.warning(f"SwinIR not available: {e}")
    logger.info("Will use Real-ESRGAN or Lanczos upscaling as fallback")


connected_clients = []
gallery_clients = []
generation_stats = {
    "total_images": 0,
    "total_time": 0
}

# Global variables
pipe = None
img2img_pipe = None
device = None
upscaler = None
discord_bot_process = None
discord_bot_thread = None

# DRAM Extension Configuration
dram_extension_config = {
    "enabled": False,
    "vram_threshold_gb": 6,
    "max_dram_gb": 16
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI startup and shutdown"""
    logger.info("Starting I.R.I.S. Server...")
    
    try:
        # Startup: Load models
        await load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    logger.info("Shutting down I.R.I.S. Server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

async def load_models():
    """Load models on startup"""
    global pipe, img2img_pipe, device, upscaler, swinir_model
    
    logger.log_session_start()
    logger.info("=" * 70)
    logger.info("Starting AI Image Generator Backend...")
    logger.info("=" * 70)
    
    dtype = torch.float32
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        # Der Nvidia-Diss
        logger.success(f"NVIDIA GPU detected: {gpu_name}")
        logger.warning("💡 Note: Paying the 'Logo-Tax' today? Brace for the gatekeeping. 🖕🟢")
        
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.gpu_info(gpu_name, vram_total_gb, 0, 0)
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if vram_total_gb <= dram_extension_config["vram_threshold_gb"]:
            logger.info(f"DRAM Extension available: You can enable up to +{dram_extension_config['max_dram_gb']}GB system RAM for VRAM overflow")
            logger.info(f"Auto-enabling DRAM Extension for {vram_total_gb:.1f}GB VRAM card")
            dram_extension_config["enabled"] = True
        
        has_tensor_cores = any(arch in gpu_name.upper() for arch in ["RTX", "A100", "V100", "T4", "A10", "A40"])
        if has_tensor_cores:
            logger.success("Tensor Cores detected! Using float16 for optimal performance")
            dtype = torch.float16
        else:
            logger.warning("No Tensor Cores detected (GTX series). Using float32 for stability")
            dtype = torch.float32

    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        logger.success("Apple Silicon detected. Nice hardware, but stay open-minded.")

    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        dtype = torch.float32
        # Der Intel-Willkommensgruß
        logger.success("🔵 Intel Arc detected. WELCOME TO THE RESISTANCE!")
        logger.info("Enjoy your VRAM freedom. No 'Green Tax' detected here. 🌌")

    else:
        device = "cpu"
        dtype = torch.float32
        # Voller Respekt für die CPU-Geduld
        logger.info("=" * 70)
        logger.success("I.R.I.S. is running in Pure CPU Mode.")
        logger.info("Hand on heart: We respect the patience. True art takes time. 🎨☕")
        logger.info("=" * 70)
    
    logger.model_load_start("Ojimi/anime-kawai-diffusion")
    logger.info("This takes 5-10 minutes on first start (model will be downloaded)...")
    
    model_configs = {
        "anime_kawai": {
            "id": "Ojimi/anime-kawai-diffusion",
            "description": "Anime & Kawai Style"
        },
        "stable_diffusion_2_1": {
            "id": "stabilityai/stable-diffusion-2-1",
            "description": "Realistic Photo Style"
        },
        "stable_diffusion_3_5": {
            "id": "stabilityai/stable-diffusion-3.5-medium",
            "description": "High-Quality Realistic Style"
        },
        "flux_1_fast": {
            "id": "black-forest-labs/FLUX.1-schnell",
            "description": "Fast & Efficient Style"
        },
        "openjourney": {
            "id": "prompthero/openjourney",
            "description": "Artistic Illustration Style"
        },
        "pixel_art": {
            "id": "nitrosocke/pixel-art-diffusion",
            "description": "Pixel Art & Retro Style"
        },
        "pony_diffusion": {
        "id": "AstraliteHeart/pony-diffusion-v6-xl",
        "description": "High-End Anime & Character Style (SDXL)"
        },
        "anything_v5": {
         "id": "stablediffusionapi/anything-v5",
            "description": "Classic Flat Anime Style (Fast)"
        },
        "animagine_xl": {
            "id": "CagliostroResearchGroup/animagine-xl-3.1",
            "description": "High-Quality Modern Anime (SDXL)"
        },
        "aom3": {
         "id": "WarriorMama777/AbyssOrangeMix3",
         "description": "Semi-Realistic Anime Style"
        },
        "counterfeit_v3": {
         "id": "stablediffusionapi/counterfeit-v30",
         "description": "Detailed Digital Illustration Style"
         }
    }
    
    model_id = model_configs["anime_kawai"]["id"]

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    except Exception as e:
        logger.error(f"Failed to load primary model: {e}")
        raise
    
    if device == "cuda":
        logger.success("Enabling memory optimizations...")
        pipe.enable_attention_slicing(slice_size=1)
        
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
            logger.success("VAE slicing enabled (reduces memory for image processing)")
        
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
            logger.success("VAE tiling enabled (handles larger resolutions)")
        
        if dram_extension_config["enabled"]:
            apply_dram_extension(pipe, img2img_pipe)
        
        if dtype == torch.float16:
            pipe.vae.to(torch.float32)
            logger.success("Mixed Precision: Model float16 + VAE float32 (optimal)")
        
        torch.cuda.empty_cache()
        logger.success("CUDA cache cleared")
        logger.success("CUDA optimizations enabled")
    
    logger.info("Loading Image-to-Image pipeline...")
    img2img_pipe = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    
    if not dram_extension_config["enabled"]:
        img2img_pipe = img2img_pipe.to(device)
    
    if device == "cuda":
        img2img_pipe.enable_attention_slicing(slice_size=1)
        if hasattr(img2img_pipe, 'enable_vae_slicing'):
            img2img_pipe.enable_vae_slicing()
        if hasattr(img2img_pipe, 'enable_vae_tiling'):
            img2img_pipe.enable_vae_tiling()
        
        if dram_extension_config["enabled"]:
            apply_dram_extension(pipe, img2img_pipe)

    logger.success("Image-to-Image pipeline ready")
    logger.success("Model loaded successfully!")
    logger.info("=" * 70)
    logger.info(f"Server ready at http://localhost:8000")
    logger.info("=" * 70)
    
    upscaler = None
    if REALESRGAN_AVAILABLE and device == "cuda":
        try:
            logger.info("Loading Real-ESRGAN upscaler...")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upscaler = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(dtype == torch.float16),
                device=device
            )
            logger.success("Real-ESRGAN upscaler loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load Real-ESRGAN: {e}")
            logger.info("Will use Lanczos upscaling as fallback")
            upscaler = None

    if device == "cuda":
        torch.cuda.empty_cache()

def apply_dram_extension(txt2img_pipe=None, img2img_pipe_obj=None):
    """Enable DRAM as VRAM extension using sequential CPU offload"""
    global pipe, img2img_pipe
    
    if device != "cuda":
        logger.warning("DRAM extension only works with CUDA devices")
        return
    
    try:
        logger.info("Enabling DRAM Extension (VRAM + System RAM)...")
        logger.info("   This allows using system RAM to supplement VRAM")
        logger.info("   Model components will move between VRAM and RAM as needed")
        
        current_pipe = txt2img_pipe or pipe
        current_img2img = img2img_pipe_obj or img2img_pipe
        
        if current_pipe is not None:
            if hasattr(current_pipe, 'enable_sequential_cpu_offload'):
                logger.info("   Applying sequential CPU offload to text-to-image pipeline...")
                current_pipe.enable_sequential_cpu_offload()
                logger.success("   Text-to-image pipeline will use VRAM + System RAM")
            elif hasattr(current_pipe, 'enable_model_cpu_offload'):
                logger.info("   Applying model CPU offload to text-to-image pipeline...")
                current_pipe.enable_model_cpu_offload()
                logger.success("   Text-to-image pipeline will offload to RAM when needed")
        
        if current_img2img is not None:
            if hasattr(current_img2img, 'enable_sequential_cpu_offload'):
                logger.info("   Applying sequential CPU offload to image-to-image pipeline...")
                current_img2img.enable_sequential_cpu_offload()
                logger.success("   Image-to-image pipeline will use VRAM + System RAM")
            elif hasattr(current_img2img, 'enable_model_cpu_offload'):
                logger.info("   Applying model CPU offload to image-to-image pipeline...")
                current_img2img.enable_model_cpu_offload()
                logger.success("   Image-to-image pipeline will offload to RAM when needed")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.success(f"DRAM Extension enabled! VRAM: {vram_gb:.1f}GB + System RAM available")
        
    except Exception as e:
        logger.error(f"Failed to enable DRAM extension: {e}")

def log_prompt(prompt: str, settings: dict):
    """Log prompt to JSON file"""
    try:
        FileManager.log_prompt(prompt, settings)
    except Exception as e:
        logger.error(f"Failed to log prompt: {e}")

# --------------------------------------------------
# Prompt Normalization (Anti-Obfuscation)
# --------------------------------------------------

def normalize_prompt(text: str) -> str:
    """
    Normalize prompt to reduce simple obfuscation attempts
    (e.g. n1pple, n*pple, n i p p l e)
    """
    text = text.lower()

    replacements = {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "@": "a",
        "*": "",
        "_": "",
        "-": "",
        " ": ""
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# --------------------------------------------------
# Allowed non-sexual body & proportion descriptors
# --------------------------------------------------

ALLOWED_BODY_DESCRIPTORS = [
    "slim",
    "slender",
    "petite",
    "tall",
    "short",
    "athletic",
    "curvy",
    "wide hips",
    "narrow waist",
    "long legs",
    "soft curves",
    "stylized proportions",
    "anime proportions",
    "manga proportions",
    "fashion model body",
    "elegant posture"
]


# --------------------------------------------------
# Explicit sexual anatomy (HARD BLOCK)
# --------------------------------------------------

EXPLICIT_SEXUAL_TERMS = [
    r"\bnipple(s)?\b",
    r"\bareola\b",
    r"\bgenitals?\b",
    r"\bvagina\b",
    r"\bpenis\b",
    r"\bclitoris\b",
    r"\btesticle(s)?\b",
]


# --------------------------------------------------
# Sexual context keywords (HARD BLOCK)
# --------------------------------------------------

SEXUAL_CONTEXT = [
    r"\bnude\b",
    r"\bnaked\b",
    r"\bsex\b",
    r"\bsexual\b",
    r"\berotic\b",
    r"\bporn\b",
    r"\bhentai\b",
    r"\borgasm\b",
    r"\bmasturbat",
    r"\bintercourse\b",
]


# --------------------------------------------------
# Helper
# --------------------------------------------------

def contains_any(patterns, text):
    return any(re.search(p, text) for p in patterns)


# --------------------------------------------------
# Main NSFW Check
# --------------------------------------------------

def check_nsfw_prompt(prompt: str, nsfw_filter_enabled: bool = True) -> dict:
    """
    Intelligent NSFW filter:
    - Allows body proportions & artistic stylization
    - Blocks explicit sexual anatomy or sexual context
    - Resistant against simple obfuscation
    """

    if not nsfw_filter_enabled:
        return {
            "is_unsafe": False,
            "reason": "",
            "category": "",
            "message": "NSFW filter disabled"
        }

    prompt_lower = prompt.lower()
    normalized = normalize_prompt(prompt)

    has_explicit_anatomy = (
        contains_any(EXPLICIT_SEXUAL_TERMS, prompt_lower) or
        contains_any(EXPLICIT_SEXUAL_TERMS, normalized)
    )

    has_sexual_context = (
        contains_any(SEXUAL_CONTEXT, prompt_lower) or
        contains_any(SEXUAL_CONTEXT, normalized)
    )

    # --------------------------------------------------
    # HARD BLOCK: Explicit sexual content
    # --------------------------------------------------

    if has_explicit_anatomy or has_sexual_context:
        return {
            "is_unsafe": True,
            "category": "sexual",
            "reason": "explicit_content",
            "message": (
                "❌ Prompt blocked: Explicit sexual content detected.\n\n"
                "Please remove sexual anatomy or explicit sexual descriptions.\n"
                "Artistic body proportions and stylized characters are allowed."
            )
        }

    # --------------------------------------------------
    # SAFE: Artistic & proportional descriptions
    # --------------------------------------------------

    return {
        "is_unsafe": False,
        "reason": "",
        "category": "",
        "message": "Prompt is safe"
    }

def generate_filename(prefix: str, seed: int, steps: int = None, scale: int = None) -> str:
    """Generate filename with metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, timestamp, str(seed)]
    
    if steps:
        parts.append(f"s{steps}")
    if scale:
        parts.append(f"x{scale}")
    
    return "_".join(parts) + ".png"

def extract_metadata_from_filename(filename: str) -> dict:
    """Extract metadata from filename"""
    parts = filename.replace(".png", "").split("_")
    metadata = {
        "type": parts[0] if len(parts) > 0 else "unknown",
        "timestamp": parts[1] if len(parts) > 1 else None,
        "seed": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None,
        "steps": int(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("s") else None,
        "scale": int(parts[4][1:]) if len(parts) > 4 and parts[4].startswith("x") else None
    }
    return metadata

def log_prompt_history(filename: str, seed: int, prompt: str, steps: int):
    """Log prompt with associated image"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "seed": seed,
        "prompt": prompt,
        "steps": steps
    }
    
    try:
        FileManager.log_prompt(prompt, log_data)
    except Exception as e:
        logger.error(f"Failed to log prompt history: {e}")

app = FastAPI(title="I.R.I.S. API", version="1.0.0", lifespan=lifespan)

try:
    app.mount(
        "/assets",
        StaticFiles(directory=str(ASSETS_DIR)),
        name="assets"
    )
    logger.success(f"Assets directory mounted: {ASSETS_DIR}")
except Exception as e:
    logger.warning(f"Failed to mount assets directory: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Config.BASE_DIR / "frontend"

@app.get("/")
async def root():
    try:
        return FileResponse(FRONTEND_DIR / "index.html")
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        raise HTTPException(status_code=500, detail="Failed to load page")

@app.get("/generate")
async def generate_page():
    try:
        return FileResponse(FRONTEND_DIR / "generate.html")
    except Exception as e:
        logger.error(f"Error serving generate page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load page")

@app.get("/settings")
async def settings_page():
    try:
        return FileResponse(FRONTEND_DIR / "settings.html")
    except Exception as e:
        logger.error(f"Error serving settings page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load page")

@app.get("/gallery")
async def gallery_page():
    try:
        return FileResponse(FRONTEND_DIR / "gallery.html")
    except Exception as e:
        logger.error(f"Error serving gallery page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load page")

@app.get("/favicon.ico")
async def favicon():
    favicon_path = ASSETS_DIR / "fav.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": device
    }

@app.get("/api/system")
async def get_system_info():
    """Get system information"""
    info = {
        "gpu_name": "Unknown",
        "device": device,
        "vram_total": 0.0,
        "vram_used": 0.0,
        "gpu_temp": 0.0,
        "dram_extension_enabled": dram_extension_config["enabled"],
        "dram_extension_available": False
    }
    
    if device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["vram_total"] = vram_total
            info["vram_used"] = torch.cuda.memory_allocated(0) / 1024**3
            
            info["dram_extension_available"] = vram_total <= dram_extension_config["vram_threshold_gb"]
            
            # Get GPU temp
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=2
                )
                info["gpu_temp"] = float(result.stdout.strip())
            except:
                info["gpu_temp"] = 0
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
    
    return info

@app.get("/api/stats")
async def get_stats():
    """Get generation statistics"""
    # Placeholder for actual stats if needed, currently not tracked globally
    return {
        "total_images": generation_stats["total_images"],
        "total_time": round(generation_stats["total_time"], 2),
        # Placeholder for GPU temp and VRAM used, would require separate calls or periodic updates
        "gpu_temp": 0, 
        "vram_used": 0
    }

@app.get("/api/version")
async def get_version_info():
    """Get version information for I.R.I.S. and dependencies"""
    import sys
    
    version_info = {
        "iris_version": "1.0.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "sd_model": "Ojimi/anime-kawai-diffusion",
        "realesrgan_available": REALESRGAN_AVAILABLE,
        "swinir_available": SWINIR_AVAILABLE # Added SwinIR availability
    }
    
    return version_info

# ========== MODELS ==========
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, out of focus, duplicate, ugly, morbid, mutilated, mutated hands, poorly drawn face, mutation, deformed, dehydrated, bad proportions, gross proportions, malformed limbs, long neck"
    style: str = "anime_kawai"
    seed: Optional[int] = -1 # Handle -1 as random seed
    steps: int = 35
    cfg_scale: float = 10.0
    width: int = 512
    height: int = 768
    batch_size: int = 1
    dram_extension_enabled: Optional[bool] = False
    nsfw_filter_enabled: Optional[bool] = True

class UpscaleRequest(BaseModel):
    filename: str
    scale: int = 2
    method: str = "realesrgan"  # Options: "realesrgan", "swinir", "lanczos"

class VariationRequest(BaseModel):
    filename: str
    strength: float = 0.5
    prompt: Optional[str] = ""

class SystemInfo(BaseModel):
    gpu_name: str
    device: str
    vram_total: float
    vram_used: float
    gpu_temp: float
    dram_extension_enabled: bool
    dram_extension_available: bool

# Helper function to auto-adjust parameters based on VRAM
def get_safe_generation_params(width: int, height: int, steps: int, vram_gb: float):
    """
    Automatically adjust generation parameters to prevent OOM on low VRAM systems
    Returns safe (width, height, steps) based on available VRAM
    """
    if dram_extension_config["enabled"]:
        logger.info(f"   🔄 DRAM Extension active - using full parameters: {width}x{height}, {steps} steps")
        return width, height, steps
    
    total_pixels = width * height
    
    # VRAM requirements (rough estimates):
    # 512x512 (20 steps) = ~2.5GB
    # 512x768 (20 steps) = ~3.2GB
    # 512x768 (35 steps) = ~3.8GB
    # 720x1280 (35 steps) = ~5.0GB  // Added HD mobile wallpaper
    # 768x768 (35 steps) = ~5.5GB
    # 1080x1920 (35 steps) = ~7.5GB
    
    if vram_gb <= 4:
        # Very conservative for 4GB cards WITHOUT DRAM extension
        if total_pixels > 512 * 512:
            # Force smaller resolution
            width = 512
            height = 512
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for 4GB VRAM (enable DRAM extension to use higher)")
        
        if steps > 25:
            steps = 25
            logger.warning(f"⚠️  Auto-adjusted steps to {steps} for 4GB VRAM (enable DRAM extension to use 35+)")
            
    elif vram_gb <= 6:
        # Conservative for 6GB cards
        if total_pixels > 720 * 1280:
            width = 720
            height = 1280
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for 6GB VRAM")
        
        if steps > 35:
            steps = 35
            logger.warning(f"⚠️  Auto-adjusted steps to {steps} for 6GB VRAM")
    
    elif vram_gb <= 8:
        # Moderate limits for 8GB cards - can handle mobile wallpaper and larger resolutions
        if total_pixels > 1080 * 1920:
            # Scale down proportionally
            scale_factor = (1080 * 1920 / total_pixels) ** 0.5
            width = int(width * scale_factor / 64) * 64  # Round to multiple of 64
            height = int(height * scale_factor / 64) * 64
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for 8GB VRAM")
    
    elif vram_gb > 8:
        # High VRAM cards (10GB+) can handle mobile wallpaper and larger resolutions
        if total_pixels > 1920 * 1080: # Check for common high-res like 1920x1080
            scale_factor = (1920 * 1080 / total_pixels) ** 0.5
            width = int(width * scale_factor / 64) * 64
            height = int(height * scale_factor / 64) * 64
            logger.warning(f"⚠️  Auto-adjusted resolution to {width}x{height} for {vram_gb:.0f}GB VRAM")
    
    return width, height, steps

# ========== ROUTES ==========

@app.get("/")
async def root():
    """Serve the HTML interface"""
    try:
        return FileResponse(FRONTEND_DIR / "index.html")
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        raise HTTPException(status_code=500, detail="Failed to load page")

@app.get("/generate")
async def generate_page():
    """Serve the image generation page"""
    try:
        return FileResponse(FRONTEND_DIR / "generate.html")
    except Exception as e:
        logger.error(f"Error serving generate page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load page")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": device
    }

@app.get("/api/system")
async def get_system_info():
    """Get system information"""
    info = {
        "gpu_name": "Unknown",
        "device": device,
        "vram_total": 0.0,
        "vram_used": 0.0,
        "gpu_temp": 0.0,
        "dram_extension_enabled": dram_extension_config["enabled"],
        "dram_extension_available": False
    }
    
    if device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["vram_total"] = vram_total
            info["vram_used"] = torch.cuda.memory_allocated(0) / 1024**3
            
            info["dram_extension_available"] = vram_total <= dram_extension_config["vram_threshold_gb"]
            
            # Get GPU temp (requires nvidia-smi)
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=2
                )
                info["gpu_temp"] = float(result.stdout.strip())
            except:
                info["gpu_temp"] = 0
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
    
    return info

@app.get("/api/stats")
async def get_stats():
    """Get generation statistics"""
    # Placeholder for actual stats if needed, currently not tracked globally
    return {
        "total_images": generation_stats["total_images"],
        "total_time": round(generation_stats["total_time"], 2),
        # Placeholder for GPU temp and VRAM used, would require separate calls or periodic updates
        "gpu_temp": 0, 
        "vram_used": 0
    }

@app.get("/api/version")
async def get_version_info():
    """Get version information for I.R.I.S. and dependencies"""
    import sys
    
    version_info = {
        "iris_version": "1.0.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "sd_model": "Ojimi/anime-kawai-diffusion",
        "realesrgan_available": REALESRGAN_AVAILABLE,
        "swinir_available": SWINIR_AVAILABLE
    }
    
    return version_info

@app.get("/api/gpu-info")
async def get_gpu_info():
    """Get current GPU information"""
    info = {
        "gpu_name": "Unknown",
        "vram_total": 0.0,
        "vram_used": 0.0,
        "vram_free": 0.0,
        "vram_percent": 0.0,
        "gpu_temp": 0.0,
        "power_draw": 0.0,
        "gpu_utilization": 0.0
    }
    
    if device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_used = torch.cuda.memory_allocated(0) / 1024**3
            vram_free = vram_total - vram_used
            
            info["vram_total"] = round(vram_total, 2)
            info["vram_used"] = round(vram_used, 2)
            info["vram_free"] = round(vram_free, 2)
            info["vram_percent"] = round((vram_used / vram_total) * 100, 1) if vram_total > 0 else 0
            
            # Get GPU stats from nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    info["gpu_temp"] = float(parts[0].strip())
                    info["power_draw"] = float(parts[1].strip())
                    info["gpu_utilization"] = float(parts[2].strip())
            except:
                pass
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
    
    return info

@app.get("/api/rpc-status")
async def get_rpc_status():
    """Get Discord RPC status"""
    return {
        "connected": False,  # Local system doesn't need RPC
        "status": "idle",
        "details": "Ready to generate",
        "generation_count": 0
    }

@app.get("/api/prompts-history")
async def get_prompts_history():
    """Get prompt history from prompts_history.json"""
    try:
        if PROMPTS_LOG_FILE.exists():
            with open(PROMPTS_LOG_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                # Return last 50 entries
                return {"history": history[-50:] if len(history) > 50 else history}
        return {"history": []}
    except Exception as e:
        logger.error(f"Error loading prompt history: {e}")
        return {"history": []}


@app.get("/api/output-gallery")
async def get_output_gallery():
    """Get list of output images"""
    try:
        os.makedirs("outputs", exist_ok=True)
        files = [f for f in os.listdir("outputs") if f.endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda x: os.path.getmtime(f"outputs/{x}"), reverse=True)
        return {"images": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error loading output gallery: {e}")
        return {"images": [], "count": 0, "error": str(e)}

@app.get("/api/output-image/{filename}")
async def get_output_image(filename: str):
    """Get a specific image from outputs directory"""
    filepath = f"outputs/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.post("/api/recreate-from-output")
async def recreate_from_output(request: dict):
    """Extract settings from output filename for recreation"""
    try:
        filename = request.get("filename", "")
        
        metadata = extract_metadata_from_filename(filename)
        
        if metadata["seed"] is not None:
            return {
                "success": True,
                "seed": metadata["seed"],
                "message": f"Extracted seed: {metadata['seed']}, type: {metadata['type']}"
            }
        
        parts = filename.replace(".png", "").split("_")
        
        seed = None
        steps = None
        
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) >= 4:
                if seed is None:
                    seed = int(part)
                else:
                    steps = int(part)
        
        if seed is None:
            return {"success": False, "error": "Could not extract seed from filename"}
        
        return {
            "success": True,
            "seed": seed,
            "steps": steps,
            "message": f"Extracted seed: {seed}" + (f", steps: {steps}" if steps else "")
        }
        
    except Exception as e:
        logger.error(f"Recreation failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/gallery")
async def get_gallery():
    """Get list of generated images"""
    os.makedirs("outputs", exist_ok=True)
    files = sorted(
        [f for f in os.listdir("outputs") if f.endswith(".png")],
        key=lambda x: os.path.getmtime(f"outputs/{x}"),
        reverse=True
    )[:12]
    
    return {"images": files}

@app.post("/api/generate")
async def generate_image(request: GenerationRequest):
    """Generate image (non-streaming version for API calls)"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Log prompt before generation starts
        log_prompt(request.prompt, request.model_dump())

        if device == "cuda":
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            request.width, request.height, request.steps = get_safe_generation_params(
                request.width, request.height, request.steps, vram_total
            )
        
        # Prepare seed
        seed = request.seed if request.seed != -1 else np.random.randint(0, 2147483647) # Handle -1 as random seed
        generator = torch.Generator(device).manual_seed(seed)
        
        # Adjust prompt based on style
        if request.style == "pixel_art": # UPDATED KEY
            full_prompt = f"pixel art, 16-bit style, {request.prompt}"
            neg_prompt = f"smooth, anti-aliased, {request.negative_prompt}"
        else:
            full_prompt = f"masterpiece, best quality, {request.prompt}"
            neg_prompt = request.negative_prompt or "lowres, bad anatomy, bad hands, worst quality"
        
        start_time = time.time()
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Generate
        result = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            width=request.width,
            height=request.height,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        # Convert image to base64
        image = result.images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        now = datetime.now()
        filename = generate_filename("generated", seed, request.steps)
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        # Call new prompt logging function
        log_prompt_history(filename, seed, request.prompt, request.steps)
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "seed": seed,
            "generation_time": round(generation_time, 2),
            "filename": filename
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            error_msg = (
                "💥 CUDA Out of Memory!\n\n"
                "Your GPU ran out of memory during generation. "
                "Try reducing resolution, steps, or enabling DRAM Extension in advanced settings."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        else:
            raise e
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    logger.websocket_connect()
    
    try:
        while True:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            data = await websocket.receive_text()
            logger.info(f"📨 Received data: {data[:100]}...")
            
            request_data = json.loads(data)
            
            # Log prompt before generation starts via WebSocket
            log_prompt(request_data.get('prompt', ''), request_data)

            if device == "cuda":
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                original_width = request_data.get("width", 512)
                original_height = request_data.get("height", 768)
                original_steps = request_data.get("steps", 35)
                
                adjusted_width, adjusted_height, adjusted_steps = get_safe_generation_params(
                    original_width, original_height, original_steps, vram_total
                )
                
                request_data["width"] = adjusted_width
                request_data["height"] = adjusted_height
                request_data["steps"] = adjusted_steps
                
                if (adjusted_width != original_width or adjusted_height != original_height or adjusted_steps != original_steps):
                    await websocket.send_json({
                        "type": "warning",
                        "message": f"Parameters auto-adjusted for low VRAM: {adjusted_width}x{adjusted_height}, {adjusted_steps} steps"
                    })
            
            logger.info("🎨 Starting generation...")
            logger.info(f"   Prompt: {request_data.get('prompt', 'N/A')[:60]}...")
            logger.info(f"   Style: {request_data.get('style', 'anime_kawai')}") # Updated default
            logger.info(f"   Steps: {request_data.get('steps', 35)} (requested)")
            
            if device == "cuda":
                available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                logger.info(f"   Available VRAM: {available_vram:.2f} GB")
            
            websocket_active = True
            
            async def safe_send(data):
                """Safely send data through WebSocket with error handling"""
                nonlocal websocket_active
                if not websocket_active:
                    return False
                try:
                    await websocket.send_json(data)
                    return True
                except Exception as e:
                    logger.error(f"WebSocket send failed (client likely disconnected): {type(e).__name__}")
                    websocket_active = False
                    return False
            
            async def broadcast_to_gallery(data):
                """Broadcast generation progress to gallery clients"""
                disconnected = []
                for client in gallery_clients:
                    try:
                        await client.send_json(data)
                    except Exception as e:
                        logger.debug(f"Gallery client send failed: {e}")
                        disconnected.append(client)
                
                for client in disconnected:
                    if client in gallery_clients:
                        gallery_clients.remove(client)
            
            if not await safe_send({"type": "started", "message": "Generation started"}):
                break
            
            await broadcast_to_gallery({
                "status": "generating",
                "progress": 0,
                "step": 0,
                "total_steps": request_data.get("steps", 35),
                "total_images": 1,
                "current_image": 1
            })
            
            seed = request_data.get("seed")
            if seed == -1 or seed is None: # Handle -1 seed in websocket endpoint
                seed = np.random.randint(0, 2147483647)
            generator = torch.Generator(device).manual_seed(seed)
            
            style = request_data.get("style", "anime_kawai") # Updated default
            prompt = request_data.get("prompt", "")
            nsfw_filter_enabled = request_data.get("nsfw_filter_enabled", True)
            
            # Check if prompt contains NSFW content BEFORE generation
            nsfw_check = check_nsfw_prompt(prompt, nsfw_filter_enabled)
            if nsfw_check["is_unsafe"]:
                logger.warning(f"NSFW prompt blocked: {nsfw_check['reason']}")
                await safe_send({
                    "type": "error",
                    "message": nsfw_check["message"],
                    "nsfw_blocked": True
                })
                break
            
            if style == "pixel_art": # UPDATED KEY
                full_prompt = f"pixel art, 16-bit style, {prompt}"
                neg_prompt = "smooth, anti-aliased, blurry, 3d render"
            else:
                full_prompt = f"masterpiece, best quality, {prompt}"
                neg_prompt = "lowres, bad anatomy, worst quality, blurry"
            
            user_neg = request_data.get("negative_prompt", "")
            if user_neg:
                neg_prompt = f"{neg_prompt}, {user_neg}"
            
            start_time = time.time()
            total_steps = request_data.get("steps", 35)
            
            step_times = []
            
            def progress_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict):
                nonlocal websocket_active
                if not websocket_active:
                    return callback_kwargs
                
                current_time = time.time()
                step_times.append(current_time)
                
                progress = (step + 1) / total_steps * 100
                
                avg_step_time = 0
                eta = 0
                if len(step_times) > 1:
                    avg_step_time = (current_time - start_time) / len(step_times)
                    remaining_steps = total_steps - (step + 1)
                    eta = avg_step_time * remaining_steps
                
                gpu_temp = 0
                vram_used = 0
                if device == "cuda":
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=1
                        )
                        temp_str, vram_str = result.stdout.strip().split(',')
                        gpu_temp = float(temp_str)
                        vram_used = float(vram_str) / 1024
                    except:
                        try:
                            vram_used = torch.cuda.memory_allocated(0) / 1024**3
                        except:
                            vram_used = 0
                
                bar_length = 50
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                elapsed_time = current_time - start_time
                print(f"\r🎨 [{bar}] {progress:.0f}% │ {step + 1}/{total_steps} │ ⏱️ {elapsed_time:.0f}s │ ETA: {eta:.0f}s │ 🌡️ {gpu_temp:.0f}°C │ 💾 {vram_used:.1f}GB", end='', flush=True)
                
                async def send_progress():
                    progress_data = {
                        "type": "progress",
                        "step": step + 1,
                        "total_steps": total_steps,
                        "progress": round(progress, 1),
                        "eta": round(eta, 1),
                        "gpu_temp": round(gpu_temp, 1),
                        "vram_used": round(vram_used, 2),
                        "avg_step_time": round(avg_step_time, 2)
                    }
                    await safe_send(progress_data)
                    
                    await broadcast_to_gallery({
                        "status": "generating",
                        "progress": round(progress, 1),
                        "step": step + 1,
                        "total_steps": total_steps,
                        "total_images": 1,
                        "current_image": 1
                    })
                
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(send_progress())
                except RuntimeError:
                    pass
                
                return callback_kwargs

            try:
                # Check if model is loaded
                if pipe is None:
                    error_msg = "Model not loaded yet. Please wait for the server to finish loading models on startup."
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "model_not_loaded",
                        "message": error_msg
                    })
                    break
                
                # Prepare kwargs for pipe call
                pipe_kwargs = {
                    "prompt": full_prompt,
                    "negative_prompt": neg_prompt,
                    "num_inference_steps": total_steps,
                    "guidance_scale": request_data.get("cfg_scale", 10.0),
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768),
                    "generator": generator
                }
                
                # Only add callback if pipe supports it
                try:
                    if pipe is not None and (hasattr(pipe, 'callback_on_step_end') or 'callback_on_step_end' in pipe.__call__.__code__.co_varnames):
                        pipe_kwargs["callback_on_step_end"] = progress_callback
                except (AttributeError, TypeError):
                    # Callback not supported, continue without it
                    pass
                
                result = pipe(**pipe_kwargs)
                
                generation_time = time.time() - start_time
                print()
                logger.success(f"Generation completed in {generation_time:.1f}s")
                
                image = result.images[0]
                
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                model_name = "Ojimi/anime-kawai-diffusion" # Hardcoded for now, could be dynamic
                prompt_data = request_data.get('prompt', '')
                negative_prompt_data = request_data.get('negative_prompt', '')
                width_data = request_data.get('width', 512)
                height_data = request_data.get('height', 768)
                steps_data = request_data.get('steps', 35)
                cfg_scale_data = request_data.get('cfg_scale', 10.0)
                style_data = request_data.get('style', 'anime_kawai')

                filename = generate_filename("generated", seed, total_steps)
                os.makedirs("outputs", exist_ok=True)
                image.save(f"outputs/{filename}")
                
                logger.info(f"💾 Saved as: {filename}")
                
                from src.utils.file_manager import FileManager
                FileManager.log_prompt(
                    prompt=prompt_data,
                    settings={
                        "negative_prompt": negative_prompt_data,
                        "seed": seed,
                        "width": width_data,
                        "height": height_data,
                        "steps": steps_data,
                        "cfg_scale": cfg_scale_data,
                        "style": style_data,
                        "model": model_name,
                        "filename": filename
                    }
                )

                success = await safe_send({
                    "type": "completed",
                    "image": f"data:image/png;base64,{img_str}",
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "filename": filename,
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768)
                })
                
                await broadcast_to_gallery({
                    "status": "complete",
                    "progress": 100,
                    "filename": filename,
                    "image": f"data:image/png;base64,{img_str}"
                })
                
                if not success:
                    logger.warning("⚠️  Client disconnected, but image was saved successfully")
                    break
                
                # Update global stats (this might need to be more robust for concurrent access)
                generation_stats["total_images"] += 1
                generation_stats["total_time"] += generation_time
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else 0
                    error_msg = (
                        "💥 CUDA Out of Memory!\n\n"
                        f"Your GPU has {vram_gb:.1f}GB VRAM but this generation requires more.\n\n"
                        "💡 Solutions (try in order):\n"
                        "1. ✅ Enable DRAM Extension in Advanced Settings\n"
                        "   → This adds 8GB system RAM to supplement your VRAM\n"
                        "2. 📐 Use smaller resolution:\n"
                        "   → 512x512 (safest for 4GB VRAM)\n"
                        "   → 512x768 (safe for 6GB VRAM)\n"
                        "3. ⚡ Reduce steps:\n"
                        "   → Use 'Fast' preset (20 steps)\n"
                        "   → Or manually set to 20-25 steps\n"
                        "4. 🔄 Restart the server:\n"
                        "   → This clears cached memory\n"
                        "5. 🎯 Alternative workflow:\n"
                        "   → Generate at 512x512, then Upscale 2x\n"
                        "   → Much faster and uses less VRAM!\n\n"
                        "Recommended settings for your GPU:\n"
                        f"• Resolution: {'512x512' if vram_gb <= 4 else '512x768' if vram_gb <= 6 else '768x768'}\n"
                        "• Steps: 20-28\n"
                        "• Quality: Fast or Balanced\n"
                        "• DRAM Extension: Enabled"
                    )
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "cuda_oom",
                        "message": error_msg
                    })
                    
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            gc.collect()
                        except:
                            pass
                    
                    break
                else:
                    raise e

            # Log prompt history after successful generation or to catch errors
            log_prompt_history(filename, seed, prompt, total_steps)
            
    except WebSocketDisconnect:
        logger.websocket_disconnect()
    except asyncio.CancelledError:
        logger.websocket_disconnect()
    except Exception as e:
        error_name = type(e).__name__
        if "Disconnect" not in error_name and "Closed" not in error_name:
            logger.error(f"WebSocket error: {error_name}: {e}")

            error_type = "unknown"
            error_message = str(e)
            
            if "out of memory" in error_message.lower() or "cuda" in error_message.lower():
                error_type = "cuda_oom"
                logger.error("💥 CUDA Out of Memory Error!")
                logger.error("Solutions:")
                logger.error("  • Enable DRAM Extension in Advanced Settings")
                logger.error("  • Use 512x512 resolution")
                logger.error("  • Reduce steps to 20-25")
                logger.error("  • Select 'Fast' quality preset")
                logger.error("  • Restart server to clear memory")
            elif "connection" in error_message.lower() or "refused" in error_message.lower() or "reset by peer" in error_message.lower():
                error_type = "connection"
                logger.error("🔌 Connection Error!")
                logger.error("Solutions:")
                logger.error("  • Check if backend server is running")
                logger.error("  • Verify port 8000 is not blocked")
                logger.error("  • Check firewall settings")
            else:
                error_type = "generic"
                logger.error(f"❌ Error: {error_message}")
            
            async def send_error_to_client():
                try:
                    if websocket_active:
                        await websocket.send_json({
                            "type": "error",
                            "error_type": error_type,
                            "message": error_message
                        })
                except RuntimeError as e:
                    logger.debug(f"Could not send error to client (WebSocket already closed): {e}")
            
            try:
                asyncio.create_task(send_error_to_client())
            except RuntimeError:
                pass
            
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.websocket_disconnect()

@app.websocket("/ws/gallery-progress")
async def websocket_gallery_progress(websocket: WebSocket):
    """WebSocket endpoint for gallery to receive generation progress updates"""
    await websocket.accept()
    gallery_clients.append(websocket)
    
    logger.info("📸 Gallery client connected for progress updates")
    
    try:
        while True:
            try:
                await websocket.receive_text()
            except:
                break
            await asyncio.sleep(1)
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Gallery WebSocket error: {e}")
    finally:
        if websocket in gallery_clients:
            gallery_clients.remove(websocket)
        logger.info("📸 Gallery client disconnected")

@app.get("/api/output-gallery")
async def get_output_gallery():
    """Get list of all images from outputs directory"""
    try:
        os.makedirs("outputs", exist_ok=True)
        files = [f for f in os.listdir("outputs") if f.endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda x: os.path.getmtime(f"outputs/{x}"), reverse=True)
        return {"images": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error loading output gallery: {e}")
        return {"images": [], "count": 0, "error": str(e)}

@app.get("/api/output-image/{filename}")
async def get_output_image(filename: str):
    """Get a specific image from outputs directory"""
    filepath = f"outputs/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.post("/api/recreate-from-output")
async def recreate_from_output(request: dict):
    """Extract settings from output filename for recreation"""
    try:
        filename = request.get("filename", "")
        
        metadata = extract_metadata_from_filename(filename)
        
        if metadata["seed"] is not None:
            return {
                "success": True,
                "seed": metadata["seed"],
                "message": f"Extracted seed: {metadata['seed']}, type: {metadata['type']}"
            }
        
        parts = filename.replace(".png", "").split("_")
        
        seed = None
        steps = None
        
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) >= 4:
                if seed is None:
                    seed = int(part)
                else:
                    steps = int(part)
        
        if seed is None:
            return {"success": False, "error": "Could not extract seed from filename"}
        
        return {
            "success": True,
            "seed": seed,
            "steps": steps,
            "message": f"Extracted seed: {seed}" + (f", steps: {steps}" if steps else "")
        }
        
    except Exception as e:
        logger.error(f"Recreation failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/gallery")
async def get_gallery():
    """Get list of generated images"""
    os.makedirs("outputs", exist_ok=True)
    files = sorted(
        [f for f in os.listdir("outputs") if f.endswith(".png")],
        key=lambda x: os.path.getmtime(f"outputs/{x}"),
        reverse=True
    )[:12]
    
    return {"images": files}

@app.post("/api/generate")
async def generate_image(request: GenerationRequest):
    """Generate image (non-streaming version for API calls)"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Log prompt before generation starts
        log_prompt(request.prompt, request.model_dump())

        if device == "cuda":
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            request.width, request.height, request.steps = get_safe_generation_params(
                request.width, request.height, request.steps, vram_total
            )
        
        # Prepare seed
        seed = request.seed if request.seed != -1 else np.random.randint(0, 2147483647) # Handle -1 as random seed
        generator = torch.Generator(device).manual_seed(seed)
        
        # Adjust prompt based on style
        if request.style == "pixel_art": # UPDATED KEY
            full_prompt = f"pixel art, 16-bit style, {request.prompt}"
            neg_prompt = f"smooth, anti-aliased, {request.negative_prompt}"
        else:
            full_prompt = f"masterpiece, best quality, {request.prompt}"
            neg_prompt = request.negative_prompt or "lowres, bad anatomy, bad hands, worst quality"
        
        start_time = time.time()
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Generate
        result = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            width=request.width,
            height=request.height,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        # Convert image to base64
        image = result.images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        now = datetime.now()
        filename = generate_filename("generated", seed, request.steps)
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        # Call new prompt logging function
        log_prompt_history(filename, seed, request.prompt, request.steps)
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "seed": seed,
            "generation_time": round(generation_time, 2),
            "filename": filename
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            error_msg = (
                "💥 CUDA Out of Memory!\n\n"
                "Your GPU ran out of memory during generation. "
                "Try reducing resolution, steps, or enabling DRAM Extension in advanced settings."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        else:
            raise e
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    logger.websocket_connect()
    
    try:
        while True:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            data = await websocket.receive_text()
            logger.info(f"📨 Received data: {data[:100]}...")
            
            request_data = json.loads(data)
            
            # Log prompt before generation starts via WebSocket
            log_prompt(request_data.get('prompt', ''), request_data)

            if device == "cuda":
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                original_width = request_data.get("width", 512)
                original_height = request_data.get("height", 768)
                original_steps = request_data.get("steps", 35)
                
                adjusted_width, adjusted_height, adjusted_steps = get_safe_generation_params(
                    original_width, original_height, original_steps, vram_total
                )
                
                request_data["width"] = adjusted_width
                request_data["height"] = adjusted_height
                request_data["steps"] = adjusted_steps
                
                if (adjusted_width != original_width or adjusted_height != original_height or adjusted_steps != original_steps):
                    await websocket.send_json({
                        "type": "warning",
                        "message": f"Parameters auto-adjusted for low VRAM: {adjusted_width}x{adjusted_height}, {adjusted_steps} steps"
                    })
            
            logger.info("🎨 Starting generation...")
            logger.info(f"   Prompt: {request_data.get('prompt', 'N/A')[:60]}...")
            logger.info(f"   Style: {request_data.get('style', 'anime_kawai')}") # Updated default
            logger.info(f"   Steps: {request_data.get('steps', 35)} (requested)")
            
            if device == "cuda":
                available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                logger.info(f"   Available VRAM: {available_vram:.2f} GB")
            
            websocket_active = True
            
            async def safe_send(data):
                """Safely send data through WebSocket with error handling"""
                nonlocal websocket_active
                if not websocket_active:
                    return False
                try:
                    await websocket.send_json(data)
                    return True
                except Exception as e:
                    logger.error(f"WebSocket send failed (client likely disconnected): {type(e).__name__}")
                    websocket_active = False
                    return False
            
            async def broadcast_to_gallery(data):
                """Broadcast generation progress to gallery clients"""
                disconnected = []
                for client in gallery_clients:
                    try:
                        await client.send_json(data)
                    except Exception as e:
                        logger.debug(f"Gallery client send failed: {e}")
                        disconnected.append(client)
                
                for client in disconnected:
                    if client in gallery_clients:
                        gallery_clients.remove(client)
            
            if not await safe_send({"type": "started", "message": "Generation started"}):
                break
            
            await broadcast_to_gallery({
                "status": "generating",
                "progress": 0,
                "step": 0,
                "total_steps": request_data.get("steps", 35),
                "total_images": 1,
                "current_image": 1
            })
            
            seed = request_data.get("seed")
            if seed == -1 or seed is None: # Handle -1 seed in websocket endpoint
                seed = np.random.randint(0, 2147483647)
            generator = torch.Generator(device).manual_seed(seed)
            
            style = request_data.get("style", "anime_kawai") # Updated default
            prompt = request_data.get("prompt", "")
            nsfw_filter_enabled = request_data.get("nsfw_filter_enabled", True)
            
            # Check if prompt contains NSFW content BEFORE generation
            nsfw_check = check_nsfw_prompt(prompt, nsfw_filter_enabled)
            if nsfw_check["is_unsafe"]:
                logger.warning(f"NSFW prompt blocked: {nsfw_check['reason']}")
                await safe_send({
                    "type": "error",
                    "message": nsfw_check["message"],
                    "nsfw_blocked": True
                })
                break
            
            if style == "pixel_art": # UPDATED KEY
                full_prompt = f"pixel art, 16-bit style, {prompt}"
                neg_prompt = "smooth, anti-aliased, blurry, 3d render"
            else:
                full_prompt = f"masterpiece, best quality, {prompt}"
                neg_prompt = "lowres, bad anatomy, worst quality, blurry"
            
            user_neg = request_data.get("negative_prompt", "")
            if user_neg:
                neg_prompt = f"{neg_prompt}, {user_neg}"
            
            start_time = time.time()
            total_steps = request_data.get("steps", 35)
            
            step_times = []
            
            def progress_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict):
                nonlocal websocket_active
                if not websocket_active:
                    return callback_kwargs
                
                current_time = time.time()
                step_times.append(current_time)
                
                progress = (step + 1) / total_steps * 100
                
                avg_step_time = 0
                eta = 0
                if len(step_times) > 1:
                    avg_step_time = (current_time - start_time) / len(step_times)
                    remaining_steps = total_steps - (step + 1)
                    eta = avg_step_time * remaining_steps
                
                gpu_temp = 0
                vram_used = 0
                if device == "cuda":
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=1
                        )
                        temp_str, vram_str = result.stdout.strip().split(',')
                        gpu_temp = float(temp_str)
                        vram_used = float(vram_str) / 1024
                    except:
                        try:
                            vram_used = torch.cuda.memory_allocated(0) / 1024**3
                        except:
                            vram_used = 0
                
                bar_length = 50
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                elapsed_time = current_time - start_time
                print(f"\r🎨 [{bar}] {progress:.0f}% │ {step + 1}/{total_steps} │ ⏱️ {elapsed_time:.0f}s │ ETA: {eta:.0f}s │ 🌡️ {gpu_temp:.0f}°C │ 💾 {vram_used:.1f}GB", end='', flush=True)
                
                async def send_progress():
                    progress_data = {
                        "type": "progress",
                        "step": step + 1,
                        "total_steps": total_steps,
                        "progress": round(progress, 1),
                        "eta": round(eta, 1),
                        "gpu_temp": round(gpu_temp, 1),
                        "vram_used": round(vram_used, 2),
                        "avg_step_time": round(avg_step_time, 2)
                    }
                    await safe_send(progress_data)
                    
                    await broadcast_to_gallery({
                        "status": "generating",
                        "progress": round(progress, 1),
                        "step": step + 1,
                        "total_steps": total_steps,
                        "total_images": 1,
                        "current_image": 1
                    })
                
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(send_progress())
                except RuntimeError:
                    pass
                
                return callback_kwargs

            try:
                # Check if model is loaded
                if pipe is None:
                    error_msg = "Model not loaded yet. Please wait for the server to finish loading models on startup."
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "model_not_loaded",
                        "message": error_msg
                    })
                    break
                
                # Prepare kwargs for pipe call
                pipe_kwargs = {
                    "prompt": full_prompt,
                    "negative_prompt": neg_prompt,
                    "num_inference_steps": total_steps,
                    "guidance_scale": request_data.get("cfg_scale", 10.0),
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768),
                    "generator": generator
                }
                
                # Only add callback if pipe supports it
                try:
                    if pipe is not None and (hasattr(pipe, 'callback_on_step_end') or 'callback_on_step_end' in pipe.__call__.__code__.co_varnames):
                        pipe_kwargs["callback_on_step_end"] = progress_callback
                except (AttributeError, TypeError):
                    # Callback not supported, continue without it
                    pass
                
                result = pipe(**pipe_kwargs)
                
                generation_time = time.time() - start_time
                print()
                logger.success(f"Generation completed in {generation_time:.1f}s")
                
                image = result.images[0]
                
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                model_name = "Ojimi/anime-kawai-diffusion" # Hardcoded for now, could be dynamic
                prompt_data = request_data.get('prompt', '')
                negative_prompt_data = request_data.get('negative_prompt', '')
                width_data = request_data.get('width', 512)
                height_data = request_data.get('height', 768)
                steps_data = request_data.get('steps', 35)
                cfg_scale_data = request_data.get('cfg_scale', 10.0)
                style_data = request_data.get('style', 'anime_kawai')

                filename = generate_filename("generated", seed, total_steps)
                os.makedirs("outputs", exist_ok=True)
                image.save(f"outputs/{filename}")
                
                logger.info(f"💾 Saved as: {filename}")
                
                from src.utils.file_manager import FileManager
                FileManager.log_prompt(
                    prompt=prompt_data,
                    settings={
                        "negative_prompt": negative_prompt_data,
                        "seed": seed,
                        "width": width_data,
                        "height": height_data,
                        "steps": steps_data,
                        "cfg_scale": cfg_scale_data,
                        "style": style_data,
                        "model": model_name,
                        "filename": filename
                    }
                )

                success = await safe_send({
                    "type": "completed",
                    "image": f"data:image/png;base64,{img_str}",
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "filename": filename,
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768)
                })
                
                await broadcast_to_gallery({
                    "status": "complete",
                    "progress": 100,
                    "filename": filename,
                    "image": f"data:image/png;base64,{img_str}"
                })
                
                if not success:
                    logger.warning("⚠️  Client disconnected, but image was saved successfully")
                    break
                
                # Update global stats (this might need to be more robust for concurrent access)
                generation_stats["total_images"] += 1
                generation_stats["total_time"] += generation_time
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else 0
                    error_msg = (
                        "💥 CUDA Out of Memory!\n\n"
                        f"Your GPU has {vram_gb:.1f}GB VRAM but this generation requires more.\n\n"
                        "💡 Solutions (try in order):\n"
                        "1. ✅ Enable DRAM Extension in Advanced Settings\n"
                        "   → This adds 8GB system RAM to supplement your VRAM\n"
                        "2. 📐 Use smaller resolution:\n"
                        "   → 512x512 (safest for 4GB VRAM)\n"
                        "   → 512x768 (safe for 6GB VRAM)\n"
                        "3. ⚡ Reduce steps:\n"
                        "   → Use 'Fast' preset (20 steps)\n"
                        "   → Or manually set to 20-25 steps\n"
                        "4. 🔄 Restart the server:\n"
                        "   → This clears cached memory\n"
                        "5. 🎯 Alternative workflow:\n"
                        "   → Generate at 512x512, then Upscale 2x\n"
                        "   → Much faster and uses less VRAM!\n\n"
                        "Recommended settings for your GPU:\n"
                        f"• Resolution: {'512x512' if vram_gb <= 4 else '512x768' if vram_gb <= 6 else '768x768'}\n"
                        "• Steps: 20-28\n"
                        "• Quality: Fast or Balanced\n"
                        "• DRAM Extension: Enabled"
                    )
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "cuda_oom",
                        "message": error_msg
                    })
                    
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            gc.collect()
                        except:
                            pass
                    
                    break
                else:
                    raise e

            # Log prompt history after successful generation or to catch errors
            log_prompt_history(filename, seed, prompt, total_steps)
            
    except WebSocketDisconnect:
        logger.websocket_disconnect()
    except asyncio.CancelledError:
        logger.websocket_disconnect()
    except Exception as e:
        error_name = type(e).__name__
        if "Disconnect" not in error_name and "Closed" not in error_name:
            logger.error(f"WebSocket error: {error_name}: {e}")

            error_type = "unknown"
            error_message = str(e)
            
            if "out of memory" in error_message.lower() or "cuda" in error_message.lower():
                error_type = "cuda_oom"
                logger.error("💥 CUDA Out of Memory Error!")
                logger.error("Solutions:")
                logger.error("  • Enable DRAM Extension in Advanced Settings")
                logger.error("  • Use 512x512 resolution")
                logger.error("  • Reduce steps to 20-25")
                logger.error("  • Select 'Fast' quality preset")
                logger.error("  • Restart server to clear memory")
            elif "connection" in error_message.lower() or "refused" in error_message.lower() or "reset by peer" in error_message.lower():
                error_type = "connection"
                logger.error("🔌 Connection Error!")
                logger.error("Solutions:")
                logger.error("  • Check if backend server is running")
                logger.error("  • Verify port 8000 is not blocked")
                logger.error("  • Check firewall settings")
            else:
                error_type = "generic"
                logger.error(f"❌ Error: {error_message}")
            
            async def send_error_to_client():
                try:
                    if websocket_active:
                        await websocket.send_json({
                            "type": "error",
                            "error_type": error_type,
                            "message": error_message
                        })
                except RuntimeError as e:
                    logger.debug(f"Could not send error to client (WebSocket already closed): {e}")
            
            try:
                asyncio.create_task(send_error_to_client())
            except RuntimeError:
                pass
            
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.websocket_disconnect()

@app.websocket("/ws/gallery-progress")
async def websocket_gallery_progress(websocket: WebSocket):
    """WebSocket endpoint for gallery to receive generation progress updates"""
    await websocket.accept()
    gallery_clients.append(websocket)
    
    logger.info("📸 Gallery client connected for progress updates")
    
    try:
        while True:
            try:
                await websocket.receive_text()
            except:
                break
            await asyncio.sleep(1)
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Gallery WebSocket error: {e}")
    finally:
        if websocket in gallery_clients:
            gallery_clients.remove(websocket)
        logger.info("📸 Gallery client disconnected")

@app.get("/api/output-gallery")
async def get_output_gallery():
    """Get list of all images from outputs directory"""
    try:
        os.makedirs("outputs", exist_ok=True)
        files = [f for f in os.listdir("outputs") if f.endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda x: os.path.getmtime(f"outputs/{x}"), reverse=True)
        return {"images": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error loading output gallery: {e}")
        return {"images": [], "count": 0, "error": str(e)}

@app.get("/api/output-image/{filename}")
async def get_output_image(filename: str):
    """Get a specific image from outputs directory"""
    filepath = f"outputs/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.post("/api/recreate-from-output")
async def recreate_from_output(request: dict):
    """Extract settings from output filename for recreation"""
    try:
        filename = request.get("filename", "")
        
        metadata = extract_metadata_from_filename(filename)
        
        if metadata["seed"] is not None:
            return {
                "success": True,
                "seed": metadata["seed"],
                "message": f"Extracted seed: {metadata['seed']}, type: {metadata['type']}"
            }
        
        parts = filename.replace(".png", "").split("_")
        
        seed = None
        steps = None
        
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) >= 4:
                if seed is None:
                    seed = int(part)
                else:
                    steps = int(part)
        
        if seed is None:
            return {"success": False, "error": "Could not extract seed from filename"}
        
        return {
            "success": True,
            "seed": seed,
            "steps": steps,
            "message": f"Extracted seed: {seed}" + (f", steps: {steps}" if steps else "")
        }
        
    except Exception as e:
        logger.error(f"Recreation failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/gallery")
async def get_gallery():
    """Get list of generated images"""
    os.makedirs("outputs", exist_ok=True)
    files = sorted(
        [f for f in os.listdir("outputs") if f.endswith(".png")],
        key=lambda x: os.path.getmtime(f"outputs/{x}"),
        reverse=True
    )[:12]
    
    return {"images": files}

@app.post("/api/generate")
async def generate_image(request: GenerationRequest):
    """Generate image (non-streaming version for API calls)"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Log prompt before generation starts
        log_prompt(request.prompt, request.model_dump())

        if device == "cuda":
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            request.width, request.height, request.steps = get_safe_generation_params(
                request.width, request.height, request.steps, vram_total
            )
        
        # Prepare seed
        seed = request.seed if request.seed != -1 else np.random.randint(0, 2147483647) # Handle -1 as random seed
        generator = torch.Generator(device).manual_seed(seed)
        
        # Adjust prompt based on style
        if request.style == "pixel_art": # UPDATED KEY
            full_prompt = f"pixel art, 16-bit style, {request.prompt}"
            neg_prompt = f"smooth, anti-aliased, {request.negative_prompt}"
        else:
            full_prompt = f"masterpiece, best quality, {request.prompt}"
            neg_prompt = request.negative_prompt or "lowres, bad anatomy, bad hands, worst quality"
        
        start_time = time.time()
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Generate
        result = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            width=request.width,
            height=request.height,
            generator=generator
        )
        
        generation_time = time.time() - start_time
        
        # Convert image to base64
        image = result.images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        now = datetime.now()
        filename = generate_filename("generated", seed, request.steps)
        os.makedirs("outputs", exist_ok=True)
        image.save(f"outputs/{filename}")
        
        # Call new prompt logging function
        log_prompt_history(filename, seed, request.prompt, request.steps)
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "seed": seed,
            "generation_time": round(generation_time, 2),
            "filename": filename
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            error_msg = (
                "💥 CUDA Out of Memory!\n\n"
                "Your GPU ran out of memory during generation. "
                "Try reducing resolution, steps, or enabling DRAM Extension in advanced settings."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        else:
            raise e
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for real-time generation updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    logger.websocket_connect()
    
    try:
        while True:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            data = await websocket.receive_text()
            logger.info(f"📨 Received data: {data[:100]}...")
            
            request_data = json.loads(data)
            
            # Log prompt before generation starts via WebSocket
            log_prompt(request_data.get('prompt', ''), request_data)

            if device == "cuda":
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                original_width = request_data.get("width", 512)
                original_height = request_data.get("height", 768)
                original_steps = request_data.get("steps", 35)
                
                adjusted_width, adjusted_height, adjusted_steps = get_safe_generation_params(
                    original_width, original_height, original_steps, vram_total
                )
                
                request_data["width"] = adjusted_width
                request_data["height"] = adjusted_height
                request_data["steps"] = adjusted_steps
                
                if (adjusted_width != original_width or adjusted_height != original_height or adjusted_steps != original_steps):
                    await websocket.send_json({
                        "type": "warning",
                        "message": f"Parameters auto-adjusted for low VRAM: {adjusted_width}x{adjusted_height}, {adjusted_steps} steps"
                    })
            
            logger.info("🎨 Starting generation...")
            logger.info(f"   Prompt: {request_data.get('prompt', 'N/A')[:60]}...")
            logger.info(f"   Style: {request_data.get('style', 'anime_kawai')}") # Updated default
            logger.info(f"   Steps: {request_data.get('steps', 35)} (requested)")
            
            if device == "cuda":
                available_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                logger.info(f"   Available VRAM: {available_vram:.2f} GB")
            
            websocket_active = True
            
            async def safe_send(data):
                """Safely send data through WebSocket with error handling"""
                nonlocal websocket_active
                if not websocket_active:
                    return False
                try:
                    await websocket.send_json(data)
                    return True
                except Exception as e:
                    logger.error(f"WebSocket send failed (client likely disconnected): {type(e).__name__}")
                    websocket_active = False
                    return False
            
            async def broadcast_to_gallery(data):
                """Broadcast generation progress to gallery clients"""
                disconnected = []
                for client in gallery_clients:
                    try:
                        await client.send_json(data)
                    except Exception as e:
                        logger.debug(f"Gallery client send failed: {e}")
                        disconnected.append(client)
                
                for client in disconnected:
                    if client in gallery_clients:
                        gallery_clients.remove(client)
            
            if not await safe_send({"type": "started", "message": "Generation started"}):
                break
            
            await broadcast_to_gallery({
                "status": "generating",
                "progress": 0,
                "step": 0,
                "total_steps": request_data.get("steps", 35),
                "total_images": 1,
                "current_image": 1
            })
            
            seed = request_data.get("seed")
            if seed == -1 or seed is None: # Handle -1 seed in websocket endpoint
                seed = np.random.randint(0, 2147483647)
            generator = torch.Generator(device).manual_seed(seed)
            
            style = request_data.get("style", "anime_kawai") # Updated default
            prompt = request_data.get("prompt", "")
            nsfw_filter_enabled = request_data.get("nsfw_filter_enabled", True)
            
            # Check if prompt contains NSFW content BEFORE generation
            nsfw_check = check_nsfw_prompt(prompt, nsfw_filter_enabled)
            if nsfw_check["is_unsafe"]:
                logger.warning(f"NSFW prompt blocked: {nsfw_check['reason']}")
                await safe_send({
                    "type": "error",
                    "message": nsfw_check["message"],
                    "nsfw_blocked": True
                })
                break
            
            if style == "pixel_art": # UPDATED KEY
                full_prompt = f"pixel art, 16-bit style, {prompt}"
                neg_prompt = "smooth, anti-aliased, blurry, 3d render"
            else:
                full_prompt = f"masterpiece, best quality, {prompt}"
                neg_prompt = "lowres, bad anatomy, worst quality, blurry"
            
            user_neg = request_data.get("negative_prompt", "")
            if user_neg:
                neg_prompt = f"{neg_prompt}, {user_neg}"
            
            start_time = time.time()
            total_steps = request_data.get("steps", 35)
            
            step_times = []
            
            def progress_callback(pipe_obj, step: int, timestep: int, callback_kwargs: dict):
                nonlocal websocket_active
                if not websocket_active:
                    return callback_kwargs
                
                current_time = time.time()
                step_times.append(current_time)
                
                progress = (step + 1) / total_steps * 100
                
                avg_step_time = 0
                eta = 0
                if len(step_times) > 1:
                    avg_step_time = (current_time - start_time) / len(step_times)
                    remaining_steps = total_steps - (step + 1)
                    eta = avg_step_time * remaining_steps
                
                gpu_temp = 0
                vram_used = 0
                if device == "cuda":
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=1
                        )
                        temp_str, vram_str = result.stdout.strip().split(',')
                        gpu_temp = float(temp_str)
                        vram_used = float(vram_str) / 1024
                    except:
                        try:
                            vram_used = torch.cuda.memory_allocated(0) / 1024**3
                        except:
                            vram_used = 0
                
                bar_length = 50
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                elapsed_time = current_time - start_time
                print(f"\r🎨 [{bar}] {progress:.0f}% │ {step + 1}/{total_steps} │ ⏱️ {elapsed_time:.0f}s │ ETA: {eta:.0f}s │ 🌡️ {gpu_temp:.0f}°C │ 💾 {vram_used:.1f}GB", end='', flush=True)
                
                async def send_progress():
                    progress_data = {
                        "type": "progress",
                        "step": step + 1,
                        "total_steps": total_steps,
                        "progress": round(progress, 1),
                        "eta": round(eta, 1),
                        "gpu_temp": round(gpu_temp, 1),
                        "vram_used": round(vram_used, 2),
                        "avg_step_time": round(avg_step_time, 2)
                    }
                    await safe_send(progress_data)
                    
                    await broadcast_to_gallery({
                        "status": "generating",
                        "progress": round(progress, 1),
                        "step": step + 1,
                        "total_steps": total_steps,
                        "total_images": 1,
                        "current_image": 1
                    })
                
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(send_progress())
                except RuntimeError:
                    pass
                
                return callback_kwargs

            try:
                # Check if model is loaded
                if pipe is None:
                    error_msg = "Model not loaded yet. Please wait for the server to finish loading models on startup."
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "model_not_loaded",
                        "message": error_msg
                    })
                    break
                
                # Prepare kwargs for pipe call
                pipe_kwargs = {
                    "prompt": full_prompt,
                    "negative_prompt": neg_prompt,
                    "num_inference_steps": total_steps,
                    "guidance_scale": request_data.get("cfg_scale", 10.0),
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768),
                    "generator": generator
                }
                
                # Only add callback if pipe supports it
                try:
                    if pipe is not None and (hasattr(pipe, 'callback_on_step_end') or 'callback_on_step_end' in pipe.__call__.__code__.co_varnames):
                        pipe_kwargs["callback_on_step_end"] = progress_callback
                except (AttributeError, TypeError):
                    # Callback not supported, continue without it
                    pass
                
                result = pipe(**pipe_kwargs)
                
                generation_time = time.time() - start_time
                print()
                logger.success(f"Generation completed in {generation_time:.1f}s")
                
                image = result.images[0]
                
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                model_name = "Ojimi/anime-kawai-diffusion" # Hardcoded for now, could be dynamic
                prompt_data = request_data.get('prompt', '')
                negative_prompt_data = request_data.get('negative_prompt', '')
                width_data = request_data.get('width', 512)
                height_data = request_data.get('height', 768)
                steps_data = request_data.get('steps', 35)
                cfg_scale_data = request_data.get('cfg_scale', 10.0)
                style_data = request_data.get('style', 'anime_kawai')

                filename = generate_filename("generated", seed, total_steps)
                os.makedirs("outputs", exist_ok=True)
                image.save(f"outputs/{filename}")
                
                logger.info(f"💾 Saved as: {filename}")
                
                from src.utils.file_manager import FileManager
                FileManager.log_prompt(
                    prompt=prompt_data,
                    settings={
                        "negative_prompt": negative_prompt_data,
                        "seed": seed,
                        "width": width_data,
                        "height": height_data,
                        "steps": steps_data,
                        "cfg_scale": cfg_scale_data,
                        "style": style_data,
                        "model": model_name,
                        "filename": filename
                    }
                )

                success = await safe_send({
                    "type": "completed",
                    "image": f"data:image/png;base64,{img_str}",
                    "seed": seed,
                    "generation_time": round(generation_time, 2),
                    "filename": filename,
                    "width": request_data.get("width", 512),
                    "height": request_data.get("height", 768)
                })
                
                await broadcast_to_gallery({
                    "status": "complete",
                    "progress": 100,
                    "filename": filename,
                    "image": f"data:image/png;base64,{img_str}"
                })
                
                if not success:
                    logger.warning("⚠️  Client disconnected, but image was saved successfully")
                    break
                
                # Update global stats (this might need to be more robust for concurrent access)
                generation_stats["total_images"] += 1
                generation_stats["total_time"] += generation_time
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == "cuda" else 0
                    error_msg = (
                        "💥 CUDA Out of Memory!\n\n"
                        f"Your GPU has {vram_gb:.1f}GB VRAM but this generation requires more.\n\n"
                        "💡 Solutions (try in order):\n"
                        "1. ✅ Enable DRAM Extension in Advanced Settings\n"
                        "   → This adds 8GB system RAM to supplement your VRAM\n"
                        "2. 📐 Use smaller resolution:\n"
                        "   → 512x512 (safest for 4GB VRAM)\n"
                        "   → 512x768 (safe for 6GB VRAM)\n"
                        "3. ⚡ Reduce steps:\n"
                        "   → Use 'Fast' preset (20 steps)\n"
                        "   → Or manually set to 20-25 steps\n"
                        "4. 🔄 Restart the server:\n"
                        "   → This clears cached memory\n"
                        "5. 🎯 Alternative workflow:\n"
                        "   → Generate at 512x512, then Upscale 2x\n"
                        "   → Much faster and uses less VRAM!\n\n"
                        "Recommended settings for your GPU:\n"
                        f"• Resolution: {'512x512' if vram_gb <= 4 else '512x768' if vram_gb <= 6 else '768x768'}\n"
                        "• Steps: 20-28\n"
                        "• Quality: Fast or Balanced\n"
                        "• DRAM Extension: Enabled"
                    )
                    logger.error(error_msg)
                    await safe_send({
                        "type": "error",
                        "error_type": "cuda_oom",
                        "message": error_msg
                    })
                    
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        try:
                            gc.collect()
                        except:
                            pass
                    
                    break
                else:
                    raise e

            # Log prompt history after successful generation or to catch errors
            log_prompt_history(filename, seed, prompt, total_steps)
            
    except WebSocketDisconnect:
        logger.websocket_disconnect()
    except asyncio.CancelledError:
        logger.websocket_disconnect()
    except Exception as e:
        error_name = type(e).__name__
        if "Disconnect" not in error_name and "Closed" not in error_name:
            logger.error(f"WebSocket error: {error_name}: {e}")

            error_type = "unknown"
            error_message = str(e)
            
            if "out of memory" in error_message.lower() or "cuda" in error_message.lower():
                error_type = "cuda_oom"
                logger.error("💥 CUDA Out of Memory Error!")
                logger.error("Solutions:")
                logger.error("  • Enable DRAM Extension in Advanced Settings")
                logger.error("  • Use 512x512 resolution")
                logger.error("  • Reduce steps to 20-25")
                logger.error("  • Select 'Fast' quality preset")
                logger.error("  • Restart server to clear memory")
            elif "connection" in error_message.lower() or "refused" in error_message.lower() or "reset by peer" in error_message.lower():
                error_type = "connection"
                logger.error("🔌 Connection Error!")
                logger.error("Solutions:")
                logger.error("  • Check if backend server is running")
                logger.error("  • Verify port 8000 is not blocked")
                logger.error("  • Check firewall settings")
            else:
                error_type = "generic"
                logger.error(f"❌ Error: {error_message}")
            
            async def send_error_to_client():
                try:
                    if websocket_active:
                        await websocket.send_json({
                            "type": "error",
                            "error_type": error_type,
                            "message": error_message
                        })
                except RuntimeError as e:
                    logger.debug(f"Could not send error to client (WebSocket already closed): {e}")
            
            try:
                asyncio.create_task(send_error_to_client())
            except RuntimeError:
                pass
            
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.websocket_disconnect()

@app.websocket("/ws/gallery-progress")
async def websocket_gallery_progress(websocket: WebSocket):
    """WebSocket endpoint for gallery to receive generation progress updates"""
    await websocket.accept()
    gallery_clients.append(websocket)
    
    logger.info("📸 Gallery client connected for progress updates")
    
    try:
        while True:
            try:
                await websocket.receive_text()
            except:
                break
            await asyncio.sleep(1)
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Gallery WebSocket error: {e}")
    finally:
        if websocket in gallery_clients:
            gallery_clients.remove(websocket)
        logger.info("📸 Gallery client disconnected")
logger.info("📸 Gallery client disconnected")
