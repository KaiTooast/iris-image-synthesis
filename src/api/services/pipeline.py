"""
Pipeline Service - Model Loading & Generation Logic
"""
import torch
import os
import gc
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from src.utils.logger import create_logger

logger = create_logger("PipelineService")

# Model configurations
# =====================
# VRAM Requirements (approximate, at 512x768):
#   - SD 1.5 based models: ~3-4 GB VRAM (with optimizations: ~2.5 GB)
#   - SDXL based models: ~8-10 GB VRAM (with optimizations: ~6 GB)
#
# Performance Notes:
#   - Speed depends on GPU, VRAM, and model architecture
#   - SD 1.5 models are ~2-3x faster than SDXL models
#   - DRAM Extension allows running on low VRAM but is slower
#
# Sorted by Performance Tier (S → C)
#
MODEL_CONFIGS = {
    # ============ S-TIER (Best Quality & Versatility) ============
    "animagine_xl": {
        "id": "cagliostrolab/animagine-xl-3.1",
        "description": "High-Quality Modern Anime (SDXL)",
        # VRAM: ~8-10 GB | Quality: ★★★★★ | Speed: ★★☆☆☆
        # Best for: High-resolution anime art, detailed illustrations
        # Note: Requires significant VRAM, not recommended for <8GB cards
    },
    "dreamshaper_8": {
        "id": "Lykon/dreamshaper-8",
        "description": "Best Allround Model (Fast)",
        # VRAM: ~3-4 GB | Quality: ★★★★★ | Speed: ★★★★☆
        # Best for: Everything - anime, realistic, fantasy
        # Highly recommended for versatility
    },
    "stable_diffusion_3_5": {
        "id": "stabilityai/stable-diffusion-3.5-medium",
        "description": "High-Quality Realistic Style",
        # VRAM: ~10-12 GB | Quality: ★★★★★ | Speed: ★★☆☆☆
        # Best for: Highest quality realistic images
        # Note: Requires high VRAM, newest architecture
    },
    
    # ============ A-TIER (High Quality) ============
    "anime_kawai": {
        "id": "Ojimi/anime-kawai-diffusion",
        "description": "Anime & Kawai Style",
        # VRAM: ~3-4 GB | Quality: ★★★★☆ | Speed: ★★★★☆
        # Best for: Cute anime style, kawaii characters
        # Recommended default for most users
    },
    "aom3": {
        "id": "WarriorMama777/AbyssOrangeMix3",
        "description": "Semi-Realistic Anime Style",
        # VRAM: ~3-4 GB | Quality: ★★★★☆ | Speed: ★★★★☆
        # Best for: Semi-realistic anime, detailed faces
        # Good balance of quality and speed
    },
    "counterfeit_v3": {
        "id": "stablediffusionapi/counterfeit-v30",
        "description": "Detailed Digital Illustration Style",
        # VRAM: ~3-4 GB | Quality: ★★★★☆ | Speed: ★★★★☆
        # Best for: Digital art, detailed illustrations
        # Excellent for character art
    },
    "openjourney": {
        "id": "prompthero/openjourney",
        "description": "Artistic Illustration Style",
        # VRAM: ~3-4 GB | Quality: ★★★★☆ | Speed: ★★★★☆
        # Best for: Midjourney-like artistic style
        # Great for creative/artistic outputs
    },
    "pixel_art": {
        "id": "nitrosocke/pixel-art-diffusion",
        "description": "Pixel Art & Retro Style",
        # VRAM: ~3-4 GB | Quality: ★★★★☆ | Speed: ★★★★★
        # Best for: Pixel art, retro game assets, 16-bit style
        # Very fast, specialized for pixel art
    },
    
    # ============ B-TIER (Good Quality, Fast) ============
    "anything_v5": {
        "id": "stablediffusionapi/anything-v5",
        "description": "Classic Flat Anime Style (Fast)",
        # VRAM: ~3-4 GB | Quality: ★★★☆☆ | Speed: ★★★★★
        # Best for: Quick generations, flat anime style
        # Very fast, good for iteration
    },
    "stable_diffusion_2_1": {
        "id": "stabilityai/stable-diffusion-2-1",
        "description": "Realistic Photo Style",
        # VRAM: ~4-5 GB | Quality: ★★★☆☆ | Speed: ★★★★☆
        # Best for: Realistic photos, landscapes
        # Official Stability AI model
    },
    "waifu_diffusion": {
        "id": "hakurei/waifu-diffusion",
        "description": "Classic Waifu & Anime Style",
        # VRAM: ~3-4 GB | Quality: ★★★☆☆ | Speed: ★★★★★
        # Best for: Classic anime style, waifus
        # One of the original anime models, very stable
    },
}


class PipelineService:
    """Manages AI model pipelines"""
    
    def __init__(self):
        self.pipe = None
        self.img2img_pipe = None
        self.device = None
        self.forced_device = None  # User-forced device (None = auto)
        self.dtype = torch.float32
        self.current_model = None
        self.upscaler = None
        
        # DRAM Extension Configuration
        self.dram_config = {
            "enabled": False,
            "vram_threshold_gb": 6,
            "max_dram_gb": 16
        }
    
    def detect_device(self, force_device: str = None):
        """Detect and configure the best available device"""
        # If user forced a specific device
        if force_device:
            self.forced_device = force_device
            if force_device == "cpu":
                self.device = "cpu"
                self.dtype = torch.float32
                logger.info("Forced CPU mode by user")
                return self.device
            elif force_device == "cuda" and torch.cuda.is_available():
                self.forced_device = "cuda"
                # Continue with CUDA detection below
            elif force_device == "mps" and torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float32
                logger.success("Forced Apple Silicon mode")
                return self.device
            elif force_device == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
                self.device = "xpu"
                self.dtype = torch.float32
                logger.success("Forced Intel Arc mode")
                return self.device
        
        # CUDA covers both NVIDIA and AMD ROCm
        if torch.cuda.is_available() and self.forced_device != "cpu":
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            
            # Detect GPU vendor
            is_amd = "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper()
            vendor = "AMD" if is_amd else "NVIDIA"
            
            logger.success(f"{vendor} GPU detected: {gpu_name}")
            
            vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"VRAM: {vram_total_gb:.1f}GB")
            
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            if vram_total_gb <= self.dram_config["vram_threshold_gb"]:
                logger.info(f"Auto-enabling DRAM Extension for {vram_total_gb:.1f}GB VRAM card")
                self.dram_config["enabled"] = True
            
            # Check for tensor cores (NVIDIA only, AMD doesn't have them in the same way)
            if not is_amd:
                has_tensor_cores = any(arch in gpu_name.upper() for arch in ["RTX", "A100", "V100", "T4", "A10", "A40"])
                if has_tensor_cores:
                    logger.success("Tensor Cores detected! Using float16")
                    self.dtype = torch.float16
                else:
                    logger.warning("No Tensor Cores detected. Using float32")
                    self.dtype = torch.float32
            else:
                # AMD GPUs - use float16 for RDNA2+ (RX 6000+, RX 7000+)
                if any(x in gpu_name.upper() for x in ["RX 6", "RX 7", "RADEON PRO"]):
                    logger.success("AMD RDNA2+ detected! Using float16")
                    self.dtype = torch.float16
                else:
                    self.dtype = torch.float32
                
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32  # MPS works best with float32
            logger.success("Apple Silicon detected (Metal Performance Shaders)")
            
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = "xpu"
            self.dtype = torch.float32
            try:
                xpu_name = torch.xpu.get_device_name(0)
                logger.success(f"Intel Arc GPU detected: {xpu_name}")
            except:
                logger.success("Intel Arc GPU detected")
            
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            logger.info("Running in CPU mode")
        
        return self.device
    
    async def switch_device(self, target_device: str):
        """Switch between GPU and CPU mode"""
        valid_devices = ["cuda", "cpu", "mps", "xpu", "auto"]
        if target_device not in valid_devices:
            raise ValueError(f"Invalid device: {target_device}. Valid: {valid_devices}")
        
        # Check if target device is available
        if target_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA/ROCm is not available on this system")
        if target_device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS (Apple Silicon) is not available on this system")
        if target_device == "xpu":
            if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
                raise ValueError("Intel XPU is not available. Install intel-extension-for-pytorch")
        
        old_device = self.device
        old_model = self.current_model
        
        logger.info(f"Switching device from {old_device} to {target_device}...")
        
        # Cleanup current pipeline
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.img2img_pipe is not None:
            del self.img2img_pipe
            self.img2img_pipe = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Re-detect device with forced setting
        if target_device == "auto":
            self.forced_device = None
            self.detect_device()
        else:
            self.detect_device(force_device=target_device)
        
        # Reload model if one was loaded
        if old_model:
            await self.load_model(old_model)
        
        logger.success(f"Device switched to {self.device}")
        return {
            "success": True,
            "old_device": old_device,
            "new_device": self.device,
            "model_reloaded": old_model is not None
        }
    
    def get_available_devices(self):
        """Get list of available devices including NVIDIA, AMD, Intel, and Apple"""
        devices = [{"id": "cpu", "name": "CPU", "available": True, "type": "cpu"}]
        
        # NVIDIA CUDA or AMD ROCm (both use cuda backend)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Detect if it's AMD (ROCm presents as CUDA)
            is_amd = "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper()
            
            devices.append({
                "id": "cuda",
                "name": f"{gpu_name} ({vram:.1f}GB)",
                "available": True,
                "type": "amd" if is_amd else "nvidia"
            })
        
        # Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            devices.append({
                "id": "mps",
                "name": "Apple Silicon (Metal)",
                "available": True,
                "type": "apple"
            })
        
        # Intel Arc (XPU) - requires intel-extension-for-pytorch
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                xpu_name = "Intel Arc GPU"
                try:
                    xpu_name = torch.xpu.get_device_name(0)
                except:
                    pass
                devices.append({
                    "id": "xpu",
                    "name": xpu_name,
                    "available": True,
                    "type": "intel"
                })
        except Exception:
            pass
        
        return devices
    
    async def load_model(self, model_name: str = "anime_kawai"):
        """Load a specific model"""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_id = MODEL_CONFIGS[model_name]["id"]
        logger.info(f"Loading model: {model_id}")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.pipe = self.pipe.to(self.device)
            
            # Apply optimizations
            if self.device == "cuda":
                self._apply_cuda_optimizations(self.pipe)
            
            # Create img2img pipeline
            self.img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
            
            if not self.dram_config["enabled"]:
                self.img2img_pipe = self.img2img_pipe.to(self.device)
            
            if self.device == "cuda":
                self._apply_cuda_optimizations(self.img2img_pipe)
            
            self.current_model = model_name
            logger.success(f"Model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_cuda_optimizations(self, pipeline):
        """Apply CUDA-specific optimizations"""
        pipeline.enable_attention_slicing(slice_size=1)
        
        # Use new VAE methods to avoid deprecation warnings
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            if hasattr(pipeline.vae, 'enable_slicing'):
                pipeline.vae.enable_slicing()
            if hasattr(pipeline.vae, 'enable_tiling'):
                pipeline.vae.enable_tiling()
        
        if self.dram_config["enabled"]:
            self._apply_dram_extension(pipeline)
        
        if self.dtype == torch.float16 and hasattr(pipeline, 'vae'):
            pipeline.vae.to(torch.float32)
        
        torch.cuda.empty_cache()
    
    def _apply_dram_extension(self, pipeline):
        """Enable DRAM as VRAM extension"""
        if self.device != "cuda":
            return
        
        try:
            if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                pipeline.enable_sequential_cpu_offload()
                logger.success("Sequential CPU offload enabled")
            elif hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                logger.success("Model CPU offload enabled")
            
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        except Exception as e:
            logger.error(f"Failed to enable DRAM extension: {e}")
    
    def get_safe_params(self, width: int, height: int, steps: int) -> tuple:
        """Auto-adjust parameters based on VRAM"""
        if self.dram_config["enabled"]:
            return width, height, steps
        
        if self.device != "cuda":
            return width, height, steps
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        total_pixels = width * height
        
        if vram_gb <= 4:
            if total_pixels > 512 * 512:
                width, height = 512, 512
                logger.warning(f"Auto-adjusted to {width}x{height} for 4GB VRAM")
            if steps > 25:
                steps = 25
                
        elif vram_gb <= 6:
            if total_pixels > 720 * 1280:
                width, height = 720, 1280
                logger.warning(f"Auto-adjusted to {width}x{height} for 6GB VRAM")
            if steps > 35:
                steps = 35
        
        return width, height, steps
    
    def check_vram_availability(self, width: int, height: int, steps: int) -> dict:
        """
        Check if there's enough VRAM for the requested generation.
        Returns dict with can_generate, estimated_vram_gb, available_vram_gb, and adjusted_params if needed.
        """
        result = {
            "can_generate": True,
            "estimated_vram_gb": 0,
            "available_vram_gb": 0,
            "adjusted_params": None,
            "dram_enabled": self.dram_config["enabled"]
        }
        
        if self.device != "cuda":
            return result
        
        try:
            # Get current VRAM status
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
            cached_vram = torch.cuda.memory_reserved(0) / (1024**3)
            available_vram = total_vram - allocated_vram
            
            result["total_vram_gb"] = round(total_vram, 2)
            result["available_vram_gb"] = round(available_vram, 2)
            result["allocated_vram_gb"] = round(allocated_vram, 2)
            
            # Estimate VRAM needed for generation
            # Base model: ~2-4GB, per megapixel: ~0.5-1GB, per 10 steps: ~0.1GB
            total_pixels = width * height
            megapixels = total_pixels / 1_000_000
            
            # Estimation formula (empirical)
            base_vram = 2.5  # Base model loaded
            pixel_vram = megapixels * 0.8  # Per megapixel
            step_vram = (steps / 50) * 0.3  # Steps overhead
            
            estimated_vram = base_vram + pixel_vram + step_vram
            result["estimated_vram_gb"] = round(estimated_vram, 2)
            
            # If DRAM extension is enabled, we can use more
            if self.dram_config["enabled"]:
                max_dram = self.dram_config["max_dram_gb"]
                effective_memory = total_vram + max_dram * 0.3  # DRAM is slower, count 30%
                result["effective_memory_gb"] = round(effective_memory, 2)
                
                if estimated_vram <= effective_memory:
                    result["can_generate"] = True
                    result["using_dram"] = estimated_vram > total_vram
                    return result
            
            # Check if we can generate
            safety_margin = 0.5  # Keep 0.5GB free
            if estimated_vram > (available_vram - safety_margin):
                result["can_generate"] = False
                
                # Try to find adjusted parameters that would work
                adjusted = self._find_safe_params(available_vram - safety_margin, width, height, steps)
                if adjusted:
                    result["adjusted_params"] = adjusted
                    result["can_generate"] = True
                    logger.info(f"VRAM check: adjusted params to {adjusted}")
            
            return result
            
        except Exception as e:
            logger.error(f"VRAM check failed: {e}")
            return result
    
    def _find_safe_params(self, available_vram: float, width: int, height: int, steps: int) -> dict:
        """Find parameters that fit in available VRAM"""
        base_vram = 2.5
        
        # Try reducing resolution first
        resolutions = [
            (width, height),
            (int(width * 0.75), int(height * 0.75)),
            (512, 768),
            (512, 512),
            (384, 512),
        ]
        
        step_options = [steps, min(steps, 35), min(steps, 25), min(steps, 15)]
        
        for w, h in resolutions:
            for s in step_options:
                megapixels = (w * h) / 1_000_000
                estimated = base_vram + megapixels * 0.8 + (s / 50) * 0.3
                
                if estimated <= available_vram:
                    # Round to multiples of 64 for optimal performance
                    w = (w // 64) * 64
                    h = (h // 64) * 64
                    return {"width": max(256, w), "height": max(256, h), "steps": s}
        
        return None
    
    def get_vram_status(self) -> dict:
        """Get current VRAM status for monitoring"""
        if self.device != "cuda":
            return {"device": self.device, "vram_available": False}
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - allocated
            
            return {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "total_vram_gb": round(total, 2),
                "allocated_vram_gb": round(allocated, 2),
                "cached_vram_gb": round(cached, 2),
                "free_vram_gb": round(free, 2),
                "utilization_percent": round((allocated / total) * 100, 1),
                "dram_extension": self.dram_config
            }
        except Exception as e:
            return {"device": "cuda", "error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore CUDA errors during shutdown
        gc.collect()


# Global instance
pipeline_service = PipelineService()
