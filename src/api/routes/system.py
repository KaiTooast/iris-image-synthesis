"""
System Routes - Health, GPU info, version endpoints
"""
import subprocess
import time
import torch
from fastapi import APIRouter

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.api.services.pipeline import pipeline_service, MODEL_CONFIGS
from src.utils.logger import create_logger

logger = create_logger("SystemRoutes")
router = APIRouter(prefix="/api", tags=["system"])

# Track server start time for uptime
_server_start_time = time.time()
_total_generations = 0
_total_generation_time = 0.0


def increment_generation_stats(gen_time: float = 0.0):
    """Called after each generation to update stats"""
    global _total_generations, _total_generation_time
    _total_generations += 1
    _total_generation_time += gen_time


@router.get("/health")
async def health_check():
    """Health check endpoint with stats"""
    uptime_seconds = time.time() - _server_start_time
    
    # Format uptime
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    if hours > 0:
        uptime_str = f"{hours}h {minutes}m"
    else:
        uptime_str = f"{minutes}m"
    
    return {
        "status": "healthy",
        "model_loaded": pipeline_service.pipe is not None,
        "device": pipeline_service.device,
        "uptime": uptime_str,
        "stats": {
            "total_generations": _total_generations,
            "total_generation_time": round(_total_generation_time, 2)
        }
    }


@router.get("/system")
async def get_system_info():
    """Get system information"""
    info = {
        "gpu_name": "Unknown",
        "device": pipeline_service.device,
        "vram_total": 0.0,
        "vram_used": 0.0,
        "gpu_temp": 0.0,
        "dram_extension_enabled": pipeline_service.dram_config["enabled"],
        "dram_extension_available": False
    }
    
    if pipeline_service.device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["vram_total"] = vram_total
            info["vram_used"] = torch.cuda.memory_allocated(0) / 1024**3
            info["dram_extension_available"] = vram_total <= pipeline_service.dram_config["vram_threshold_gb"]
            
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


@router.get("/gpu-info")
async def get_gpu_info():
    """Get detailed GPU, CPU, and RAM information"""
    info = {
        "gpu_name": "Unknown",
        "vram_total": 0.0,
        "vram_used": 0.0,
        "vram_free": 0.0,
        "vram_percent": 0.0,
        "gpu_temp": 0,
        "power_draw": 0.0,
        "gpu_utilization": 0,
        "status": "unknown",
        # CPU info
        "cpu_percent": 0.0,
        "cpu_freq": 0.0,
        "cpu_cores": 0,
        # RAM info
        "ram_total": 0.0,
        "ram_used": 0.0,
        "ram_free": 0.0,
        "ram_percent": 0.0,
        # System power (if available)
        "system_power": None
    }
    
    # Get CPU and RAM info using psutil
    if PSUTIL_AVAILABLE:
        try:
            # CPU
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            info["cpu_cores"] = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["cpu_freq"] = round(cpu_freq.current / 1000, 2)  # GHz
            
            # RAM
            ram = psutil.virtual_memory()
            info["ram_total"] = round(ram.total / (1024**3), 2)  # GB
            info["ram_used"] = round(ram.used / (1024**3), 2)  # GB
            info["ram_free"] = round(ram.available / (1024**3), 2)  # GB
            info["ram_percent"] = ram.percent
        except Exception as e:
            logger.warning(f"psutil error: {e}")
    
    if pipeline_service.device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["status"] = "success"
            
            # Get accurate GPU stats from nvidia-smi (real-time values)
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 5:
                        info["gpu_temp"] = int(float(parts[0].strip()))
                        try:
                            info["power_draw"] = float(parts[1].strip())
                        except:
                            info["power_draw"] = 0.0
                        info["gpu_utilization"] = int(float(parts[2].strip()))
                        # VRAM in MB from nvidia-smi, convert to GB
                        vram_used_mb = float(parts[3].strip())
                        vram_total_mb = float(parts[4].strip())
                        info["vram_used"] = round(vram_used_mb / 1024, 2)
                        info["vram_total"] = round(vram_total_mb / 1024, 2)
                        info["vram_free"] = round((vram_total_mb - vram_used_mb) / 1024, 2)
                        info["vram_percent"] = round((vram_used_mb / vram_total_mb) * 100, 1) if vram_total_mb > 0 else 0
            except FileNotFoundError:
                # nvidia-smi not found, fallback to PyTorch values
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                vram_reserved = torch.cuda.memory_reserved(0) / 1024**3
                info["vram_total"] = round(vram_total, 2)
                info["vram_used"] = round(vram_reserved, 2)
                info["vram_free"] = round(vram_total - vram_reserved, 2)
                info["vram_percent"] = round((vram_reserved / vram_total) * 100, 1) if vram_total > 0 else 0
            except Exception as e:
                logger.warning(f"nvidia-smi error: {e}")
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            info["status"] = "error"
    else:
        info["status"] = "no_gpu"
    
    return {"status": info["status"], "gpu": info}


@router.get("/version")
async def get_version_info():
    """Get version information"""
    import sys
    import platform
    
    # Detect OS
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "macOS"
    
    return {
        "iris_version": "1.2.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "CPU",
        "current_model": pipeline_service.current_model,
        "device": pipeline_service.device,
        "os": os_name
    }


@router.get("/models")
async def get_available_models():
    """Get list of available models"""
    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append({
            "id": key,
            "name": config["id"].split("/")[-1],
            "description": config["description"],
            "huggingface_id": config["id"],
            "is_loaded": pipeline_service.current_model == key
        })
    return {"models": models}


@router.get("/rpc-status")
async def get_rpc_status():
    """Get RPC status (placeholder)"""
    return {
        "connected": True,
        "status": "ready",
        "details": "Ready for requests"
    }


@router.get("/devices")
async def get_available_devices():
    """Get list of available compute devices"""
    devices = pipeline_service.get_available_devices()
    return {
        "devices": devices,
        "current_device": pipeline_service.device,
        "forced_device": pipeline_service.forced_device
    }


@router.post("/device")
async def switch_device(device: str):
    """Switch between GPU and CPU mode"""
    try:
        result = await pipeline_service.switch_device(device)
        return result
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to switch device: {str(e)}")
