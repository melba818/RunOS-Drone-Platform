from fastapi import APIRouter

from config import settings

router = APIRouter()


@router.get("/api/health")
async def health_check():
    import os
    return {
        "status": "healthy",
        "version": "3.0.0",
        "tenant_admin_id": int(os.environ.get("RUNOS_OWNER_ADMIN_ID", "1")),
    }


@router.get("/api/device-info")
async def device_info():
    # Share the read-only NVML hardware probe used by the settings page. Avoid
    # torch.cuda.is_available()/get_device_name(), which can initialize a CUDA
    # context even though this endpoint only needs to display device metadata.
    hardware = await hardware_info()
    info: dict = {
        "device": "cuda" if hardware["cuda_available"] else "cpu",
        "cuda_available": hardware["cuda_available"],
        "cuda_version": hardware["cuda_version"],
        "torch_version": hardware.get("torch_version", "not installed"),
        "gpus": [gpu["name"] for gpu in hardware["gpus"]],
        "gpu_status": settings._gpu_status,
    }
    return info


@router.get("/api/system")
async def system_stats():
    import asyncio, importlib

    stats: dict = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "gpu_percent": None,
        "gpu_memory_percent": None,
        "gpu_name": None,
    }

    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is not None:
        import psutil
        loop = asyncio.get_event_loop()
        cpu_pct = await loop.run_in_executor(None, lambda: psutil.cpu_percent(interval=0.5))
        stats["cpu_percent"] = cpu_pct
        vm = psutil.virtual_memory()
        stats["ram_percent"] = vm.percent

    pynvml_spec = importlib.util.find_spec("pynvml")
    if pynvml_spec is not None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            stats["gpu_percent"] = float(util.gpu)
            stats["gpu_memory_percent"] = round(mem_info.used / mem_info.total * 100, 1)
            stats["gpu_name"] = name if isinstance(name, str) else name.decode()
        except Exception:
            pass

    return stats


@router.get("/api/hardware")
async def hardware_info():
    import importlib, platform

    info: dict = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
        "cpu_cores": None,
        "cpu_model": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
    }

    psutil_spec = importlib.util.find_spec("psutil")
    if psutil_spec is not None:
        import psutil
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024 ** 3, 1)
        info["ram_available_gb"] = round(vm.available / 1024 ** 3, 1)

    try:
        if platform.system() == "Windows":
            import subprocess
            r = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=3,
            )
            for line in r.stdout.splitlines():
                if line.startswith("Name="):
                    info["cpu_model"] = line.split("=", 1)[1].strip()
                    break
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            info["cpu_model"] = r.stdout.strip()
    except Exception:
        pass

    # Read device inventory through NVML first. These calls inspect driver state
    # only and do not create a CUDA context or reserve device memory.
    if importlib.util.find_spec("pynvml") is not None:
        nvml_initialized = False
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml_initialized = True
            count = pynvml.nvmlDeviceGetCount()
            info["gpu_count"] = count
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["gpus"].append({
                    "index": i,
                    "name": name if isinstance(name, str) else name.decode(),
                    "memory_total_mb": mem.total // (1024 ** 2),
                    "memory_free_mb": mem.free // (1024 ** 2),
                })
        except Exception:
            pass
        finally:
            if nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    if not info["gpus"]:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        info["gpus"].append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                        })
                if info["gpus"]:
                    info["gpu_count"] = len(info["gpus"])
        except Exception:
            pass

    # Importing torch and reading its build metadata does not initialize CUDA.
    # With the NVIDIA runtime exposing the GPUs, this reports whether the
    # installed PyTorch build has CUDA support while keeping this page read-only.
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        try:
            import torch
            info["torch_version"] = torch.__version__
            cuda_version = getattr(torch.version, "cuda", None)
            if info["gpus"] and cuda_version:
                info["cuda_available"] = True
                info["cuda_version"] = cuda_version
        except Exception:
            pass

    return info
