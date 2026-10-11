import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Safely select the execution device.
    
    If preference is 'auto' or 'cuda':
      Uses CUDA if available (e.g., in Google Colab).
      Otherwise falls back to CPU gracefully with logging.
    Notice: Intel Iris Xe and other integrated GPUs do not support CUDA;
    CPU is the correct, reliable execution device on local Windows.
    """
    pref = preference.lower().strip()
    if pref in ["cuda", "gpu", "auto"]:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {device_name}")
            return dev
        else:
            if pref != "auto":
                logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Return dictionary of current device and PyTorch info."""
    cuda_avail = torch.cuda.is_available()
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU (Host System)",
    }
    return info
