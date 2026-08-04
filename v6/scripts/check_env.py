import importlib.metadata

import torch


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


assert torch.cuda.is_available(), "CUDA is unavailable"

device = torch.cuda.current_device()
print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"VRAM_GiB: {torch.cuda.get_device_properties(device).total_memory / 2**30:.2f}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {version('transformers')}")
