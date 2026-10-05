import os

# MPS'te desteklenmeyen bir op olursa CPU'ya düşsün (torch import edilmeden önce ayarlanmalı)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def get_device(prefer=None):
    """CUDA > MPS (Apple Silicon) > CPU. `prefer` ile zorlanabilir ('cpu', 'mps', 'cuda')."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
