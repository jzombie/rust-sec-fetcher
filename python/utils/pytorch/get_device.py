import torch


def get_device() -> str:
    """
    Utility function to return the appropriate device based on availability:
    - If MPS is available, returns "mps".
    - If CUDA is available, returns "cuda".
    - Otherwise, returns "cpu".
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device
