import torch


def get_device() -> torch.device:
    """
    Detect the optimal available PyTorch device and return it as a torch.device
    instance. Prefers MPS > CUDA > CPU.

    Returns:
        torch.device: The best available device ("mps", "cuda", or "cpu").
    """

    if torch.backends.mps.is_available():
        # TODO: Check for related: https://pytorch.org/docs/main/notes/mps.html
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return torch.device(device)
