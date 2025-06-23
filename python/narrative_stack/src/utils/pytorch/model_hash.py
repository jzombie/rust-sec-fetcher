import torch
import hashlib


def model_hash(model: torch.nn.Module) -> str:
    """
    Compute a SHA-256 hash of all parameters in a PyTorch model.

    This is useful for verifying whether two models have identical weights,
    regardless of device placement or internal architecture references.

    Parameters:
        model (torch.nn.Module):
            The PyTorch model to hash.

    Returns:
        str:
            A hexadecimal SHA-256 digest representing the model's parameters.
    """
    h = hashlib.sha256()
    for param in model.parameters():
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()
