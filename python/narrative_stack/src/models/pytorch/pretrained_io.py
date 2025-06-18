import os
import json
import torch


class PretrainedIO:
    """
    Mixin class for saving and loading PyTorch Lightning models
    using HuggingFace-style config + weights structure.
    """

    @classmethod
    def from_pretrained(cls, save_directory, device=None):
        """
        Load a model from a directory containing `config.json` and `pytorch_model.bin`.

        Args:
            save_directory (str): Path to the directory containing model files.
            device (Union[str, torch.device], optional): Device to load weights onto.

        Returns:
            model (nn.Module): Instantiated model with loaded weights.
        """
        config_path = os.path.join(save_directory, "config.json")
        weights_path = os.path.join(save_directory, "pytorch_model.bin")

        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save weights
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        config = self.hparams if hasattr(self, "hparams") else self.__dict__
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
