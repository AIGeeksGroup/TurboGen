import torch

from ltx_core.model.model_protocol import ModelConfigurator


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    """
    Feature extractor module for Gemma models.
    This module applies a single linear projection to the input tensor.
    It expects a flattened feature tensor of shape (batch_size, 3840*49).
    The linear layer maps this to a (batch_size, 3840) embedding.
    Attributes:
        aggregate_embed (torch.nn.Linear): Linear projection layer.
    """

    def __init__(self) -> None:
        """
        Initialize the GemmaFeaturesExtractorProjLinear module.
        The input dimension is expected to be 3840 * 49, and the output is 3840.
        """
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature extractor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3840 * 49).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3840).
        """
        return self.aggregate_embed(x)

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], _config: dict) -> "GemmaFeaturesExtractorProjLinear":
        # Some LTX checkpoints do not include this image feature projection weight.
        # The ODE text-prompt generation path does not use it, but leaving it on
        # the meta device makes Module.to() fail. Materialize it on CPU so the
        # model can be moved normally when the checkpoint lacks this optional key.
        with torch.device("cpu"):
            return cls()
