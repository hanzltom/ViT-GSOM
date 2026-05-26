import torch
import torch.nn as nn

class ViTLoss(nn.Module):
    """
    Class to calculate ViT reconstruction loss for weights update
    """

    def __init__(self):
        """
        Constructor for ViTLoss class
        """
        super().__init__()

        self.mseLoss = nn.MSELoss()

    def forward(self, original_img: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute ViT loss using MSE
        :param original_img: The real input image of shape ``(batch_size, channels, height, width)``
        :param reconstructed: The reconstructed image by the decoder of shape ``(batch_size, channels, height, width)``
        :return: Scalar loss tensor
        """
        l_nn = self.mseLoss(original_img, reconstructed)

        return l_nn