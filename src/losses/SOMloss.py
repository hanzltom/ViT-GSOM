import torch
import torch.nn as nn

from src.utils.helpers import gaussian_neighbourhood_torch, cosine_distance_torch

class SomLoss(nn.Module):
    """
    Class to calculate SOM loss for weights update
    """

    def __init__(self):
        """
        Constructor for SomLoss class
        """
        super().__init__()

    def forward(self,
                latent_vectors: torch.Tensor,
                som_weights: torch.Tensor,
                grid_coords: torch.Tensor,
                sigma: float) -> torch.Tensor:
        """
        Forward pass to compute SOM loss
        :param latent_vectors: Patch embeddings from the encoder of shape ``(batch_size, seq_len + 1, embed_dim)``
        :param som_weights: SOM weight tensor of shape ``(n_nodes, n_features)``
        :param grid_coords: Grid coordinate tensor of shape ``(n_nodes, 2)``
        :param sigma: The current neighborhood radius
        :return: Scalar loss tensor
        """
        # latent vector shape: (batch, sequence of patches + cls, embed_dim), cls not needed for SOM, only patches
        patches = latent_vectors[:, 1:, :]

        som_input = patches.reshape(patches.shape[0], -1)

        # distance for all samples in batch, shape (batch, Num_Units)
        dists = cosine_distance_torch(som_weights, som_input)

        # indices of bmu for each sample in batch, size (batch,)
        bmu_indices = torch.argmin(dists, dim=1)

        # coordinates of the bmus for this batch, shape (batch, 2)
        bmu_coords = grid_coords[bmu_indices]

        # calculating euclidean distance between bmus and all other neuron units along the coordinate dimension
        # unsqueezing to allow broadcasting
        # (batch, 1, 2) - (1, Num_Units, 2) -> (batch, Num_Units, 2)
        dist_grid = torch.sum((bmu_coords.unsqueeze(1) - grid_coords.unsqueeze(0)) ** 2, dim=2)

        # calculating neighbourhood influence through neighbourhood function - gaussian
        neighbourhood_influence = gaussian_neighbourhood_torch(dist_grid, sigma)

        loss = neighbourhood_influence * dists
        return loss.sum(dim=1).mean()