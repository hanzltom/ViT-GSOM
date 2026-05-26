import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np

def unpatch(x: torch.Tensor, patch_size: int = 4, channels: int = 1) -> torch.Tensor:
    """
    Function which transforms the patches from the decoder back to its original input size
    :param x: Patch embeddings of shape ``(batch_size, num_patches, embed_dim)``
    :param patch_size: Size of the patch. Defaults to ``4``
    :param channels: Number of input channels. Defaults to ``1``
    :return: Sequence of picture in original input size
    """  # E.g. (8, 49, 3*4*4): batch of 8, 7x7 grid, num_of_channels * patch_size * patch_size
    B, num_patches, pixels_per_patch = x.shape

    if pixels_per_patch != channels * patch_size * patch_size:
        raise ValueError(
            f'Number of pixels in patch {pixels_per_patch} must be equal to channels * patch_size * patch_size: {channels * patch_size * patch_size}')

    # get size of the grid
    # sqrt(49) = 7 -> 7x7 grid of patches
    grid_h = int(num_patches ** 0.5)
    grid_w = int(num_patches ** 0.5)

    # (B, 49, 48) -> (B, 49, 3, 4, 4): (batch, num_patches, num_of_channels, patch_height, patch_width)
    x = x.reshape(B, num_patches, channels, patch_size, patch_size)

    # (B, 49, 3, 4, 4) -> (B, 7, 7, 3, 4, 4): (Batch, grid_H, grid_W, num_of_channels, patch_H, patch_W)
    x = x.reshape(B, grid_h, grid_w, channels, patch_size, patch_size)

    # (Batch, grid_H, grid_W, num_of_channels, patch_H, patch_W) -> (Batch, num_of_channels, grid_H, patch_H, grid_W, patch_W)
    # (B, 7, 7, 3, 4, 4) -> (B, 3, 7, 4, 7, 4)
    x = x.permute(0, 3, 1, 4, 2, 5)

    # get original size of image
    # (B, 3, 7, 4, 7, 4) -> (B, 3, 7 * 4, 7 * 4)
    x = x.reshape(B, channels, grid_h * patch_size, grid_w * patch_size)

    return x

def cosine_distance_torch(weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Function calculating cosine distance between weights and inputs
    :param weights: SOM weights of shape ``(som_rows * som_cols, num_patches * embed_dim)``
    :param inputs: Input batch tensor of shape ``(batch_size, n_features)``
    :return: Distance of shape ``(batch_size, som_rows * som_cols)``
    """
    # eg. 3x3 grid with weights of dim 4: (3,3,4) -> (9,4)
    if weights.ndim == 3:
        weights_flat = weights.reshape(-1, weights.shape[-1])
    else:
        weights_flat = weights

    # input size: (batch size, dim size), e.g. (32,4)
    inputs_norm = F.normalize(inputs, dim=1)
    # (9,4)
    weights_norm = F.normalize(weights_flat, dim=1)

    # e.g. (32,4) dot (4,9) = (32,9)
    similarity = torch.mm(inputs_norm, weights_norm.t())

    return 1 - similarity

def gaussian_neighbourhood_torch(grid_dists: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """
    Function calculating gaussian neighbourhood influence
    :param grid_dists: Squared Euclidean distances between the BMU and other neurons of shape ``(batch_size, n_nodes)``
    :param sigma_t: Current neighbourhood radius
    :return: Neighbourhood influence tensor of shape ``(batch_size, n_nodes)``
    """
    return torch.exp(-grid_dists / (2 * sigma_t ** 2))

def decay_exponential(initial_value: float, beta: float, t: int) -> float:
    """
    Decay exponential function
    :param initial_value: Initial value
    :param beta: Beta value, must satisfy: 0 < beta < 1
    :param t: Current time
    :return: Decayed initial value
    """
    return initial_value * (beta ** t)

def get_grid_coords(row_num: int, col_num: int, device: torch.device | str) -> torch.Tensor:
    """
    Function calculating grid of 2D coordinates for the SOM
    :param row_num: Number of rows in the SOM grid
    :param col_num: Number of columns in the SOM grid
    :param device: The torch device
    :return: Grid coordinates of shape ``(row_num * col_num, 2)``.
    """
    y_coords, x_coords = torch.meshgrid( torch.arange(row_num, dtype=torch.float32),
        torch.arange(col_num, dtype=torch.float32),
        indexing='ij'
    )

    # coords are 2 dim tensors, we stack them over new dimension to shape (row_num, col_num, 2)
    # reshape them to shape (num_units, 2)
    coords = torch.stack((x_coords, y_coords), dim=-1).reshape(-1, 2)
    return coords.to(device)

def calculate_QE_TE_Purity(model: 'AutoEncoder',
                           loader: torch.utils.data.DataLoader,
                           device: torch.device) -> dict[str: float]:
    """
    Function calculating QE, TE and Purity metrics for model evaluation
    :param model: Trained ViT-SOM Autoencoder
    :param loader: Dataloader
    :param device: The torch device
    :return: A tuple containing (QE, TE, Purity)
    """
    model.eval()
    true_label = []
    cluster_labels = []
    total_qe = 0.0
    total_te = 0.0
    total_samples = 0

    rows, cols = model.get_som_shape()
    grid_coords = get_grid_coords(rows, cols, device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            _, latent = model(images)

            # extract cls token with sequence of patches - not needed
            # shape (batch, embed_dim)
            patches = latent[:, 1:, :]

            # flatten to create som input
            som_input = patches.reshape(patches.shape[0], -1)

            # calculate distance, shape (batch, neuron unit num)
            dists = cosine_distance_torch(model.get_som_weights(), som_input)
            min_dists, bmu_indices = torch.min(dists, dim=1)

            # QE
            total_qe += torch.sum(min_dists).item()

            # TE
            _, top2_indices = torch.topk(dists, k=2, dim=1, largest=False)
            bmu1_idx = top2_indices[:, 0]
            bmu2_idx = top2_indices[:, 1]

            bmu1_coords = grid_coords[bmu1_idx]
            bmu2_coords = grid_coords[bmu2_idx]

            grid_dists = torch.norm(bmu1_coords - bmu2_coords, p=2, dim=1)
            total_te += torch.sum(grid_dists > 1.42).item()

            # Purity
            true_label.append(labels.cpu())
            cluster_labels.append(bmu_indices.cpu())

            total_samples += dists.shape[0]

    output = {}
    output["QE"] = total_qe / total_samples if total_samples > 0 else 0
    output["TE"] = total_te / total_samples if total_samples > 0 else 0

    true_labels_np = torch.cat(true_label).cpu().numpy()
    cluster_labels_np = torch.cat(cluster_labels).cpu().numpy()
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels_np, cluster_labels_np)
    output["Purity"] = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    return output