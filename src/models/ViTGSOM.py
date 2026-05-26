import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.ViT import ViTEncoder, ViTDecoder
from src.utils.helpers import unpatch, cosine_distance_torch

class ViTGSOM(nn.Module):
    """
    ViTGSOM class: Vision Autoencoder with an integrated dynamic Self-Organizing Map layer
    """

    def __init__(self, config: dict[str, int | float]):
        """
        Constructor for ViT-GSOM
        :param config: Configuration dictionary
        """
        super().__init__()

        # ensure image size can be divided by the size of the patch
        if config['img_size'] % config['patch_size'] != 0:
            raise ValueError(f"Image size ({config['img_size']}) must be divisible by patch size ({config['patch_size']}).")

        self.num_of_channels = config['num_of_channels']
        self.patch_size = config['patch_size']
        self.num_of_patches = (config['img_size'] // config['patch_size']) ** 2

        # Encoder: Image -> Latent
        self.encoder = ViTEncoder(config['img_size'], self.patch_size, self.num_of_channels, config['embed_dim'], config['enc_depth'], config['num_heads'], config['mlp_dim'])
        # Decoder: Latent -> Reconstructed patches
        self.decoder = ViTDecoder(self.num_of_patches, self.patch_size, self.num_of_channels, config['embed_dim'], config['dec_depth'], config['num_heads'], config['mlp_dim'])

        self.current_row_num = config['som_rows']
        self.current_col_num = config['som_cols']
        # SOM weights as a torch Parameter
        # shape (Num_SOM_Nodes, Num_Patches * Embed_Dim)
        self.som_dim = self.num_of_patches * config['embed_dim']
        self.som_weights = nn.Parameter(torch.randn(self.current_row_num * self.current_col_num, self.som_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | bool]:
        """
        Forward pass for AutoEncoder
        :param x: Input images of shape ``(batch_size, in_channels, img_size, img_size)``
        :return: Tuple containing:
                 - output: Reconstructed images of shape ``(batch_size, num_channels, img_size, img_size)``
                 - latent: Latent representations of shape ``(batch_size, num_patches + 1, embed_dim)``
        """
        latent = self.encoder(x)
        patched_output = self.decoder(latent)
        output = unpatch(patched_output, self.patch_size, self.num_of_channels)

        return output, latent

    def get_num_of_neurons(self) -> int:
        """
        Returns the current number of neurons
        :return Number of neurons
        """
        return self.som_weights.size()[0]

    def get_sigma(self) -> float:
        """
        Calculates the initial sigma for the SOM as half of the image size
        :return: Sigma value
        """
        return np.ceil(min(self.current_row_num, self.current_col_num) / 2)

    def get_som_shape(self) -> tuple[int, int]:
        """
        Returns the shape of the SOM grid
        :return: A tuple (rows, cols)
        """
        return self.current_row_num, self.current_col_num

    def get_som_weights(self) -> torch.Tensor:
        """
        Returns the current weights of the SOM
        :return: Tensor of shape ``(som_rows * som_cols, num_patches * embed_dim)``
        """
        return self.som_weights

    def get_weight_of_node(self, flat_idx: int) -> torch.Tensor:
        """
        Returns the weight of the node in the SOM at the given index
        :param flat_idx: Flat index of the neuron
        :return: Weight Tensor
        """
        return self.som_weights[flat_idx]

    def find_dissimilar_neighbour(self,
                                  e_index: tuple[int, int],
                                  e_index_flat: int) -> tuple[int, int]:
        """
        Method to find the most dissimilar neighbour in a rectangular grid to the neuron unit e at given index
        :param e_index: Index on the 2d grid of the neuron unit e
        :param e_index_flat: Flat index of the neuron unit e in the Parameter
        :return: Index of the most dissimilar neighbour d to the unit e
        """
        e_weight = self.get_weight_of_node(e_index_flat)
        r, c = e_index
        max_dist = -1.0
        d_idx = None

        coords_neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for rn, cn in coords_neighbours:
            if 0 <= cn < self.current_col_num and 0 <= rn < self.current_row_num:
                # calculate flat index of neighbour to get the weight
                flat_idx_n = rn * self.current_col_num + cn
                neighbor_weight = self.get_weight_of_node(flat_idx_n)

                dist = 1 - F.cosine_similarity(e_weight, neighbor_weight, dim=0)
                if dist > max_dist:
                    max_dist = dist
                    d_idx = (rn, cn)

        return d_idx

    def add_col_between(self, col1: int, col2: int):
        """
        Method which inserts a new column between given column indices
        :param col1: Index of the first column
        :param col2: Index of the second column
        """
        flat_weights = self.som_weights.data
        grid_weights = flat_weights.view(self.current_row_num, self.current_col_num, self.som_dim)

        insert_idx = max(col1, col2)

        # calculate the weights of new column as a mean of neighbours
        col_left = grid_weights[:, col1:col1 + 1, :]
        col_right = grid_weights[:, col2:col2 + 1, :]
        new_col = (col_left + col_right) / 2

        part_left = grid_weights[:, :insert_idx, :]
        part_right = grid_weights[:, insert_idx:, :]
        new_grid = torch.cat([part_left, new_col, part_right], dim=1)

        self.current_col_num += 1
        self.som_weights = nn.Parameter(new_grid.reshape(-1, self.som_dim))

    def add_row_between(self, row1: int, row2: int):
        """
        Method which inserts a new row between given row indices
        :param row1: Index of the first row
        :param row2: Index of the second row
        """
        flat_weights = self.som_weights.data
        grid_weights = flat_weights.view(self.current_row_num, self.current_col_num, self.som_dim)

        insert_idx = max(row1, row2)

        # calculate the weights of new row as a mean of neighbours
        row_top = grid_weights[row1:row1 + 1, :, :]
        row_bottom = grid_weights[row2:row2 + 1, :, :]
        new_row = (row_top + row_bottom) / 2

        part_top = grid_weights[:insert_idx, :, :]
        part_bottom = grid_weights[insert_idx:, :, :]
        new_grid = torch.cat([part_top, new_row, part_bottom], dim=0)

        self.current_row_num += 1
        self.som_weights = nn.Parameter(new_grid.reshape(-1, self.som_dim))

    def grow(self, unit_error_matrix: np.ndarray):
        """
        Method which grows the grid by one row or one column
        :param unit_error_matrix: Error matrix to find the neuron unit e with the highest error
        """
        e_index_flat = np.argmax(unit_error_matrix)
        e_index = np.unravel_index(e_index_flat, unit_error_matrix.shape)

        d_index = self.find_dissimilar_neighbour(e_index, e_index_flat)

        er, ec = e_index
        dr, dc = d_index

        if er == dr:  # same row, adding column between their cols
            self.add_col_between(ec, dc)
        elif ec == dc:  # same col, adding row between their rows
            self.add_row_between(er, dr)
        else:
            raise ValueError("e_unit and d_unit not adjacent")

    def calculate_unit_errors(self,
                              loader: torch.utils.data.DataLoader,
                              device: torch.device) -> np.ndarray[np.ndarray[float]]:
        """
        Method calculating error for each neuron unit
        :param loader: Dataloader
        :param device: The torch device
        :return: Error for each unit
        """

        unit_errors = np.zeros((self.current_row_num, self.current_col_num))

        total_samples = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                total_samples += images.shape[0]

                latent = self.encoder(images)
                patches = latent[:, 1:, :]
                som_input = patches.reshape(patches.shape[0], -1)

                # find bmu for batch
                dists = cosine_distance_torch(self.get_som_weights(), som_input)

                # get min distance and flat index in the batch
                # min_dists: minimal error value for each image
                # flat_indices: flatted indices of winning node (0,n-1)
                min_dists, flat_indices = torch.min(dists, dim=1)
                min_dists = min_dists.cpu().numpy()
                flat_indices = flat_indices.cpu().numpy()

                # e.g. flat index 7 in 5 col grid -> row 1, col 2
                row_indices = flat_indices // self.current_col_num
                col_indices = flat_indices % self.current_col_num

                # add minimal distance to the indices
                np.add.at(unit_errors, (row_indices, col_indices), min_dists)

        return unit_errors

    def start_growth(self, loader: torch.utils.data.DataLoader, device: torch.device):
        """
        Method which starts growing grid by one row or one column
        :param loader: Dataloader
        :param device: The torch device
        """
        self.eval()
        unit_errors = self.calculate_unit_errors(loader, device)

        self.grow(unit_errors)
        print(f"Current grid size: ({self.current_row_num}, {self.current_col_num})")
        self.to(device)
        self.train()