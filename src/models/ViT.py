import torch
import torch.nn as nn

# https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
class PatchEmbedding(nn.Module):
    """
    Class to split images into patches and embed them using convolutional layers
    """

    def __init__(self,
                 img_size: int = 28,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 16):
        """
        Constructor for PatchEmbedding
        :param img_size: Size of input image. Defaults to ``28``
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param in_channels: Number of input channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # convolution with the stride size same as patch size -> no overlapping
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create patch embeddings
        :param x: Input batch of images of shape ``(batch_size, in_channels, img_size, img_size)``
        :return: Patch embeddings of shape ``(batch_size, num_patches, embed_dim)``
        """
        # Example: batch = 8, embed_dim=64, img_height=28, img_width=28, input_channels=1, patch_size=4
        # x.shape: (8, 1, 28, 28)
        # 28 / 4 = 7 -> 7x7 grid
        # proj(x).shape: (8, 64, 7, 7)
        # proj(x).flatten(2): (8, 64, 7 * 7)
        # proj(x).flatten(2).transpose(1, 2): (8, 49, 16) : (B, 7x7 grid as sequence, embed_dim)
        x = self.proj(x).flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron class
    """

    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float):
        """
        Constructor for MLP
        :param embed_dim: The embedding dimension
        :param mlp_dim: The dimension of the hidden layer
        :param dropout: The dropout probability
        """
        super().__init__()
        self.dense_1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP
        :param x: Input tensor of shape ``(batch_size, seq_len, embed_dim)``
        :return: Output tensor of shape ``(batch_size, seq_len, embed_dim)``
        """
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Encoder Block consisting of Self-Attention and MLP
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout: float = 0.1):
        """
        Constructor for Block
        :param embed_dim: The embedding dimension
        :param num_heads: The number of attention heads
        :param mlp_dim: The dimension of the hidden layer
        :param dropout: The dropout probability. Defaults to ``0.1``
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder Block
        :param x: Input tensor of shape ``(batch_size, seq_len, embed_dim)``
        :return: Output tensor of shape ``(batch_size, seq_len, embed_dim)``
        """
        # Self-attention
        attention_output, _ = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.ln2(x))
        # Skip connection
        x = x + mlp_output
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.
    """

    def __init__(self,
                 img_size: int = 28,
                 patch_size: int = 4,
                 in_channels: int = 1,
                 embed_dim: int = 16,
                 depth: int = 4,
                 num_heads: int = 2,
                 mlp_dim: int = 64):
        """
        Constructor for ViTEncoder
        :param img_size: Size of input image. Defaults to ``28``
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param in_channels: Number of input channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        :param depth: The number of transformer encoder blocks. Defaults to ``4``
        :param num_heads: The number of attention heads. Defaults to ``2``
        :param mlp_dim: The dimension of the hidden layer. Defaults to ``64``
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # learnable positional embedding and cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            block = Block(embed_dim, num_heads, mlp_dim)
            self.blocks.append(block)

        self.ln1 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ViTEncoder
        :param x: Input images of shape ``(batch_size, in_channels, img_size, img_size)``
        :return: Encoded features of shape ``(batch_size, num_patches + 1, embed_dim)``
        """
        # create patches
        # x shape: (batch_size, num_patches, embed_dim)
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add CLS token
        # cls_tokens shape: (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # x shape: (batch_size, num_patches + 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embedding
        x = x + self.pos_embed

        # apply self-attention layers and mlp
        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        return x


class ViTDecoder(nn.Module):
    """
    Vision Transformer Decoder.
    """

    def __init__(self,
                 num_patches: int,
                 patch_size: int = 4,
                 output_dim: int = 1,
                 embed_dim: int = 16,
                 depth: int = 2,
                 num_heads: int = 2,
                 mlp_dim: int = 64):
        """
        Constructor for ViTDecoder
        :param num_patches: Total number of patches
        :param patch_size: Size of individual patch. Defaults to ``4``
        :param output_dim: Number of output channels. Defaults to ``1``
        :param embed_dim: The embedding dimension. Defaults to ``16``
        :param depth: The number of transformer encoder blocks. Defaults to ``4``
        :param num_heads: The number of attention heads. Defaults to ``2``
        :param mlp_dim: The dimension of the hidden layer. Defaults to ``64``
        """
        super().__init__()

        # reconstruction to original pixels: patch_size * patch_size * channels
        self.pixels_per_patch = patch_size * patch_size * output_dim
        self.num_patches = num_patches
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # same list of transformer blocks as in encoder
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            block = Block(embed_dim, num_heads, mlp_dim)
            self.blocks.append(block)

        self.ln1 = nn.LayerNorm(embed_dim)

        # final projection to map embeddings back to pixel values
        self.head = nn.Linear(embed_dim, self.pixels_per_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ViTDecoder
        :param x: Encoded features of shape ``(batch_size, num_patches + 1, embed_dim)``
        :return: Reconstructed patches of shape ``(batch_size, num_patches, pixels_per_patch)``
        """
        # positional embeddings in latent space
        x = x + self.pos_embed

        # applying self attention layers and mlp
        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        # removing CLS token (8, 50, 64) -> (8, 49, 64)
        x = x[:, 1:, :]

        # projection back to pixel space (8, 49, 64) -> (8, 49, 16): (B, grid 7x7, pixels per patch)
        x = self.head(x)
        return x