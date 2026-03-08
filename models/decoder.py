import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

class ViTDecoder(nn.Module):
    """
    ViT-based Representation Autoencoder (RAE) Decoder.
    Reconstructs an image from adapted ViT tokens.
    """
    def __init__(self, hidden_dim=512, num_layers=8, num_heads=8, patch_size=16, img_size=224):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 224 // 16 = 14
        self.num_patches = self.grid_size ** 2   # 14 * 14 = 196
        self.out_channels = 3

        # learned positional embeddings for each patch token (196 tokens, each with hidden_dim features)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        
        # Transformer Encoder blocks to process the token sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to project from hidden_dim back to pixel space (16*16*3 = 768)
        self.pixels_per_patch = (patch_size ** 2) * self.out_channels
        self.head = nn.Linear(hidden_dim, self.pixels_per_patch)

    def unpatchify(self, x):
        """
        Reshapes the sequence of patch pixels back into a 2D image.
        Mapping: 196 x 768 -> 14 x 14 x 16 x 16 x 3 -> 224 x 224 x 3
        
        Args:
            x: Tensor of shape (B*K, 196, 768)
        Returns:
            imgs: Tensor of shape (B*K, 3, 224, 224)
        """
        B_K = x.shape[0]
        p = self.patch_size
        h = w = self.grid_size
        
        # Reshape to separate the patch pixels and channels
        x = x.view(B_K, h, w, p, p, self.out_channels)
        
        # Permute to move the channels to the correct position for unpatchifying
        x = x.permute(0, 5, 1, 3, 2, 4)
        
        # Finally, reshape to get the full image dimensions
        imgs = x.reshape(B_K, self.out_channels, h * p, w * p)
        return imgs

    def forward(self, z_tilde):
        """
        Args:
            z_tilde: Noised adapted tokens of shape (B, K, 196, 512)
        Returns:
            I_hat: Reconstructed images of shape (B, K, 3, 224, 224)
        """
        B, K, N, D = z_tilde.shape
        
        # Collapse B and K for a single forward pass through the transformer
        x = z_tilde.view(B * K, N, D)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        x = self.transformer(x)
        
        # Linear projection to pixel space
        x = self.head(x)
        
        # Reshape to image dimensions
        img_flat = self.unpatchify(x)
        
        # Separate B and K back out to match the input format
        I_hat = img_flat.view(B, K, self.out_channels, self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        
        return I_hat


if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = ViTDecoder().to(device)
    
    # use dummy input for testing
    B, K, N, D = 2, 4, 196, 512
    dummy_z_tilde = torch.randn(B, K, N, D).to(device)
    
    I_hat = decoder(dummy_z_tilde)
    
    # test output shape
    expected_shape = (B, K, 3, 224, 224)
    assert I_hat.shape == expected_shape, f"Error: Expected {expected_shape}, got {I_hat.shape}"

    print("\n All Decoder Tests Passed successfully!")