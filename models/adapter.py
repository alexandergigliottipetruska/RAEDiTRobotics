import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

class TraineableAdapter(nn.Module):
    """
    Trainable 2-layer MLP adapter that projects DINOv3 tokens from 1024 to 512 dimensions.
    """
    def __init__(self):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

    def forward(self, z):
        """
        Args:
            z: Normalized patch tokens from encoder of shape (B, K, 196, 1024)
        Returns:
            z_bar: Adapted tokens of shape (B, K, 196, 512)
        """
        z_bar = self.adapter(z)
        return z_bar


def noise_augment(z_bar, tau=0.8, training=True):
    """
    Applies half-normal noise to the adapted tokens. 
    
    Args:
        z_bar: Adapted tokens of shape (B, K, 196, 512)
        tau: Noise scale factor (default: 0.8)
        training: Boolean flag. If False, returns z_bar unchanged.
    Returns:
        z_tilde: Noised adapted tokens of shape (B, K, 196, 512)
    """
    if not training:
        return z_bar
        
    # Sample sigma from a half-normal distribution
    sigma = torch.abs(torch.randn(1, device=z_bar.device) * tau)
    
    # Add Gaussian noise scaled by sigma
    z_tilde = z_bar + torch.randn_like(z_bar) * sigma
    
    return z_tilde


if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = TraineableAdapter().to(device)
    
    # use dummy input for testing
    B, K, N, d = 2, 4, 196, 1024
    dummy_z = torch.randn(B, K, N, d).to(device)
    
    z_bar = adapter(dummy_z)
    
    # test output shape
    expected_shape = (B, K, N, 512)
    assert z_bar.shape == expected_shape, f"Error: Expected {expected_shape}, got {z_bar.shape}"
    
    # test noise augmentation in training mode
    z_tilde_train = noise_augment(z_bar, tau=0.8, training=True)
    assert z_tilde_train.shape == expected_shape, "Error: Noise augmentation changed tensor shape!"
    
    # test noise was added
    assert not torch.allclose(z_bar, z_tilde_train), "Error: z_tilde is identical to z_bar during training!"
    
    # test noise augmentation in evaluation mode
    z_tilde_eval = noise_augment(z_bar, tau=0.8, training=False)
    
    # test no noise was added
    assert torch.allclose(z_bar, z_tilde_eval), "Error: Noise was added even when training=False!"

    print("\n✅ All Adapter Tests Passed successfully!")