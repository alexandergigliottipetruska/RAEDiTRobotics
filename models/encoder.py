import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import login
import yaml

# Load the secret things from the YAML file
with open("configs/secrets.yaml", "r") as file:
    secrets = yaml.safe_load(file)
    
# Log in using the token
login(token=secrets["huggingface_token"])

""" TODO: need to create a new conda environment to fix the OpenMP error with transforners and torch"""

class FrozenMultiViewEncoder(nn.Module):
    """
    Frozen DINOv3 wrapper using Hugging Face, with Cancel-Affine LayerNorm for Multi-View processing.
    """
    def __init__(self):
        super().__init__()
        
        # load the DINOv3 ViT-L16 from Hugging Face
        self.backbone = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
        self.expected_tokens = 196
        
        # Cancel-Affine LayerNorm
        self.cancel_affine_ln = nn.LayerNorm(1024, elementwise_affine=False)
        
        self.freeze_all()

    def freeze_all(self):
        """Freezes the backbone parameters and sets it to eval mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        """Override train to ensure the backbone stays in eval mode."""
        super().train(mode)
        self.backbone.eval()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, K, 3, 224, 224)
        Returns:
            z: Adapted tokens of shape (B, K, 196, 1024)
        """
        B, K, C, H, W = x.shape
        
        # collapse B and K for a single faster forward pass
        x_flat = x.view(B * K, C, H, W)
        
        with torch.no_grad():
            # forward pass through backbone
            outputs = self.backbone(pixel_values=x_flat)
            
            # Extract the patch tokens (skip CLS token and register tokens by taking only the expected number)
            patch_tokens = outputs.last_hidden_state[:, -self.expected_tokens:, :]
            
        # Apply the cancel-affine LayerNorm per token
        z_flat = self.cancel_affine_ln(patch_tokens)
        
        # Reshape back to separate the views
        z = z_flat.view(B, K, self.expected_tokens, 1024)
        
        return z


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrozenMultiViewEncoder().to(device)
    
    # check if frozen
    any_req_grad = any(p.requires_grad for p in model.parameters())
    assert not any_req_grad, "Error: Some encoder parameters have requires_grad=True!"
    
    checksum_before = sum(p.sum().item() for p in model.parameters())
    
    # use dummy data
    B, K = 2, 4
    dummy_input = torch.randn(B, K, 3, 224, 224).to(device)
    
    output = model(dummy_input)
    
    # check if weights changed
    checksum_after = sum(p.sum().item() for p in model.parameters())
    assert checksum_before == checksum_after, "Error: Encoder weights changed during forward pass!"
    
    # check expected output shape
    expected_shape = (B, K, 196, 1024) 
    assert output.shape == expected_shape, f"Error: Expected {expected_shape}, got {output.shape}"
    
    # check cancel-affine LayerNorm stats
    mean_val = output.mean().item()
    std_val = output.std().item()
    assert abs(mean_val) < 0.1, f"Error: Mean is not close to 0 (Got: {mean_val})"
    assert abs(std_val - 1.0) < 0.1, f"Error: Std is not close to 1 (Got: {std_val})"

    print("\nAll Encoder Tests Passed successfully!")