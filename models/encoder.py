import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from types import SimpleNamespace

""" TODO: need to create a new conda environment to fix the OpenMP error with transforners and torch"""

class _MockBackbone(nn.Module):
    """Lightweight frozen substitute for DINOv3-L in mock mode.

    Pools + projects image to 196 tokens of 1024 dimensions.
    All parameters frozen. Returns object with last_hidden_state.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(14)   # (B, 3, 14, 14) = 196 positions
        self.proj = nn.Linear(3, 1024)

    def forward(self, pixel_values):
        x = self.pool(pixel_values)            # (B, 3, 14, 14)
        x = x.permute(0, 2, 3, 1)             # (B, 14, 14, 3)
        x = x.reshape(x.shape[0], 196, 3)     # (B, 196, 3)
        tokens = self.proj(x)                  # (B, 196, 1024)
        return SimpleNamespace(last_hidden_state=tokens)


class FrozenMultiViewEncoder(nn.Module):
    """
    Frozen DINOv3 wrapper using Hugging Face, with Cancel-Affine LayerNorm for Multi-View processing.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            from transformers import AutoModel
            from huggingface_hub import login
            import yaml
            
            # Log in using secrets.yaml if available, otherwise assume huggingface-cli login
            try:
                with open("configs/secrets.yaml", "r") as file:
                    secrets = yaml.safe_load(file)
                login(token=secrets["huggingface_token"])
            except FileNotFoundError:
                pass  # assume huggingface-cli login was already used
            
            # load the DINOv3 ViT-L16 from Hugging Face
            self.backbone = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', trust_remote_code=True)
        else:
            self.backbone = _MockBackbone()

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
        return self

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224) ImageNet-normalized images
        Returns:
            z: Normalized patch tokens of shape (B, 196, 1024)
        """
        with torch.no_grad():
            # forward pass through backbone
            outputs = self.backbone(pixel_values=x)
            
            # Extract the patch tokens (skip CLS token and register tokens by taking only the expected number)
            patch_tokens = outputs.last_hidden_state[:, -self.expected_tokens:, :]
            
        # Apply the cancel-affine LayerNorm per token
        z = self.cancel_affine_ln(patch_tokens)
        
        return z


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrozenMultiViewEncoder(pretrained=False).to(device)
    
    # check if frozen
    any_req_grad = any(p.requires_grad for p in model.parameters())
    assert not any_req_grad, "Error: Some encoder parameters have requires_grad=True!"
    
    checksum_before = sum(p.sum().item() for p in model.parameters())
    
    # use dummy data
    B = 2
    dummy_input = torch.randn(B, 3, 224, 224).to(device)
    
    output = model(dummy_input)
    
    # check if weights changed
    checksum_after = sum(p.sum().item() for p in model.parameters())
    assert checksum_before == checksum_after, "Error: Encoder weights changed during forward pass!"
    
    # check expected output shape
    expected_shape = (B, 196, 1024) 
    assert output.shape == expected_shape, f"Error: Expected {expected_shape}, got {output.shape}"
    
    # check cancel-affine LayerNorm stats
    mean_val = output.mean().item()
    std_val = output.std().item()
    assert abs(mean_val) < 0.1, f"Error: Mean is not close to 0 (Got: {mean_val})"
    assert abs(std_val - 1.0) < 0.1, f"Error: Std is not close to 1 (Got: {std_val})"

    print("\nAll Encoder Tests Passed successfully!")
