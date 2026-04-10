"""Quick benchmark: DINOv3-L vs DINOv3-S forward pass speed.

Usage:
    PYTHONPATH=. python training/bench_dinov3.py
"""
import time
import yaml
import torch
from transformers import AutoModel
from huggingface_hub import login

with open("configs/secrets.yaml", "r") as f:
    secrets = yaml.safe_load(f)
login(token=secrets["huggingface_token"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "DINOv3-S/16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "DINOv3-B/16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "DINOv3-L/16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

dummy = torch.randn(1, 3, 224, 224, device=device)

for name, model_id in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Loading {name} ({model_id})...")
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    except Exception as e:
        print(f"  FAILED to load: {e}")
        continue

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    # Get output shape
    with torch.no_grad():
        out = model(pixel_values=dummy)
        tokens = out.last_hidden_state
        print(f"  Output: {tokens.shape}  (last_hidden_state)")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(pixel_values=dummy)
    torch.cuda.synchronize()

    # Benchmark: single image
    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            model(pixel_values=dummy)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"  Single image:  {dt:.1f} ms")

    # Benchmark: batch of 4 (one per camera view)
    dummy4 = torch.randn(4, 3, 224, 224, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            model(pixel_values=dummy4)
    torch.cuda.synchronize()
    dt4 = (time.perf_counter() - t0) / N * 1000
    print(f"  Batch of 4:    {dt4:.1f} ms")

    # Benchmark: batch of 8 (4 cameras × 2 obs horizon)
    dummy8 = torch.randn(8, 3, 224, 224, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            model(pixel_values=dummy8)
    torch.cuda.synchronize()
    dt8 = (time.perf_counter() - t0) / N * 1000
    print(f"  Batch of 8:    {dt8:.1f} ms")

    del model
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("Done.")
