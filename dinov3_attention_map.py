"""
DINOv3 Attention Map Generator for Medical Scan Overlay

Generates self-attention maps from DINOv3 (ViT-S/16) and overlays them
on medical images (DICOM, NIfTI, PNG/JPG) for anomaly visualization.

The model (~85MB) is auto-downloaded from HuggingFace on first run.
Runs on GPU if available, otherwise CPU.

Usage:
    python dinov3_attention_map.py --image scan.png
    python dinov3_attention_map.py --image scan.png --output overlay.png --head 3
    python dinov3_attention_map.py --image volume.nii.gz --slice-idx 50
    python dinov3_attention_map.py --image scan.dcm --colormap hot --alpha 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Optional imports checked at runtime
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_medical_image(image_path: str, slice_idx: int | None = None) -> np.ndarray:
    """Load image from various medical formats. Returns 2D numpy array (H, W) in [0, 255]."""
    path = Path(image_path)
    suffix = path.suffix.lower()

    # NIfTI
    if suffix in (".nii", ".gz"):
        import nibabel as nib
        vol = nib.load(str(path)).get_fdata()
        if vol.ndim == 3:
            if slice_idx is None:
                slice_idx = vol.shape[2] // 2
            img = vol[:, :, slice_idx]
        elif vol.ndim == 2:
            img = vol
        else:
            # 4D: take first volume, middle slice
            vol = vol[..., 0]
            img = vol[:, :, slice_idx or vol.shape[2] // 2]
        # Normalize to 0-255
        img = img.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        return img.astype(np.uint8)

    # DICOM
    if suffix == ".dcm":
        try:
            import pydicom
        except ImportError:
            sys.exit("pydicom is required for DICOM files: pip install pydicom")
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        return img.astype(np.uint8)

    # Standard image (PNG, JPG, etc.)
    img = np.array(Image.open(path).convert("L"))
    return img


def preprocess_for_dinov3(img_gray: np.ndarray, device: torch.device, patch_size: int = 14) -> torch.Tensor:
    """Preprocess a grayscale image for DINO/DINOv2/DINOv3. Returns tensor (1, 3, H, W)."""
    # Image size must be divisible by patch_size. Common sizes:
    #   patch_size=14 (DINOv2) → 518 (37 patches) or 224 (16 patches)
    #   patch_size=16 (DINOv3) → 224 (14 patches)
    img_size = (patch_size * 16)  # 224 for ps=14, 256 for ps=16

    img_pil = Image.fromarray(img_gray).resize((img_size, img_size), Image.LANCZOS)
    img_np = np.array(img_pil, dtype=np.float32) / 255.0

    # Replicate grayscale to 3 channels
    img_3ch = np.stack([img_np] * 3, axis=0)  # (3, H, W)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_3ch = (img_3ch - mean) / std

    tensor = torch.from_numpy(img_3ch).float().unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device)


def extract_attention_maps(model, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract CLS-to-patch attention from the last transformer block.
    Works with DINOv2/DINOv3 HuggingFace models by hooking into the
    self-attention layer and computing softmax(Q @ K^T / sqrt(d)).
    Returns: attention maps of shape (num_heads, h_patches, w_patches)
    """
    attn_store = {}

    # Find the last self-attention module (Dinov2SelfAttention or similar)
    last_attn_module = None
    for name, mod in model.named_modules():
        if hasattr(mod, "query") and hasattr(mod, "key") and hasattr(mod, "value"):
            last_attn_module = (name, mod)

    if last_attn_module is None:
        raise RuntimeError("Could not find a self-attention layer with query/key/value in the model.")

    attn_name, attn_mod = last_attn_module
    num_heads = attn_mod.attention.num_attention_heads if hasattr(attn_mod, "attention") else model.config.num_attention_heads

    def hook_fn(module, input, output):
        """Capture input to self-attention, compute attention weights manually."""
        hidden_states = input[0]  # (B, N, C)
        B, N, C = hidden_states.shape

        q = module.query(hidden_states)  # (B, N, C)
        k = module.key(hidden_states)    # (B, N, C)

        head_dim = C // num_heads
        # Reshape to (B, num_heads, N, head_dim)
        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)
        attn_weights = attn_weights.softmax(dim=-1)

        attn_store["attn"] = attn_weights.detach()

    handle = attn_mod.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(img_tensor)

    handle.remove()

    if "attn" not in attn_store:
        raise RuntimeError("Attention hook did not fire.")

    # attn_store["attn"] shape: (1, num_heads, N+1, N+1)
    last_attn = attn_store["attn"]

    # CLS token attention to all patch tokens (row 0, columns 1:)
    cls_attn = last_attn[0, :, 0, 1:]  # (num_heads, num_patches)

    # Reshape to spatial grid
    num_patches = cls_attn.shape[-1]
    h = w = int(num_patches ** 0.5)
    attn_maps = cls_attn.reshape(-1, h, w)  # (num_heads, h, w)

    return attn_maps.cpu().numpy()


def create_overlay(
    img_gray: np.ndarray,
    attn_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create an overlay of attention map on the original image.
    Returns RGB image as numpy array (H, W, 3) in [0, 255].
    """
    h, w = img_gray.shape

    # Upscale attention map to original image size
    attn_tensor = torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0)
    attn_upscaled = F.interpolate(attn_tensor, size=(h, w), mode="bilinear", align_corners=False)
    attn_upscaled = attn_upscaled.squeeze().numpy()

    # Normalize to [0, 1]
    attn_upscaled = (attn_upscaled - attn_upscaled.min()) / (attn_upscaled.max() - attn_upscaled.min() + 1e-8)

    # Apply colormap
    if HAS_MATPLOTLIB:
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(attn_upscaled)[:, :, :3]  # (H, W, 3) in [0, 1]
    else:
        # Fallback: simple red-yellow heatmap without matplotlib
        heatmap = np.zeros((h, w, 3), dtype=np.float32)
        heatmap[:, :, 0] = attn_upscaled  # Red channel
        heatmap[:, :, 1] = attn_upscaled * 0.5  # Partial green -> yellow tones
        heatmap[:, :, 2] = 0

    # Convert grayscale to RGB
    img_rgb = np.stack([img_gray / 255.0] * 3, axis=-1)

    # Alpha blend
    overlay = (1 - alpha) * img_rgb + alpha * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def save_result(overlay: np.ndarray, output_path: str):
    """Save the overlay image."""
    Image.fromarray(overlay).save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Attention Map Overlay for Medical Scans")
    parser.add_argument("--image", required=True, help="Path to input image (PNG, JPG, NIfTI, DICOM)")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_attention.png)")
    parser.add_argument("--head", type=int, default=None,
                        help="Attention head index to visualize (default: mean of all heads)")
    parser.add_argument("--slice-idx", type=int, default=None,
                        help="Slice index for 3D volumes (NIfTI). Default: middle slice")
    parser.add_argument("--colormap", default="jet", help="Matplotlib colormap (default: jet)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity (default: 0.45)")
    parser.add_argument("--save-all-heads", action="store_true",
                        help="Save individual attention maps for each head")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold (0-1) to binarize attention for anomaly highlighting")
    parser.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or 'mps'")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading DINOv3 model (auto-download on first run)...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    img_gray = load_medical_image(args.image, args.slice_idx)
    img_tensor = preprocess_for_dinov3(img_gray, device)

    # Extract attention maps
    print("Extracting attention maps...")
    attn_maps = extract_attention_maps(model, img_tensor)
    num_heads = attn_maps.shape[0]
    print(f"Extracted {num_heads} attention heads, spatial size: {attn_maps.shape[1]}x{attn_maps.shape[2]}")

    # Select attention map
    if args.head is not None:
        if args.head >= num_heads:
            sys.exit(f"Head index {args.head} out of range (model has {num_heads} heads)")
        selected_attn = attn_maps[args.head]
        print(f"Using attention head {args.head}")
    else:
        selected_attn = attn_maps.mean(axis=0)
        print("Using mean of all attention heads")

    # Apply threshold if requested
    if args.threshold is not None:
        selected_attn_norm = (selected_attn - selected_attn.min()) / (selected_attn.max() - selected_attn.min() + 1e-8)
        selected_attn = np.where(selected_attn_norm >= args.threshold, selected_attn, 0)
        print(f"Applied threshold: {args.threshold}")

    # Create and save overlay
    output_path = args.output or str(Path(args.image).stem + "_attention.png")
    overlay = create_overlay(img_gray, selected_attn, args.colormap, args.alpha)
    save_result(overlay, output_path)

    # Save all heads if requested
    if args.save_all_heads:
        out_dir = Path(output_path).parent / f"{Path(args.image).stem}_heads"
        out_dir.mkdir(exist_ok=True)
        for h in range(num_heads):
            head_overlay = create_overlay(img_gray, attn_maps[h], args.colormap, args.alpha)
            save_result(head_overlay, str(out_dir / f"head_{h:02d}.png"))
        print(f"All {num_heads} head overlays saved to {out_dir}/")


if __name__ == "__main__":
    main()
