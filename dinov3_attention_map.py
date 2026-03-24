"""
DINO Attention Map Generator for Medical Scan Overlay

Generates self-attention maps from DINOv2/DINOv3/3DINO models and overlays them
on medical images (DICOM, NIfTI, PNG/JPG) for anomaly visualization.

Supports:
  - CT windowing (brain, bone, subdural, stroke presets)
  - Multi-layer attention extraction (any transformer block, not just last)
  - Percentile-based medical image normalization
  - Native 3DINO model loading from .pth weights
  - Higher resolution inference (518px for DINOv2)

Usage:
    python dinov3_attention_map.py --image scan.png
    python dinov3_attention_map.py --image scan.dcm --window brain
    python dinov3_attention_map.py --image volume.nii.gz --slice-idx 50 --layer 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# CT windowing presets: (window_width, window_level)
# ---------------------------------------------------------------------------
CT_WINDOW_PRESETS = {
    "brain":    (80, 40),
    "subdural": (200, 75),
    "stroke":   (40, 40),
    "bone":     (2000, 500),
    "lung":     (1500, -600),
    "soft":     (400, 50),
    "none":     None,
}


def apply_ct_window(img: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Apply CT windowing (window/level) to an image in Hounsfield units.
    Returns uint8 array in [0, 255]."""
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower + 1e-8) * 255
    return img.astype(np.uint8)


def _dicom_to_hu(ds) -> np.ndarray:
    """Convert DICOM pixel_array to Hounsfield Units using RescaleSlope/Intercept."""
    img = ds.pixel_array.astype(np.float64)
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    return img * float(slope) + float(intercept)


def load_medical_image(
    image_path: str,
    slice_idx: int | None = None,
    ct_window: str | None = None,
    custom_wl: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load image from various medical formats.

    Args:
        image_path: path to image file
        slice_idx: slice index for 3D volumes
        ct_window: preset name from CT_WINDOW_PRESETS (e.g. "brain", "bone")
        custom_wl: (window_width, window_level) tuple, overrides ct_window

    Returns:
        2D numpy array (H, W) in [0, 255] as uint8.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    # NIfTI
    if suffix in (".nii", ".gz"):
        import nibabel as nib
        vol = nib.load(str(path)).get_fdata()
        if vol.ndim == 3:
            if slice_idx is None:
                slice_idx = vol.shape[2] // 2
            img = vol[:, :, slice_idx].astype(np.float64)
        elif vol.ndim == 2:
            img = vol.astype(np.float64)
        else:
            vol = vol[..., 0]
            img = vol[:, :, slice_idx or vol.shape[2] // 2].astype(np.float64)

        # Apply windowing if requested
        if custom_wl is not None:
            return apply_ct_window(img, custom_wl[0], custom_wl[1])
        if ct_window and ct_window != "none" and ct_window in CT_WINDOW_PRESETS:
            ww, wl = CT_WINDOW_PRESETS[ct_window]
            return apply_ct_window(img, ww, wl)
        # Default: percentile normalization
        p_low, p_high = np.percentile(img, [0.5, 99.5])
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8) * 255
        return img.astype(np.uint8)

    # DICOM
    if suffix == ".dcm":
        try:
            import pydicom
        except ImportError:
            sys.exit("pydicom is required for DICOM files: pip install pydicom")
        ds = pydicom.dcmread(str(path))
        img = _dicom_to_hu(ds)

        if custom_wl is not None:
            return apply_ct_window(img, custom_wl[0], custom_wl[1])
        if ct_window and ct_window != "none" and ct_window in CT_WINDOW_PRESETS:
            ww, wl = CT_WINDOW_PRESETS[ct_window]
            return apply_ct_window(img, ww, wl)
        # Auto-detect window from DICOM header
        ww = getattr(ds, "WindowWidth", None)
        wl = getattr(ds, "WindowCenter", None)
        if ww is not None and wl is not None:
            ww = float(ww) if not hasattr(ww, '__iter__') else float(ww[0])
            wl = float(wl) if not hasattr(wl, '__iter__') else float(wl[0])
            return apply_ct_window(img, ww, wl)
        # Fallback: percentile
        p_low, p_high = np.percentile(img, [0.5, 99.5])
        img = np.clip(img, p_low, p_high)
        img = (img - p_low) / (p_high - p_low + 1e-8) * 255
        return img.astype(np.uint8)

    # Standard image (PNG, JPG)
    img = np.array(Image.open(path).convert("L"))
    return img


def preprocess_for_dino(
    img_gray: np.ndarray,
    device: torch.device,
    patch_size: int = 14,
    high_res: bool = True,
) -> torch.Tensor:
    """Preprocess a grayscale image for DINO models.

    Args:
        img_gray: (H, W) uint8 grayscale
        device: torch device
        patch_size: model patch size
        high_res: if True, use higher resolution (37 patches per side for ps=14)

    Returns:
        tensor (1, 3, H, W) on device, ImageNet-normalized
    """
    if high_res and patch_size == 14:
        img_size = patch_size * 37  # 518 — DINOv2's native high-res
    else:
        img_size = patch_size * 16  # 224 or 256

    img_pil = Image.fromarray(img_gray).resize((img_size, img_size), Image.LANCZOS)
    img_np = np.array(img_pil, dtype=np.float32) / 255.0

    # Replicate grayscale to 3 channels
    img_3ch = np.stack([img_np] * 3, axis=0)  # (3, H, W)

    # ImageNet normalization (DINOv2/v3 was trained with this)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_3ch = (img_3ch - mean) / std

    tensor = torch.from_numpy(img_3ch).float().unsqueeze(0)
    return tensor.to(device)


# Keep old name as alias for backwards compatibility
preprocess_for_dinov3 = preprocess_for_dino


def extract_attention_maps(
    model,
    img_tensor: torch.Tensor,
    layer: int = -1,
) -> np.ndarray:
    """Extract CLS-to-patch attention from a specified transformer block.

    Works with both HuggingFace DINOv2/v3 models and native 3DINO models.

    Args:
        model: the DINO model
        img_tensor: preprocessed input tensor
        layer: which transformer block to extract from.
               -1 = last block (default), 0 = first block, etc.

    Returns:
        attention maps of shape (num_heads, h_patches, w_patches)
    """
    # --- Native 3DINO model (DinoVisionTransformer / DinoVisionTransformer3d) ---
    if hasattr(model, "blocks") and hasattr(model, "prepare_tokens_with_masks"):
        return _extract_attention_native(model, img_tensor, layer)

    # --- HuggingFace DINOv2/v3 model ---
    return _extract_attention_hf(model, img_tensor, layer)


def _extract_attention_native(model, img_tensor, layer):
    """Extract attention from a native DinoVisionTransformer model."""
    # Flatten all blocks from chunks
    all_blocks = []
    if hasattr(model, "chunked_blocks") and model.chunked_blocks:
        for chunk in model.blocks:
            for blk in chunk:
                if not isinstance(blk, torch.nn.Identity):
                    all_blocks.append(blk)
    else:
        all_blocks = list(model.blocks)

    num_blocks = len(all_blocks)
    target_idx = layer if layer >= 0 else num_blocks + layer

    # Run tokens through blocks, capturing attention at target
    with torch.no_grad():
        x = model.prepare_tokens_with_masks(img_tensor, masks=None)
        attn = None
        for i, blk in enumerate(all_blocks):
            if i == target_idx:
                x, attn = blk(x, return_attn=True)
            else:
                x = blk(x)

    if attn is None:
        raise RuntimeError(f"Could not extract attention from block {target_idx}")

    # attn shape: (B, num_heads, N+1, N+1) where N = num_patches
    cls_attn = attn[0, :, 0, 1:]  # (num_heads, num_patches)
    num_patches = cls_attn.shape[-1]

    # Determine spatial dimensions
    ndim = img_tensor.ndim
    if ndim == 5:  # 3D: (B, C, H, W, D)
        ps = model.patch_size
        h = img_tensor.shape[2] // ps
        w = img_tensor.shape[3] // ps
        d = img_tensor.shape[4] // ps
        attn_maps = cls_attn.reshape(-1, h, w, d)
    else:  # 2D
        h = w = int(num_patches ** 0.5)
        attn_maps = cls_attn.reshape(-1, h, w)

    return attn_maps.cpu().numpy()


def _extract_attention_hf(model, img_tensor, layer):
    """Extract attention from a HuggingFace DINOv2/v3 model using hooks."""
    attn_store = {}

    # Collect all self-attention modules
    attn_modules = []
    for name, mod in model.named_modules():
        if hasattr(mod, "query") and hasattr(mod, "key") and hasattr(mod, "value"):
            attn_modules.append((name, mod))

    if not attn_modules:
        raise RuntimeError("Could not find self-attention layers in the model.")

    # Select target layer
    target_idx = layer if layer >= 0 else len(attn_modules) + layer
    target_idx = max(0, min(target_idx, len(attn_modules) - 1))
    attn_name, attn_mod = attn_modules[target_idx]

    num_heads = (
        attn_mod.attention.num_attention_heads
        if hasattr(attn_mod, "attention")
        else model.config.num_attention_heads
    )

    def hook_fn(module, input, output):
        hidden_states = input[0]  # (B, N, C)
        B, N, C = hidden_states.shape
        q = module.query(hidden_states)
        k = module.key(hidden_states)
        head_dim = C // num_heads
        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_store["attn"] = attn_weights.detach()

    handle = attn_mod.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img_tensor)
    handle.remove()

    if "attn" not in attn_store:
        raise RuntimeError("Attention hook did not fire.")

    last_attn = attn_store["attn"]
    cls_attn = last_attn[0, :, 0, 1:]  # (num_heads, num_patches)
    num_patches = cls_attn.shape[-1]
    h = w = int(num_patches ** 0.5)
    attn_maps = cls_attn.reshape(-1, h, w)

    return attn_maps.cpu().numpy()


def get_num_layers(model) -> int:
    """Return the number of transformer blocks in the model."""
    # Native 3DINO
    if hasattr(model, "blocks") and hasattr(model, "prepare_tokens_with_masks"):
        if hasattr(model, "chunked_blocks") and model.chunked_blocks:
            count = 0
            for chunk in model.blocks:
                for blk in chunk:
                    if not isinstance(blk, torch.nn.Identity):
                        count += 1
            return count
        return len(model.blocks)
    # HuggingFace
    attn_count = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, "query") and hasattr(mod, "key") and hasattr(mod, "value"):
            attn_count += 1
    return attn_count


def create_overlay(
    img_gray: np.ndarray,
    attn_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """Create an overlay of attention map on the original image.
    Returns RGB image as numpy array (H, W, 3) in [0, 255]."""
    h, w = img_gray.shape[:2]

    # Handle 3D attention maps — take a central slice if needed
    if attn_map.ndim == 3:
        mid = attn_map.shape[-1] // 2
        attn_map = attn_map[:, :, mid]

    attn_tensor = torch.from_numpy(attn_map).float().unsqueeze(0).unsqueeze(0)
    attn_upscaled = F.interpolate(attn_tensor, size=(h, w), mode="bilinear", align_corners=False)
    attn_upscaled = attn_upscaled.squeeze().numpy()

    # Normalize to [0, 1]
    attn_min, attn_max = attn_upscaled.min(), attn_upscaled.max()
    attn_upscaled = (attn_upscaled - attn_min) / (attn_max - attn_min + 1e-8)

    # Apply colormap
    if HAS_MATPLOTLIB:
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(attn_upscaled)[:, :, :3]
    else:
        heatmap = np.zeros((h, w, 3), dtype=np.float32)
        heatmap[:, :, 0] = attn_upscaled
        heatmap[:, :, 1] = attn_upscaled * 0.5

    img_rgb = np.stack([img_gray / 255.0] * 3, axis=-1)
    overlay = (1 - alpha) * img_rgb + alpha * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def save_result(overlay: np.ndarray, output_path: str):
    """Save the overlay image."""
    Image.fromarray(overlay).save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DINO Attention Map Overlay for Medical Scans")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--head", type=int, default=None, help="Attention head index (default: mean)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Transformer block index (-1=last, 0=first)")
    parser.add_argument("--slice-idx", type=int, default=None, help="Slice index for 3D volumes")
    parser.add_argument("--window", default="none", choices=list(CT_WINDOW_PRESETS.keys()),
                        help="CT window preset (default: none)")
    parser.add_argument("--ww", type=float, default=None, help="Custom window width")
    parser.add_argument("--wl", type=float, default=None, help="Custom window level")
    parser.add_argument("--colormap", default="jet", help="Matplotlib colormap")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity")
    parser.add_argument("--save-all-heads", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print("Loading DINOv2 model...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model = model.to(device).eval()
    patch_size = getattr(model.config, "patch_size", 14)

    custom_wl = (args.ww, args.wl) if args.ww is not None and args.wl is not None else None
    img_gray = load_medical_image(args.image, args.slice_idx, ct_window=args.window, custom_wl=custom_wl)
    img_tensor = preprocess_for_dino(img_gray, device, patch_size=patch_size)

    attn_maps = extract_attention_maps(model, img_tensor, layer=args.layer)
    num_heads = attn_maps.shape[0]
    print(f"Extracted {num_heads} heads from layer {args.layer}, spatial: {attn_maps.shape[1]}x{attn_maps.shape[2]}")

    if args.head is not None:
        selected_attn = attn_maps[args.head]
    else:
        selected_attn = attn_maps.mean(axis=0)

    output_path = args.output or str(Path(args.image).stem + "_attention.png")
    overlay = create_overlay(img_gray, selected_attn, args.colormap, args.alpha)
    save_result(overlay, output_path)

    if args.save_all_heads:
        out_dir = Path(output_path).parent / f"{Path(args.image).stem}_heads"
        out_dir.mkdir(exist_ok=True)
        for h in range(num_heads):
            head_overlay = create_overlay(img_gray, attn_maps[h], args.colormap, args.alpha)
            save_result(head_overlay, str(out_dir / f"head_{h:02d}.png"))


if __name__ == "__main__":
    main()
