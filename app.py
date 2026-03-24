"""
3DINO — Attention Map Viewer for Medical Imaging

Drag & drop a medical scan (PNG, JPG, NIfTI, DICOM), visualize it,
and toggle the DINO attention heatmap overlay on/off.

Features:
  - CT windowing presets (Brain, Bone, Subdural, Stroke, Lung, Soft tissue)
  - Multi-layer attention extraction (select any transformer block)
  - Higher resolution inference (518px for DINOv2 patch_size=14)
  - Native 3DINO model support (load .pth weights for 3D medical ViT)
  - Proper DICOM Hounsfield unit conversion

Launch:
    python app.py                                          # DINOv2-small (default)
    python app.py --model facebook/dinov2-base             # DINOv2-base
    python app.py --weights path/to/3dino.pth              # Native 3DINO weights

Then open http://localhost:7860 in your browser.
"""

import argparse
import os
import sys

import numpy as np
import torch
import gradio as gr

from dinov3_attention_map import (
    load_medical_image,
    preprocess_for_dino,
    extract_attention_maps,
    create_overlay,
    get_num_layers,
    CT_WINDOW_PRESETS,
)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model", default="facebook/dinov2-small",
                     help="HuggingFace model ID (default: facebook/dinov2-small)")
_parser.add_argument("--weights", default=None,
                     help="Path to native 3DINO .pth weights (overrides --model)")
_parser.add_argument("--port", type=int, default=7860)
_parser.add_argument("--high-res", action="store_true", default=True,
                     help="Use high resolution (518px) for DINOv2 (default: True)")
_parser.add_argument("--no-high-res", dest="high_res", action="store_false")
_args, _ = _parser.parse_known_args()

MODEL_ID = _args.model
USE_HIGH_RES = _args.high_res

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

IS_NATIVE_3DINO = False

if _args.weights and os.path.isfile(_args.weights):
    # Load native 3DINO model from .pth weights
    print(f"Loading native 3DINO model from {_args.weights}...")
    from dinov2.configs import load_and_merge_config_3d
    from dinov2.models import build_model_from_cfg
    import dinov2.utils.utils as dinov2_utils

    cfg = load_and_merge_config_3d("train/vit3d_highres")
    MODEL, _embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    try:
        dinov2_utils.load_pretrained_weights(MODEL, _args.weights, "teacher")
    except Exception as e:
        print(f"Warning loading weights: {e}")
        print("Trying direct state_dict load...")
        state_dict = torch.load(_args.weights, map_location="cpu")
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                      for k, v in state_dict.items()}
        MODEL.load_state_dict(state_dict, strict=False)
    MODEL = MODEL.to(DEVICE).eval()
    PATCH_SIZE = MODEL.patch_size
    IS_NATIVE_3DINO = True
    print(f"3DINO model ready on {DEVICE}. (patch_size={PATCH_SIZE})")
else:
    # Load HuggingFace model
    print(f"Loading model: {MODEL_ID} (auto-download on first run)...")
    print(f"Device: {DEVICE}")
    from transformers import AutoModel
    MODEL = AutoModel.from_pretrained(MODEL_ID)
    MODEL = MODEL.to(DEVICE).eval()
    PATCH_SIZE = getattr(MODEL.config, "patch_size", 14)
    print(f"Model ready. (patch_size={PATCH_SIZE})")

NUM_LAYERS = get_num_layers(MODEL)
print(f"Transformer blocks: {NUM_LAYERS}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_nifti_depth(path: str) -> int | None:
    import nibabel as nib
    vol = nib.load(path).get_fdata()
    if vol.ndim >= 3:
        return vol.shape[2]
    return None


def _gray_to_rgb(img_gray: np.ndarray) -> np.ndarray:
    return np.stack([img_gray] * 3, axis=-1)


def _select_attn(attn_maps: np.ndarray, head: int) -> np.ndarray:
    if head < 0 or head >= attn_maps.shape[0]:
        return attn_maps.mean(axis=0)
    return attn_maps[head]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_upload(file, ct_window, custom_ww, custom_wl, layer):
    """User dropped a file -> load image, run model, return overlay."""
    if file is None:
        return None, None, None, None, gr.update(), gr.update()

    path = file.name if hasattr(file, "name") else str(file)

    # Detect NIfTI 3-D
    is_nifti = path.lower().endswith((".nii", ".nii.gz", ".gz"))
    depth = None
    if is_nifti:
        try:
            depth = _get_nifti_depth(path)
        except Exception:
            depth = None

    mid_slice = depth // 2 if depth else None

    # CT windowing
    custom_wl_tuple = None
    if custom_ww and custom_wl and custom_ww > 0:
        custom_wl_tuple = (float(custom_ww), float(custom_wl))
    window_preset = ct_window if ct_window != "auto" else None

    img_gray = load_medical_image(
        path, slice_idx=mid_slice,
        ct_window=window_preset, custom_wl=custom_wl_tuple,
    )

    # Run model
    tensor = preprocess_for_dino(img_gray, DEVICE, patch_size=PATCH_SIZE, high_res=USE_HIGH_RES)
    layer_idx = int(layer) if layer is not None else -1
    attn_maps = extract_attention_maps(MODEL, tensor, layer=layer_idx)
    num_heads = int(attn_maps.shape[0])

    selected = attn_maps.mean(axis=0)
    overlay = create_overlay(img_gray, selected, colormap="jet", alpha=0.45)

    if depth is not None and depth > 1:
        slice_update = gr.update(visible=True, minimum=0, maximum=depth - 1, value=mid_slice)
    else:
        slice_update = gr.update(visible=False, value=0)

    head_update = gr.update(minimum=-1, maximum=num_heads - 1, value=-1)

    return overlay, img_gray, attn_maps, path, slice_update, head_update


def on_slice_change(
    slice_idx, file_path, show_overlay, colormap, alpha, head,
    ct_window, custom_ww, custom_wl, layer,
    _img_gray, _attn_maps,
):
    """User moved the slice slider -> reload slice, re-run model."""
    if file_path is None:
        return None, None, None

    custom_wl_tuple = None
    if custom_ww and custom_wl and custom_ww > 0:
        custom_wl_tuple = (float(custom_ww), float(custom_wl))
    window_preset = ct_window if ct_window != "auto" else None

    img_gray = load_medical_image(
        file_path, slice_idx=int(slice_idx),
        ct_window=window_preset, custom_wl=custom_wl_tuple,
    )
    tensor = preprocess_for_dino(img_gray, DEVICE, patch_size=PATCH_SIZE, high_res=USE_HIGH_RES)
    layer_idx = int(layer) if layer is not None else -1
    attn_maps = extract_attention_maps(MODEL, tensor, layer=layer_idx)

    if show_overlay:
        selected = _select_attn(attn_maps, int(head))
        display = create_overlay(img_gray, selected, colormap=colormap, alpha=alpha)
    else:
        display = _gray_to_rgb(img_gray)

    return display, img_gray, attn_maps


def on_layer_change(
    layer, file_path, slice_idx, show_overlay, colormap, alpha, head,
    ct_window, custom_ww, custom_wl,
    img_gray, _attn_maps,
):
    """User changed the layer -> re-run model on same image."""
    if img_gray is None:
        return None, None

    tensor = preprocess_for_dino(img_gray, DEVICE, patch_size=PATCH_SIZE, high_res=USE_HIGH_RES)
    layer_idx = int(layer) if layer is not None else -1
    attn_maps = extract_attention_maps(MODEL, tensor, layer=layer_idx)

    if show_overlay:
        selected = _select_attn(attn_maps, int(head))
        display = create_overlay(img_gray, selected, colormap=colormap, alpha=alpha)
    else:
        display = _gray_to_rgb(img_gray)

    return display, attn_maps


def on_window_change(
    ct_window, custom_ww, custom_wl, file_path, slice_idx,
    show_overlay, colormap, alpha, head, layer,
    _img_gray, _attn_maps,
):
    """User changed CT window -> reload image with new windowing, re-run model."""
    if file_path is None:
        return None, None, None

    is_nifti = file_path.lower().endswith((".nii", ".nii.gz", ".gz"))
    s_idx = int(slice_idx) if is_nifti else None

    custom_wl_tuple = None
    if custom_ww and custom_wl and custom_ww > 0:
        custom_wl_tuple = (float(custom_ww), float(custom_wl))
    window_preset = ct_window if ct_window != "auto" else None

    img_gray = load_medical_image(
        file_path, slice_idx=s_idx,
        ct_window=window_preset, custom_wl=custom_wl_tuple,
    )
    tensor = preprocess_for_dino(img_gray, DEVICE, patch_size=PATCH_SIZE, high_res=USE_HIGH_RES)
    layer_idx = int(layer) if layer is not None else -1
    attn_maps = extract_attention_maps(MODEL, tensor, layer=layer_idx)

    if show_overlay:
        selected = _select_attn(attn_maps, int(head))
        display = create_overlay(img_gray, selected, colormap=colormap, alpha=alpha)
    else:
        display = _gray_to_rgb(img_gray)

    return display, img_gray, attn_maps


def on_controls_change(show_overlay, colormap, alpha, head, img_gray, attn_maps):
    """Toggle / colormap / alpha / head changed -> cheap re-render, no model call."""
    if img_gray is None or attn_maps is None:
        return None

    if not show_overlay:
        return _gray_to_rgb(img_gray)

    selected = _select_attn(attn_maps, int(head))
    return create_overlay(img_gray, selected, colormap=colormap, alpha=alpha)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
COLORMAPS = ["jet", "hot", "inferno", "viridis", "plasma", "magma", "turbo"]
WINDOW_CHOICES = ["auto", "none"] + [k for k in CT_WINDOW_PRESETS.keys() if k != "none"]

model_label = f"3DINO ({os.path.basename(_args.weights)})" if IS_NATIVE_3DINO else MODEL_ID

with gr.Blocks(
    theme=gr.themes.Monochrome(),
    css="""
        .main-viewer img { image-rendering: pixelated; }
        footer { display: none !important; }
    """,
) as demo:

    # ---- State ----
    state_img_gray = gr.State(None)
    state_attn_maps = gr.State(None)
    state_file_path = gr.State(None)

    # ---- Header ----
    gr.Markdown(
        f"# 3DINO — Attention Map Viewer\n"
        f"Model: `{model_label}` | Device: `{DEVICE}` | "
        f"Blocks: {NUM_LAYERS} | Patch: {PATCH_SIZE}px | "
        f"Resolution: {'518' if USE_HIGH_RES and PATCH_SIZE == 14 else PATCH_SIZE * 16}px\n\n"
        f"Drag & drop a medical scan, configure CT windowing, then explore attention across layers and heads."
    )

    with gr.Row():
        # ---- Left panel: controls ----
        with gr.Column(scale=1, min_width=280):
            file_input = gr.File(
                label="Drop your scan here",
                file_types=[".png", ".jpg", ".jpeg", ".dcm", ".nii", ".nii.gz"],
                type="filepath",
            )

            gr.Markdown("### CT Windowing")
            ct_window_dd = gr.Dropdown(
                choices=WINDOW_CHOICES,
                value="auto",
                label="Window preset",
                info="auto = DICOM header or percentile; brain = W80/L40",
            )
            with gr.Row():
                custom_ww = gr.Number(label="Custom W", value=0, precision=0)
                custom_wl_input = gr.Number(label="Custom L", value=0, precision=0)
            gr.Markdown(
                "*Set Custom W > 0 to override the preset.*\n\n"
                "Presets: Brain W80/L40, Subdural W200/L75, Stroke W40/L40, "
                "Bone W2000/L500, Lung W1500/L-600, Soft W400/L50"
            )

            gr.Markdown("### Attention Controls")
            overlay_toggle = gr.Checkbox(label="Show attention overlay", value=True)
            colormap_dd = gr.Dropdown(choices=COLORMAPS, value="jet", label="Colormap")
            alpha_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Overlay opacity")
            head_slider = gr.Slider(
                -1, 5, value=-1, step=1,
                label="Attention head (-1 = mean)",
            )
            layer_slider = gr.Slider(
                0, max(NUM_LAYERS - 1, 0), value=max(NUM_LAYERS - 1, 0), step=1,
                label=f"Transformer block (0-{NUM_LAYERS - 1})",
                info="Early layers = edges/textures, late layers = semantics",
            )
            slice_slider = gr.Slider(
                0, 100, value=50, step=1,
                label="Slice index (3-D volumes)",
                visible=False,
            )

        # ---- Right panel: image viewer ----
        with gr.Column(scale=3):
            image_output = gr.Image(
                label="Viewer",
                type="numpy",
                height=620,
            )

    # ---- Event wiring ----

    # Upload -> full pipeline
    file_input.change(
        fn=on_upload,
        inputs=[file_input, ct_window_dd, custom_ww, custom_wl_input, layer_slider],
        outputs=[
            image_output,
            state_img_gray,
            state_attn_maps,
            state_file_path,
            slice_slider,
            head_slider,
        ],
    )

    # Slice change -> re-run model
    slice_slider.release(
        fn=on_slice_change,
        inputs=[
            slice_slider, state_file_path,
            overlay_toggle, colormap_dd, alpha_slider, head_slider,
            ct_window_dd, custom_ww, custom_wl_input, layer_slider,
            state_img_gray, state_attn_maps,
        ],
        outputs=[image_output, state_img_gray, state_attn_maps],
    )

    # Layer change -> re-run model on same image
    layer_slider.release(
        fn=on_layer_change,
        inputs=[
            layer_slider, state_file_path, slice_slider,
            overlay_toggle, colormap_dd, alpha_slider, head_slider,
            ct_window_dd, custom_ww, custom_wl_input,
            state_img_gray, state_attn_maps,
        ],
        outputs=[image_output, state_attn_maps],
    )

    # CT window change -> reload image with new windowing
    for ctrl in [ct_window_dd, custom_ww, custom_wl_input]:
        ctrl.change(
            fn=on_window_change,
            inputs=[
                ct_window_dd, custom_ww, custom_wl_input, state_file_path, slice_slider,
                overlay_toggle, colormap_dd, alpha_slider, head_slider, layer_slider,
                state_img_gray, state_attn_maps,
            ],
            outputs=[image_output, state_img_gray, state_attn_maps],
        )

    # Toggle / colormap / alpha / head -> cheap overlay re-render
    cheap_controls = [overlay_toggle, colormap_dd, alpha_slider, head_slider]
    cheap_inputs = cheap_controls + [state_img_gray, state_attn_maps]
    for ctrl in cheap_controls:
        ctrl.change(
            fn=on_controls_change,
            inputs=cheap_inputs,
            outputs=[image_output],
        )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=_args.port,
        share=False,
        show_error=True,
    )
