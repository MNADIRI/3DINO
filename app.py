"""
DINOv3 Attention Map Viewer — Local Web Interface

Drag & drop a medical scan (PNG, JPG, NIfTI, DICOM), visualize it,
and toggle the DINOv3 attention heatmap overlay on/off.

Launch:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import numpy as np
import torch
import gradio as gr

from dinov3_attention_map import (
    load_medical_image,
    preprocess_for_dinov3,
    extract_attention_maps,
    create_overlay,
)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
print("Loading DINOv3 model (auto-download ~85 MB on first run)...")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

from transformers import AutoModel  # noqa: E402

MODEL = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
MODEL = MODEL.to(DEVICE).eval()
print("Model ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_nifti_depth(path: str) -> int | None:
    """Return the number of slices if the file is a 3-D NIfTI volume."""
    import nibabel as nib

    vol = nib.load(path).get_fdata()
    if vol.ndim >= 3:
        return vol.shape[2]
    return None


def _gray_to_rgb(img_gray: np.ndarray) -> np.ndarray:
    """Convert (H, W) uint8 grayscale to (H, W, 3) RGB."""
    return np.stack([img_gray] * 3, axis=-1)


def _select_attn(attn_maps: np.ndarray, head: int) -> np.ndarray:
    """Pick a single head or mean of all heads. head == -1 → mean."""
    if head < 0 or head >= attn_maps.shape[0]:
        return attn_maps.mean(axis=0)
    return attn_maps[head]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_upload(file):
    """User dropped a file → load image, run model, return overlay."""
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
    img_gray = load_medical_image(path, slice_idx=mid_slice)

    # Run model
    tensor = preprocess_for_dinov3(img_gray, DEVICE)
    attn_maps = extract_attention_maps(MODEL, tensor)
    num_heads = int(attn_maps.shape[0])

    # Build default overlay (mean of heads)
    selected = attn_maps.mean(axis=0)
    overlay = create_overlay(img_gray, selected, colormap="jet", alpha=0.45)

    # Slice slider: visible only for 3-D volumes
    if depth is not None and depth > 1:
        slice_update = gr.update(visible=True, minimum=0, maximum=depth - 1, value=mid_slice)
    else:
        slice_update = gr.update(visible=False, value=0)

    # Head slider max
    head_update = gr.update(minimum=-1, maximum=num_heads - 1, value=-1)

    return overlay, img_gray, attn_maps, path, slice_update, head_update


def on_slice_change(
    slice_idx, file_path, show_overlay, colormap, alpha, head,
    _img_gray, _attn_maps,
):
    """User moved the slice slider → reload slice, re-run model."""
    if file_path is None:
        return None, None, None

    img_gray = load_medical_image(file_path, slice_idx=int(slice_idx))
    tensor = preprocess_for_dinov3(img_gray, DEVICE)
    attn_maps = extract_attention_maps(MODEL, tensor)

    if show_overlay:
        selected = _select_attn(attn_maps, int(head))
        display = create_overlay(img_gray, selected, colormap=colormap, alpha=alpha)
    else:
        display = _gray_to_rgb(img_gray)

    return display, img_gray, attn_maps


def on_controls_change(show_overlay, colormap, alpha, head, img_gray, attn_maps):
    """Toggle / colormap / alpha / head changed → cheap re-render, no model call."""
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

with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="DINOv3 Attention Viewer",
    css="""
        .main-viewer img { image-rendering: pixelated; }
        footer { display: none !important; }
    """,
) as demo:

    # ---- State (invisible, persists across callbacks) ----
    state_img_gray = gr.State(None)
    state_attn_maps = gr.State(None)
    state_file_path = gr.State(None)

    # ---- Header ----
    gr.Markdown(
        "# DINOv3 — Attention Map Viewer\n"
        "Drag & drop a medical scan, then toggle the attention overlay."
    )

    with gr.Row():
        # ---- Left panel: controls ----
        with gr.Column(scale=1, min_width=260):
            file_input = gr.File(
                label="Drop your scan here",
                file_types=[".png", ".jpg", ".jpeg", ".dcm", ".nii", ".nii.gz"],
                type="filepath",
            )

            overlay_toggle = gr.Checkbox(label="Show attention overlay", value=True)

            colormap_dd = gr.Dropdown(
                choices=COLORMAPS, value="jet", label="Colormap",
            )
            alpha_slider = gr.Slider(
                0.0, 1.0, value=0.45, step=0.05, label="Overlay opacity",
            )
            head_slider = gr.Slider(
                -1, 5, value=-1, step=1,
                label="Attention head (−1 = mean)",
            )
            slice_slider = gr.Slider(
                0, 100, value=50, step=1,
                label="Slice index (3-D volumes)",
                visible=False,
            )

            gr.Markdown(
                "---\n"
                "**Formats** : PNG · JPG · NIfTI (.nii.gz) · DICOM (.dcm)\n\n"
                "**Model** : `dinov3-vits16` — auto-downloaded from HuggingFace\n\n"
                f"**Device** : `{DEVICE}`"
            )

        # ---- Right panel: image viewer ----
        with gr.Column(scale=3):
            image_output = gr.Image(
                label="Viewer",
                type="numpy",
                height=620,
                elem_classes=["main-viewer"],
                show_download_button=True,
            )

    # ---- Event wiring ----

    # Upload → full pipeline
    file_input.change(
        fn=on_upload,
        inputs=[file_input],
        outputs=[
            image_output,
            state_img_gray,
            state_attn_maps,
            state_file_path,
            slice_slider,
            head_slider,
        ],
    )

    # Slice change → re-run model (new image content)
    slice_slider.release(
        fn=on_slice_change,
        inputs=[
            slice_slider, state_file_path,
            overlay_toggle, colormap_dd, alpha_slider, head_slider,
            state_img_gray, state_attn_maps,
        ],
        outputs=[image_output, state_img_gray, state_attn_maps],
    )

    # Toggle / colormap / alpha / head → cheap overlay re-render
    controls = [overlay_toggle, colormap_dd, alpha_slider, head_slider]
    control_inputs = controls + [state_img_gray, state_attn_maps]

    for ctrl in controls:
        ctrl.change(
            fn=on_controls_change,
            inputs=control_inputs,
            outputs=[image_output],
        )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
