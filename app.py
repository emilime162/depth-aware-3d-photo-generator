# ===========================================================
# app.py — Depth-Aware 3D Photo Generator (Final Submission)
# Author: Emily Chen
# ===========================================================
# Implements all required parts:
# Part 1 – Depth Estimation
# Part 2 – Depth-Guided Layer Separation
# Part 3 – Intelligent Background Reconstruction
# Part 4 – Depth-Aware Motion Synthesis
# Part 5 – Depth-of-Field & Bokeh Effects
# Part 6 – Interactive Deployment (Gradio UI)
#
# Engineering Challenges (2 pts):
# ✅ 1. Dynamic Zoom Effect — via cv2.getRotationMatrix2D()
# ✅ 3. Custom Bokeh Shapes — hexagonal kernel blur
# ✅ 4. Customization — adjustable parallax, aperture, zoom, frame count, bokeh type
# ===========================================================

import gradio as gr
import numpy as np
import cv2
import torch
from PIL import Image
import imageio.v3 as iio
from transformers import DPTImageProcessor, DPTForDepthEstimation

# -----------------------------------------------------------
# Part 1 – Depth Estimation [2.5 pts]
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
model.eval()

def estimate_depth(image_pil):
    """Return normalized depth map [0, 1]."""
    max_size = 640
    if max(image_pil.size) > max_size:
        ratio = max_size / max(image_pil.size)
        new_size = tuple(int(dim * ratio) for dim in image_pil.size)
        image_pil = image_pil.resize(new_size, Image.LANCZOS)

    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    size = (int(image_pil.size[1]), int(image_pil.size[0]))
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=size,
        mode="bicubic",
        align_corners=False,
    )
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return depth


# -----------------------------------------------------------
# Helpers (used across parts)
# -----------------------------------------------------------
def smoothstep(t): return t * t * (3 - 2 * t)

# Engineering Challenge #3 – Custom Bokeh Shape (Hex)
def hex_bokeh_kernel(size=25):
    """Create hexagonal convolution kernel for custom bokeh."""
    mask = np.zeros((size, size), np.uint8)
    pts = np.array([
        [size//2, 0],
        [size-1, size//3],
        [size-1, 2*size//3],
        [size//2, size-1],
        [0, 2*size//3],
        [0, size//3]
    ])
    cv2.fillPoly(mask, [pts], 1)
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel


# -----------------------------------------------------------
# Part 5 – Depth-of-Field & Bokeh Effects [4.5 pts]
# (with Challenge #3 integrated)
# -----------------------------------------------------------
def blur_by_depth(image, depth_map, aperture='f2.8', bokeh_shape='gaussian'):
    """Depth-aware blur with adjustable aperture and bokeh shape."""
    # 🔧 Fix: resize depth_map to match image
    if depth_map.shape[:2] != image.shape[:2]:
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    fnum_to_kernel = {'f1.4': (31,31), 'f2.8': (17,17), 'f5.6': (7,7)}
    ksize = fnum_to_kernel.get(aperture, (17,17))

    if bokeh_shape == 'hex':
        kernel = hex_bokeh_kernel(max(ksize))
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = cv2.GaussianBlur(image, ksize, 0)

    far_mask = 1 - depth_map
    far_mask = np.clip(far_mask, 0, 1)
    far_mask = cv2.GaussianBlur(far_mask, (21,21), 0)
    far_mask = np.stack([far_mask]*3, axis=2)

    return (blurred * far_mask + image * (1 - far_mask)).astype(np.uint8)


# -----------------------------------------------------------
# Part 3 – Background Reconstruction & Part 4 – Parallax Motion
# (Simplified to foreground + background only)
# -----------------------------------------------------------

def create_parallax_gif(image_pil, fg_strength=12, bg_strength=4,
                        aperture='f2.8', num_frames=20,
                        zoom_max=1.10, bokeh_shape='gaussian',
                        progress=gr.Progress(track_tqdm=True)):

    progress(0, desc="Estimating depth…")
    depth_map = estimate_depth(image_pil)
    img_rgb = np.array(image_pil.convert("RGB"), dtype=np.uint8)

    # Part 2 – Foreground mask (Otsu)
    depth_u8 = (depth_map * 255).astype(np.uint8)
    _, mask = cv2.threshold(depth_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # 🔧 ensure mask same size as image
    if mask.shape[:2] != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask.astype(bool)

    # Part 3 – Inpainting background (Telea)
    progress(1, desc="Reconstruc ting background…")
    bg = cv2.inpaint(img_rgb, mask, 3, cv2.INPAINT_TELEA)

    bg_blurred = blur_by_depth(bg, depth_map, aperture, bokeh_shape)

    # Engineering Challenge #1 – Dynamic Zoom
    def apply_zoom(image, zoom_factor=1.05):
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    # Part 4 – Parallax Motion
    progress(2, desc="Generating animation frames…")
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        e = smoothstep(t)
        #print(f"Frame {i+1}/{num_frames}, e={e:.3f}")
        zoom = zoom_max - 0.05 * e
        
        fg_img = img_rgb * mask_bool[...,None]
        bg_img = bg_blurred

        fg_img_zoomed = apply_zoom(fg_img, zoom)
        bg_img_zoomed = apply_zoom(bg_img, zoom)
        mask_zoomed = apply_zoom(mask_bool.astype(np.uint8)*255, zoom) > 128


        fg_shift = int(round((2*e - 1) * fg_strength))
        bg_shift = int(round((1 - 2*e) * bg_strength))

        fg_shifted = np.roll(fg_img_zoomed, fg_shift, axis=1)
        bg_shifted = np.roll(bg_img_zoomed, bg_shift, axis=1)
        mask_shifted = np.roll(mask_zoomed, fg_shift, axis=1)

        out = bg_shifted.copy()
        out[mask_shifted] = fg_shifted[mask_shifted]
        frames.append(out.astype(np.uint8))



    progress(3, desc="Saving animation…")
    gif_path = "spatial_photo.gif"
    iio.imwrite(gif_path, frames, duration=0.06, loop=0)
    return gif_path

# -----------------------------------------------------------
# Part 6 – Interactive Deployment [4 pts]
# -----------------------------------------------------------
def generate(image, parallax_strength, aperture, num_frames, zoom_max, bokeh_shape):
    fg_strength = parallax_strength
    bg_strength = max(1, parallax_strength // 3)
    # resize image
    image = image.resize((640, int(640 * image.height / image.width)), Image.LANCZOS)
    gif_path = create_parallax_gif(image, fg_strength, bg_strength,
                                   aperture, num_frames, zoom_max,
                                   bokeh_shape=bokeh_shape)
    return gif_path, gif_path


print("gradio version: ", gr.__version__)


with gr.Blocks(title="Depth-Aware 3D Photo") as demo:
    gr.Markdown("""
    ## 🎥 Depth-Aware 3D Photo Generator
    Upload a photo to generate a cinematic 3D parallax GIF with dynamic zoom and custom bokeh.
    """)

    with gr.Row():
        input_img = gr.Image(label="Upload Photo", type="pil")
        output_gif = gr.Image(label="Animated GIF Preview")

    with gr.Row():
        parallax_slider = gr.Slider(4, 20, value=10, step=1, label="Parallax Strength")
        aperture_dropdown = gr.Dropdown(choices=["f1.4","f2.8","f5.6"], value="f2.8", label="Aperture (Depth of Field)")
        frame_slider = gr.Slider(10, 40, value=20, step=2, label="Frame Count")
        zoom_slider = gr.Slider(1.05, 1.20, value=1.10, step=0.01, label="Max Zoom")
        bokeh_dropdown = gr.Dropdown(choices=["gaussian", "hex"], value="gaussian", label="Bokeh Shape")

    run_btn = gr.Button("Generate Animation")
    download = gr.File(label="Download GIF")

    gr.Examples(
        examples=[
            "sample.jpg",
        ],
        inputs=input_img,
        label="Click a sample to try!"
    )
    run_btn.click(
        fn=generate,
        inputs=[input_img, parallax_slider, aperture_dropdown, frame_slider, zoom_slider, bokeh_dropdown],
        outputs=[output_gif, download]
    )

demo.launch()

# if __name__ == '__main__':


#     generate(
#         image=Image.open("/Users/emilychen/Desktop/neu/5330 Computer Vision/Assignments/HW4/sample_img.jpeg"),
#         parallax_strength=12,
#         aperture='f2.8',
#         num_frames=5,
#         zoom_max=1.10,
#         bokeh_shape='hex',
#     )
