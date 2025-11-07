import os, json, cv2, numpy as np, torch, timm, gradio as gr
from functools import lru_cache
from typing import Tuple
from PIL import Image

# ---- OPTIONAL: if Spaces runs on CPU only, this still works fine ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384

# ---- Albumentations (lightweight import) ----
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Grad-CAM ----
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

TRANSFORM = A.Compose([
    A.LongestMaxSize(IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(),
    ToTensorV2()
])

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
OPERATING_POINT_PATH = os.getenv("OPERATING_POINT_PATH", "operating_point.json")

DISCLAIMER = (
    "⚠️ Educational demo only — NOT for diagnostic or clinical use.\n"
    "Trained on HAM10000; model may be biased or inaccurate for some skin tones/devices."
)

def _load_threshold(default: float = 0.5) -> float:
    if os.path.exists(OPERATING_POINT_PATH):
        try:
            data = json.load(open(OPERATING_POINT_PATH))
            return float(data.get("threshold", default))
        except Exception:
            return default
    return default

@lru_cache(maxsize=1)
def load_model_and_cam():
    # Build the same architecture used in training
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}\n"
            "Upload your trained 'best.pt' to the Space repository."
        )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    # Choose a good conv layer for CAM
    if hasattr(model, "conv_head"):
        target_layers = [model.conv_head]
    else:
        # Fallback for timm variants
        target_layers = [list(model.children())[-2]]

    cam = GradCAM(model=model, target_layers=target_layers)
    thr = _load_threshold(0.5)
    return model, cam, thr

def preprocess(pil_img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    rgb = np.array(pil_img.convert("RGB"))
    t = TRANSFORM(image=rgb)["image"].unsqueeze(0)  # (1,3,H,W)
    return t.to(DEVICE), rgb

def predict(pil_img: Image.Image):
    model, cam, thr = load_model_and_cam()
    x, rgb = preprocess(pil_img)

    with torch.no_grad():
        logit = model(x).squeeze(1).item()
        prob = 1.0 / (1.0 + np.exp(-logit))

    label = "Melanoma" if prob >= thr else "Non-melanoma"

    # Grad-CAM for single-logit head -> index 0
    targets = [BinaryClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # (H_cam, W_cam)

    # Resize image to CAM size BEFORE overlay, then scale back for display
    h, w = grayscale_cam.shape
    rgb_norm = (rgb / 255.0).astype(np.float32)
    rgb_resized = cv2.resize(rgb_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = show_cam_on_image(rgb_resized, grayscale_cam, use_rgb=True)
    # upscale back to original for nicer viewing
    overlay = cv2.resize(overlay, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    return {label: float(f"{prob:.3f}")}, f"{prob:.3f}", f"{thr:.3f}", overlay

with gr.Blocks(title="Skin Lesion Classifier (Educational Demo)") as demo:
    gr.Markdown("# 🩺 Skin Lesion Classifier (Melanoma vs Non-melanoma)")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload dermoscopy image (JPG/PNG)")
    with gr.Row():
        out_label = gr.Label(num_top_classes=1, label="Prediction")
    with gr.Row():
        out_prob = gr.Textbox(label="Probability (melanoma)")
        out_thr  = gr.Textbox(label="Decision threshold")
    with gr.Row():
        out_cam  = gr.Image(type="numpy", label="Grad-CAM overlay")

    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=inp, outputs=[out_label, out_prob, out_thr, out_cam])

    gr.Markdown(
        "### Notes\n"
        "- Model: EfficientNet-B0 (timm). Binary sigmoid head.\n"
        "- Threshold selected on validation to target ~90% specificity.\n"
        "- Source dataset: HAM10000.\n"
    )

if __name__ == "__main__":
    demo.launch()
