import streamlit as st
import numpy as np, cv2, torch, timm, json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_page_config(page_title="Skin Cancer Detector", page_icon="🩺", layout="centered")
st.title("🩺 Skin Lesion Classifier (Melanoma vs Non-melanoma)")
st.caption("Educational demo only — NOT for diagnosis.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384

TF = A.Compose([
    A.LongestMaxSize(IMG_SIZE),
    A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(),
    ToTensorV2()
])

@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    state = torch.load("best.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(DEVICE); model.eval()
    return model

@st.cache_resource
def load_threshold():
    try:
        with open("operating_point.json") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return 0.5

def preprocess(pil_image: Image.Image):
    rgb = np.array(pil_image.convert("RGB"))
    x = TF(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    return x, rgb

def predict_one(pil_image: Image.Image):
    model = load_model(); thr = load_threshold()
    x, rgb = preprocess(pil_image)
    with torch.no_grad():
        logit = model(x).squeeze(1).item()
        prob = 1/(1+np.exp(-logit))
    label = "Melanoma" if prob >= thr else "Non-melanoma"
    return label, prob, thr, rgb

uploaded = st.file_uploader("Upload a dermoscopy image (JPG/PNG)", type=["jpg","jpeg","png"])
col1, col2 = st.columns(2)

if uploaded:
    pil = Image.open(uploaded)
    col1.image(pil, caption="Input image", use_container_width=True)

    if st.button("Predict"):
        label, prob, thr, rgb = predict_one(pil)
        st.success(f"Prediction: **{label}**")
        st.write(f"Probability (melanoma): **{prob:.3f}**  |  Threshold: **{thr:.3f}**")

        # Optional Grad-CAM; remove block if it errors on free hosting
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image

            model = load_model()
            target_layers = [model.conv_head] if hasattr(model,"conv_head") else [list(model.children())[-2]]
            cam = GradCAM(model=model, target_layers=target_layers)

            x, _ = preprocess(pil)
            grayscale_cam = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(0)])[0]
            h,w = grayscale_cam.shape
            rgb_norm = (rgb/255.0).astype(np.float32)
            rgb_resized = cv2.resize(rgb_norm, (w,h), interpolation=cv2.INTER_LINEAR)
            overlay = show_cam_on_image(rgb_resized, grayscale_cam, use_rgb=True)
            overlay = cv2.resize(overlay, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            col2.image(overlay, caption="Grad-CAM heatmap", use_container_width=True)
        except Exception as e:
            st.info("Grad-CAM not available on this deployment.")
            st.caption(f"(Reason: {e})")

st.divider()
st.markdown("**Model:** EfficientNet-B0 (timm). Trained on HAM10000. *Educational demo only.*")
