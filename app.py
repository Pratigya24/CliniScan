
import streamlit as st
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO

from utils.gradcam_utils import generate_gradcam, overlay_heatmap
from utils.detect_utils import run_detection

st.set_page_config(page_title="🩻 CliniScan - Lung Abnormality Detection", layout="wide")

# -------------------------------
# 1️⃣ Load class names
# -------------------------------
with open("models/classes.json", "r") as f:
    CLASS_NAMES = json.load(f)

# -------------------------------
# 2️⃣ Load Models (cached so they load only once)
# -------------------------------
@st.cache_resource
def load_classification_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/classification_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_detection_model():
    return YOLO("models/detection_model.pt")

clf_model = load_classification_model()
det_model = load_detection_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# 3️⃣ Streamlit UI
# -------------------------------
st.title("🩻 CliniScan: Lung Abnormality Detection using AI")
st.markdown(
    "Upload a **Chest X-ray** to analyze for abnormalities and visualize "
    "interpretability (Grad-CAM + Bounding Boxes)."
)

uploaded_file = st.file_uploader("Upload Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    col1, col2 = st.columns(2)

    # ---------------- CLASSIFICATION + GRAD-CAM ----------------
    with col1:
        st.subheader("🔍 Classification Results")

        img_tensor = transform(image)
        with torch.no_grad():
            preds = clf_model(img_tensor.unsqueeze(0))
            probs = torch.sigmoid(preds)[0]

        sorted_idx = torch.argsort(probs, descending=True)
        for idx in sorted_idx[:5]:
            st.write(f"**{CLASS_NAMES[idx]}**: {probs[idx]:.2%}")

        st.subheader("🧠 Grad-CAM Heatmap")
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        resized_img = np.array(image.resize((224, 224)))
        overlay = overlay_heatmap(resized_img, heatmap)
        st.image(overlay, caption="Grad-CAM Interpretability", use_column_width=True)

    # ---------------- DETECTION ----------------
    with col2:
