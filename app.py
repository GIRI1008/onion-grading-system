import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(
    page_title="AgriLens: Onion Grading AI",
    page_icon="🧅",
    layout="centered"
)

st.title("🧅 AgriLens: AI Multi-Grade Onion Scanner")
st.caption("Smart India Hackathon 2026 | Automated 4-Tier Mandi Grading")

MODEL_PATH = "best_model.pt"
GDRIVE_ID = "1Z7FimHPV8BNZPYa8jvBhwz87DF0EENV1"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_trained_model()

grade_colors = {
    "Grade A": (46, 204, 113),   # Emerald Green
    "Grade B": (52, 152, 219),   # Sky Blue
    "Grade C": (243, 156, 18),   # Amber Orange
    "Grade D (Rot/Reject)": (231, 76, 60) # Red
}

with st.sidebar:
    st.header("⚙️ Detection Controls")
    conf_thresh = st.slider("Confidence Filter", min_value=0.20, max_value=0.80, value=0.35, step=0.05)
    st.markdown("""
    - **Higher (0.40+):** Eliminates room background false positives.
    - **Lower (0.25):** Detects onions in darker environments.
    """)

def evaluate_onion_crop(crop_img):
    """
    Evaluates real physical characteristics of the detected bulb:
    - Rot / Dark necrosis spots
    - Spherical symmetry (aspect ratio)
    - Surface color texture uniformity
    """
    arr = np.array(crop_img).astype(np.float32)
    h, w, _ = arr.shape
    if h < 10 or w < 10:
        return "Grade B", 0.80

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3.0

    # Defect / Rot threshold
    dark_pixels = np.sum(brightness < 45)
    total_pixels = h * w
    defect_ratio = dark_pixels / max(total_pixels, 1)

    # Circularity ratio
    aspect_ratio = min(w, h) / max(w, h)

    # Skin color texture variance
    color_std = float(np.std(r) + np.std(g))

    if defect_ratio > 0.14 or brightness.mean() < 50:
        return "Grade D (Rot/Reject)", min(0.95, 0.70 + (defect_ratio * 1.5))
    elif defect_ratio > 0.04 or aspect_ratio < 0.65 or color_std > 60:
        return "Grade C", 0.82
    elif aspect_ratio >= 0.78 and color_std < 48 and defect_ratio < 0.02:
        return "Grade A", 0.92
    else:
        return "Grade B", 0.86

uploaded_file = st.camera_input("Take a photo of onions") or st.file_uploader(
    "Or select a photo from your gallery", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Scale image to max dimension 640px
    w, h = raw_img.size
    scale = min(640 / w, 640 / h)
    small_img = raw_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w_img, h_img = small_img.size

    # Pure YOLO AI Inference
    results = model.predict(source=small_img, conf=conf_thresh, iou=0.45, imgsz=640, verbose=False)[0]

    draw = ImageDraw.Draw(small_img)
    counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D (Rot/Reject)": 0}
    detected_count = 0

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w, box_h = x2 - x1, y2 - y1

            # Filter 1: Ignore bounding boxes taking entire screen background
            if (box_w / w_img) > 0.85 and (box_h / h_img) > 0.85:
                continue

            # Filter 2: Ignore microscopic artifacts under 20x20 pixels
            if box_w < 20 or box_h < 20:
                continue

            detected_count += 1
            crop = small_img.crop((max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)))
            grade, conf = evaluate_onion_crop(crop)

            counts[grade] += 1
            color = grade_colors[grade]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f" {grade} {int(conf * 100)}% "
            draw.text((x1, max(y1 - 16, 2)), label, fill=color)

    st.image(small_img, caption="AI Inspection Result", use_container_width=True)

    if detected_count > 0:
        st.subheader("📊 Mandi Quality Assessment")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Grade A", counts["Grade A"])
        c2.metric("🔵 Grade B", counts["Grade B"])
        c3.metric("🟠 Grade C", counts["Grade C"])
        c4.metric("🔴 Grade D", counts["Grade D (Rot/Reject)"])

        st.markdown(f"""
| Grade | Quality Standard | Count Detected | Recommended Action |
| :--- | :--- | :---: | :--- |
| 🟢 **Grade A** | Prime / Export Quality | **{counts['Grade A']}** | High-value cold storage & export |
| 🔵 **Grade B** | Standard / Domestic Market | **{counts['Grade B']}** | Domestic retail market sale |
| 🟠 **Grade C** | Minor Defect / Processing | **{counts['Grade C']}** | Rapid local sale or processing |
| 🔴 **Grade D** | Rot / Sprouted / Reject | **{counts['Grade D (Rot/Reject)']}** | **Discard / Isolate immediately** |
        """)
    else:
        st.info("ℹ️ No onions detected in the photo. Point camera at actual onion bulbs.")
