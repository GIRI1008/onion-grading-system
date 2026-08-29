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
    "Grade C": (243, 156, 18),   # Amber/Orange
    "Grade D (Rot/Reject)": (231, 76, 60) # Red
}

def analyze_onion_patch(crop_img):
    """
    Evaluates real pixel characteristics of each detected onion bulb:
    - Defect/Rot ratio (black/dark mold or extreme discoloration)
    - Uniformity/Texture variance
    - Aspect ratio / Shape deformity
    """
    arr = np.array(crop_img).astype(np.float32)
    h, w, _ = arr.shape
    if h < 5 or w < 5:
        return "Grade B", 0.75

    # Color brightness & channel ratios
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3.0

    # Defect/Rot detection: severely dark spots or unnatural mold discoloration
    dark_pixels = np.sum(brightness < 45)
    total_pixels = h * w
    defect_ratio = dark_pixels / total_pixels

    # Aspect ratio / symmetry factor (spherical shape check)
    aspect_ratio = min(w, h) / max(w, h)

    # Color variance / peel skin health
    color_std = np.std(r) + np.std(g)

    # Multi-tier quality assessment algorithm
    if defect_ratio > 0.12 or brightness.mean() < 60:
        grade = "Grade D (Rot/Reject)"
        conf = min(0.95, 0.65 + defect_ratio)
    elif defect_ratio > 0.04 or aspect_ratio < 0.65 or color_std > 58:
        grade = "Grade C"
        conf = 0.82
    elif aspect_ratio >= 0.80 and color_std < 42 and defect_ratio < 0.015:
        grade = "Grade A"
        conf = 0.91
    else:
        grade = "Grade B"
        conf = 0.85

    return grade, conf

uploaded_file = st.camera_input("Take a photo of onions") or st.file_uploader("Or select a photo from your gallery/camera", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Scale image to max dimension 640px
    w, h = raw_img.size
    scale = min(640 / w, 640 / h)
    small_img = raw_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w_img, h_img = small_img.size

    # Run YOLO detection for bulb locations
    results = model.predict(source=small_img, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]

    draw = ImageDraw.Draw(small_img)
    counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D (Rot/Reject)": 0}

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w, box_h = x2 - x1, y2 - y1

            # Discard false-positive boxes that cover whole screen
            if (box_w / w_img) > 0.90 and (box_h / h_img) > 0.90:
                continue

            # Crop the detected onion region for quality analysis
            crop = small_img.crop((max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)))
            
            # Run visual quality analysis on the cropped bulb
            matched_grade, conf = analyze_onion_patch(crop)

            counts[matched_grade] += 1
            color = grade_colors[matched_grade]

            # Draw bounding box and evaluated grade label
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f" {matched_grade} {int(conf * 100)}% "
            draw.text((x1, max(y1 - 16, 0)), label, fill=color)

    st.image(small_img, caption="4-Tier Mandi Quality Detection Overlay", use_container_width=True)

    total = sum(counts.values())
    if total > 0:
        st.subheader("📊 Mandi Quality Assessment")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Grade A", counts["Grade A"])
        c2.metric("🔵 Grade B", counts["Grade B"])
        c3.metric("🟠 Grade C", counts["Grade C"])
        c4.metric("🔴 Grade D", counts["Grade D (Rot/Reject)"])

        st.markdown(f"""
| Grade | Quality Standard | Count Detected | Recommended Action |
| :--- | :--- | :---: | :--- |
| 🟢 **Grade A** | Prime / Uniform / Export Quality | **{counts['Grade A']}** | High-value cold storage & export |
| 🔵 **Grade B** | Standard / Domestic Market | **{counts['Grade B']}** | Domestic retail market sale |
| 🟠 **Grade C** | Minor Blemishes / Asymmetric | **{counts['Grade C']}** | Rapid local sale or processing |
| 🔴 **Grade D** | Rot / Severe Defects / Reject | **{counts['Grade D (Rot/Reject)']}** | **Discard / Isolate immediately** |
        """)
    else:
        st.warning("⚠️ No onions detected in the frame. Point camera closer to the bulbs.")
