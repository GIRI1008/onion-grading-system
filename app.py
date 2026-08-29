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

# Palette for any detected class
color_palette = [
    (46, 204, 113),  # Emerald Green
    (52, 152, 219),  # Blue
    (243, 156, 18),  # Amber / Orange
    (231, 76, 60),   # Red
    (155, 89, 182),  # Purple
    (26, 188, 156)   # Teal
]

# Simple single-screen inputs
uploaded_file = st.camera_input("Take a photo of onions") or st.file_uploader("Or select a photo from your gallery/camera", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Scale image to max dimension 640px
    w, h = raw_img.size
    scale = min(640 / w, 640 / h)
    small_img = raw_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w_img, h_img = small_img.size

    # YOLO inference
    results = model.predict(source=small_img, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]

    draw = ImageDraw.Draw(small_img)
    counts = {}

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w, box_h = x2 - x1, y2 - y1

            # Discard whole-image background false positives
            if (box_w / w_img) > 0.90 and (box_h / h_img) > 0.90:
                continue

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Get the exact label name assigned during training
            if hasattr(results, "names") and cls_id in results.names:
                label_name = str(results.names[cls_id]).strip()
            else:
                label_name = f"Class {cls_id}"

            counts[label_name] = counts.get(label_name, 0) + 1
            color = color_palette[cls_id % len(color_palette)]

            # Draw box & label
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f" {label_name} {int(conf * 100)}% "
            draw.text((x1, max(y1 - 16, 0)), label, fill=color)

    st.image(small_img, caption="Detection Overlay", use_container_width=True)

    total = sum(counts.values())
    if total > 0:
        st.subheader("📊 Mandi Quality Assessment")
        
        # Display dynamic metric columns for whatever classes are detected
        cols = st.columns(min(len(counts), 4))
        for idx, (grade, cnt) in enumerate(counts.items()):
            cols[idx % len(cols)].metric(f"🏷️ {grade}", cnt)

        # Dynamic breakdown table
        table_md = "| Detected Grade / Class | Count | Status |\n| :--- | :---: | :--- |\n"
        for grade, cnt in counts.items():
            table_md += f"| **{grade}** | **{cnt}** | Assessed |\n"
        st.markdown(table_md)
    else:
        st.warning("⚠️ No onions detected in the frame. Point camera closer with good lighting.")
