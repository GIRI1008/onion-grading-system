import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="AgriLens: Onion Grading AI", page_icon="🧅", layout="centered")

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
    "Grade B": (52, 152, 219),   # Blue
    "Grade C": (243, 156, 18),   # Amber/Orange
    "Grade D (Rot/Reject)": (231, 76, 60) # Red
}

uploaded_file = st.camera_input("Take a photo of onions") or st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Scale image to max 640px
    w, h = img.size
    scale = min(640 / w, 640 / h)
    small_img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w_img, h_img = small_img.size

    # Run YOLOv8 detection
    results = model.predict(source=small_img, conf=0.35, iou=0.45, imgsz=320, verbose=False)[0]

    draw = ImageDraw.Draw(small_img)
    counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D (Rot/Reject)": 0}

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w, box_h = x2 - x1, y2 - y1

            # Ignore background false positives covering >65% of screen
            if (box_w / w_img) > 0.65 and (box_h / h_img) > 0.65:
                continue

            cls_id = int(box.cls[0])
            raw_name = results.names[cls_id] if results.names else f"Grade {cls_id}"
            conf = float(box.conf[0])

            if "a" in str(raw_name).lower() and "d" not in str(raw_name).lower():
                matched_grade = "Grade A"
            elif "b" in str(raw_name).lower():
                matched_grade = "Grade B"
            elif "c" in str(raw_name).lower():
                matched_grade = "Grade C"
            else:
                matched_grade = "Grade D (Rot/Reject)"

            counts[matched_grade] += 1
            color = grade_colors[matched_grade]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f" {matched_grade} {int(conf * 100)}% "
            draw.text((x1, max(y1 - 16, 0)), label, fill=color)

    st.image(small_img, caption="4-Tier Detection Overlay", use_container_width=True)

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
| 🟢 **Grade A** | Prime / Export Quality | **{counts['Grade A']}** | High-value cold storage & export |
| 🔵 **Grade B** | Domestic / Good Market | **{counts['Grade B']}** | Domestic retail market sale |
| 🟠 **Grade C** | Minor Defect / Fair | **{counts['Grade C']}** | Rapid local sale or processing |
| 🔴 **Grade D** | Rot / Sprouted / Reject | **{counts['Grade D (Rot/Reject)']}** | **Discard / Isolate immediately** |
        """)
    else:
        st.warning("No onions detected in the frame. Point camera closer to the bulbs.")
