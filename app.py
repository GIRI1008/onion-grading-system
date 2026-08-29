import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(
    page_title="AgriLens: Onion Grading AI",
    page_icon="🧅",
    layout="centered",
    initial_sidebar_state="collapsed"
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
    "Grade B": (52, 152, 219),   # Blue
    "Grade C": (243, 156, 18),   # Amber/Orange
    "Grade D (Rot/Reject)": (231, 76, 60) # Red
}

# Sidebar controls to tune sensitivity during live testing
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.80, value=0.20, step=0.05)
    iou_thresh = st.slider("IoU Overlap Threshold", min_value=0.10, max_value=0.70, value=0.40, step=0.05)

tab1, tab2 = st.tabs(["📸 Snap with Back Camera (Upload)", "🤳 Built-in Webcam"])

with tab1:
    st.info("💡 **Recommended for Mobile:** Tap below to open your phone's native **Rear/Back Camera**.")
    uploaded_file = st.file_uploader(
        "Take photo with camera or choose image", 
        type=["jpg", "png", "jpeg"], 
        key="file_upload"
    )

with tab2:
    webcam_file = st.camera_input("Take a photo via browser webcam", key="webcam_input")

target_file = uploaded_file or webcam_file

if target_file is not None:
    raw_img = Image.open(target_file).convert("RGB")
    
    # Resize keeping aspect ratio (max 640px)
    w, h = raw_img.size
    scale = min(640 / w, 640 / h)
    img_resized = raw_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    w_img, h_img = img_resized.size

    # Run YOLOv8 detection with 640px resolution for high precision
    results = model.predict(
        source=img_resized,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=640,
        verbose=False
    )[0]

    draw = ImageDraw.Draw(img_resized)
    counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D (Rot/Reject)": 0}

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w, box_h = x2 - x1, y2 - y1

            # Ignore false-positive boxes covering >90% of entire frame
            if (box_w / w_img) > 0.90 and (box_h / h_img) > 0.90:
                continue

            cls_id = int(box.cls[0])
            raw_name = results.names.get(cls_id, f"Grade {cls_id}") if hasattr(results, "names") else f"Grade {cls_id}"
            conf = float(box.conf[0])
            name_str = str(raw_name).lower().strip()

            # Robust 4-tier class categorization
            if "a" in name_str and "d" not in name_str:
                matched_grade = "Grade A"
            elif "b" in name_str:
                matched_grade = "Grade B"
            elif "c" in name_str:
                matched_grade = "Grade C"
            elif "d" in name_str or "rot" in name_str or "reject" in name_str:
                matched_grade = "Grade D (Rot/Reject)"
            else:
                # Fallback based on class index if names are numeric (0=A, 1=B, 2=C, 3=D)
                mapping = {0: "Grade A", 1: "Grade B", 2: "Grade C", 3: "Grade D (Rot/Reject)"}
                matched_grade = mapping.get(cls_id, "Grade A")

            counts[matched_grade] += 1
            color = grade_colors[matched_grade]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f" {matched_grade} {int(conf * 100)}% "
            draw.text((x1, max(y1 - 16, 0)), label, fill=color)

    st.image(img_resized, caption="4-Tier Detection Overlay", use_container_width=True)

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
        st.warning("⚠️ No onions detected. Try adjusting the 'Confidence Threshold' slider in the left sidebar or move camera closer.")
