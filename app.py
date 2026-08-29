import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AgriLens: Onion Grading AI", page_icon="🧅", layout="centered")

st.title("🧅 AgriLens: AI Multi-Grade Onion Scanner")
st.caption("Smart India Hackathon 2026 | Automated 4-Tier Mandi Grading")

@st.cache_resource
def load_model():
    # Bypass PyTorch 2.4+ safe load restriction for custom Ultralytics weights
    try:
        torch.serialization.add_safe_globals([YOLO])
    except Exception:
        pass
    return YOLO("best.pt")

model = load_model()

grade_colors = {
    "Grade A": (46, 204, 113),   # Emerald Green
    "Grade B": (52, 152, 219),   # Blue
    "Grade C": (243, 156, 18),   # Amber/Orange
    "Grade D (Rot/Reject)": (231, 76, 60) # Red
}

uploaded_file = st.camera_input("Take a photo of onions") or st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    frame = np.array(img)

    h, w = frame.shape[:2]
    scale = min(640 / w, 640 / h)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h_img, w_img = small.shape[:2]

    # Model inference
    results = model.predict(source=small, conf=0.35, iou=0.45, imgsz=320, verbose=False)[0]

    annotated = small.copy()
    counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D (Rot/Reject)": 0}

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w, box_h = x2 - x1, y2 - y1

            # Ignore background detections taking >65% of screen
            if (box_w / w_img) > 0.65 and (box_h / h_img) > 0.65:
                continue

            cls_id = int(box.cls[0])
            raw_name = results.names[cls_id]
            conf = float(box.conf[0])

            if "a" in raw_name.lower() and "d" not in raw_name.lower():
                matched_grade = "Grade A"
            elif "b" in raw_name.lower():
                matched_grade = "Grade B"
            elif "c" in raw_name.lower():
                matched_grade = "Grade C"
            else:
                matched_grade = "Grade D (Rot/Reject)"

            counts[matched_grade] += 1
            color = grade_colors[matched_grade]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{matched_grade} {int(conf*100)}%", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    st.image(annotated, caption="4-Tier Detection Overlay", use_container_width=True)

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
