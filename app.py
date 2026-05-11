import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

class FaceAgingModel:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.face_net = cv2.dnn.readNetFromCaffe(
            os.path.join(BASE_DIR, "models", "deploy.prototxt"),
            os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        )
        self.age_net = cv2.dnn.readNetFromCaffe(
            os.path.join(BASE_DIR, "models", "age_deploy.prototxt"),
            os.path.join(BASE_DIR, "models", "age_net.caffemodel")
        )
        self.AGE_BUCKETS = [
            "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(20-25)",
            "(25-32)", "(38-43)", "(48-53)", "(60-100)"
        ]

    def detect_faces(self, img):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                faces.append(((startX, startY, endX, endY), confidence))
        return faces

    def predict_age(self, img, face_box):
        (startX, startY, endX, endY) = face_box
        face = img[startY:endY, startX:endX]
        if face.size == 0:
            return "Unknown"
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.768, 114.895))
        self.age_net.setInput(face_blob)
        preds = self.age_net.forward()
        age = self.AGE_BUCKETS[preds[0].argmax()]
        return age

    def predict(self, img):
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces_data = self.detect_faces(img_bgr)
        results = []
        output_img = img_bgr.copy()
        for (face_box, confidence) in faces_data:
            age = self.predict_age(img_bgr, face_box)
            results.append({"age": age, "confidence": confidence})
            (startX, startY, endX, endY) = face_box
            cv2.rectangle(output_img, (startX, startY), (endX, endY), (74, 158, 255), 2)
            cv2.putText(output_img, age, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (74, 158, 255), 2)
        output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        return output_rgb, results


st.set_page_config(page_title="Face Analysis System", layout="centered")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
.stApp { background-color: #111111; color: #ffffff; }
.block-container { max-width: 800px !important; padding-top: 3rem !important; }
* { font-family: 'Inter', sans-serif !important; }
#MainMenu, header, footer { visibility: hidden; }
.title { color: #ffffff; font-size: 2.2rem; font-weight: 700; text-align: center; }
.subtitle { color: #888888; font-size: 0.9rem; text-align: center; margin-bottom: 1.5rem; }
.divider { height: 1px; background-color: #2a2a2a; margin-bottom: 2rem; }
.result-card { background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px;
               padding: 24px; margin-top: 1.5rem; text-align: center; }
.result-label { color: #888888; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05rem; }
.result-value { color: #4A9EFF; font-size: 2.5rem; font-weight: 700; }
.result-conf { color: #555555; font-size: 0.8rem; margin-top: 4px; }
.no-face { color: #555555; font-style: italic; }
.footer-text { color: #333333; font-size: 0.75rem; text-align: center; margin-top: 4rem; }
[data-testid="stCameraInput"] video { border-radius: 8px !important; border: 1px solid #2a2a2a !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Face Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time age detection using deep learning</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return FaceAgingModel()

model = load_model()

camera_image = st.camera_input("", label_visibility="collapsed")

if camera_image is not None:
    img = Image.open(camera_image)
    img_np = np.array(img)
    processed_img, results = model.predict(img_np)
    st.image(processed_img, use_column_width=True)
    if results:
        res = results[0]
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Detected Age</div>
                <div class="result-value">{res['age']}</div>
                <div class="result-conf">Confidence: {res['confidence']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-card">
                <div class="no-face">No face detected</div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="result-card">
            <div class="no-face">Awaiting camera capture...</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer-text">Built by Akshat Dange · JSPM RSCOE · 2026</div>', unsafe_allow_html=True)