import time

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

from model import FaceAnalyzer
from tracker import FaceTracker
from overlay import draw_overlay, build_stats, tag_primary
from theme import PALETTE, FONTS, SIZES

st.set_page_config(page_title="Face Analysis System", layout="centered")

# Theme variables injected once; the static CSS below references them via var().
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:{PALETTE['bg']}; --text:{PALETTE['text']}; --muted:{PALETTE['muted']};
  --faint:{PALETTE['faint']}; --divider:{PALETTE['divider']};
  --ui:{FONTS['ui']}; --mono:{FONTS['mono']}; --maxw:{SIZES['max_width']}px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp { background:var(--bg); color:var(--text); }
html, body, [class*="css"], .stMarkdown, .stButton, .stCaption { font-family:var(--ui); }
.block-container { max-width:var(--maxw) !important; padding-top:2.2rem !important; padding-bottom:3rem !important; }
#MainMenu, header, footer { visibility:hidden; }
.klabel { color:var(--muted); font-size:0.66rem; letter-spacing:0.30em; text-transform:uppercase;
          text-align:center; margin:0 0 1.6rem 0; }
/* mode switch — borderless uppercase text, no boxes/fills */
.stButton > button { background:transparent !important; border:none !important; border-radius:0 !important;
  box-shadow:none !important; color:var(--muted) !important; font-size:0.8rem !important;
  letter-spacing:0.18em !important; text-transform:uppercase; padding:0.15rem 0 0.4rem 0 !important; }
.stButton > button:hover, .stButton > button:focus { color:var(--text) !important; box-shadow:none !important; }
/* the image/video is the hero */
div[data-testid="stImage"] { display:flex; justify-content:center; }
[data-testid="stImage"] img { border:1px solid var(--divider); }
/* download button — minimal text */
.stDownloadButton > button { background:transparent !important; border:none !important; box-shadow:none !important;
  color:var(--muted) !important; font-size:0.72rem !important; letter-spacing:0.18em !important; text-transform:uppercase; }
.stDownloadButton > button:hover { color:var(--text) !important; }
[data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; letter-spacing:0.04em; }
.disclaimer { color:var(--faint); font-size:0.7rem; letter-spacing:0.05em; text-align:center; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="klabel">Face Analysis</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model once (cached). Clear error if any model file is missing.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return FaceAnalyzer()

try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Place all required model files in the models/ folder and restart.")
    st.stop()

# ---------------------------------------------------------------------------
# Mode switch (default LIVE CAMERA). Not rendering the webrtc component in
# upload mode unmounts it, which stops the camera — satisfying "switching
# modes must stop the camera and clear old results".
# ---------------------------------------------------------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "LIVE CAMERA"

col_u, col_l = st.columns(2)
with col_u:
    if st.button("UPLOAD PHOTO", key="btn_upload", use_container_width=True):
        st.session_state.mode = "UPLOAD PHOTO"
with col_l:
    if st.button("LIVE CAMERA", key="btn_live", use_container_width=True):
        st.session_state.mode = "LIVE CAMERA"

# Active option: off-white with a 1px underline; inactive stays muted grey.
_active = ""
for _key, _name in (("btn_upload", "UPLOAD PHOTO"), ("btn_live", "LIVE CAMERA")):
    if st.session_state.mode == _name:
        _active += (
            "div.st-key-%s button { color:%s !important; "
            "border-bottom:1px solid %s !important; }"
            % (_key, PALETTE["text"], PALETTE["text"])
        )
st.markdown("<style>%s</style>" % _active, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Collapsed settings shared by both modes. Detection threshold applies to both;
# prediction interval is live-only (it drives the tracker's throttling).
# ---------------------------------------------------------------------------
def render_settings(show_interval):
    with st.expander("SETTINGS", expanded=False):
        threshold = st.slider("Detection threshold", 0.10, 0.95, 0.50, 0.05)
        interval = 5
        if show_interval:
            interval = st.slider("Prediction interval (frames)", 1, 15, 5, 1,
                                 help="Higher = faster (fewer age/gender runs), lower = more responsive labels.")
    return threshold, interval


# ---------------------------------------------------------------------------
# UPLOAD PHOTO mode — single image, no tracker/smoothing.
# ---------------------------------------------------------------------------
def render_upload():
    threshold, _ = render_settings(show_interval=False)
    uploaded = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.caption("Upload a photo to begin.")
        return

    # A new file re-runs this cleanly; nothing stale persists.
    img_rgb = np.array(Image.open(uploaded).convert("RGB"))
    faces = model.analyze_image(img_rgb, threshold=threshold)
    tag_primary(faces)                            # accent goes on the largest face

    frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    stats = build_stats(faces, fps=None)          # no FPS in a still image
    annotated = draw_overlay(frame_bgr.copy(), faces, stats)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    if not faces:
        st.caption("NO FACE")
        return

    ok, buf = cv2.imencode(".png", annotated)
    if ok:
        st.download_button(
            "DOWNLOAD ANNOTATED IMAGE", data=buf.tobytes(),
            file_name="face_analysis.png", mime="image/png",
        )


# ---------------------------------------------------------------------------
# LIVE CAMERA mode — streamlit-webrtc video wired to the Phase 2 tracker.
# ---------------------------------------------------------------------------
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.analyzer = load_model()      # cached -> same instance as `model`
        self.tracker = FaceTracker()
        self.threshold = 0.5              # updated from the slider each rerun
        self.predict_interval = 5
        self._t_prev = time.time()
        self._fps = 0.0

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        # threshold passed per-call (never mutates the shared cached analyzer)
        self.tracker.predict_interval = self.predict_interval
        dets = self.analyzer.detect_faces(img_bgr, self.threshold)
        faces = self.tracker.update(dets, img_bgr, self.analyzer)

        now = time.time()
        dt = now - self._t_prev
        self._t_prev = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)   # smoothed FPS

        stats = build_stats(faces, fps=self._fps)
        draw_overlay(img_bgr, faces, stats)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


def render_live():
    threshold, interval = render_settings(show_interval=True)
    st.caption("Real-time. Press START to begin; allow camera access when prompted. Nothing is stored.")
    ctx = webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    # Push the current slider values into the running processor thread.
    if ctx.video_processor is not None:
        ctx.video_processor.threshold = threshold
        ctx.video_processor.predict_interval = interval
    if not ctx.state.playing:
        st.caption("Camera stopped. If it will not start, allow camera access for this site in your browser.")


# ---------------------------------------------------------------------------
if st.session_state.mode == "UPLOAD PHOTO":
    render_upload()
else:
    render_live()

st.markdown(
    '<div class="disclaimer">Pretrained Caffe models. Estimates only, not identity.</div>',
    unsafe_allow_html=True,
)


