import time

import cv2
import streamlit as st

DeepFace = None


DEFAULT_SCORES = {
    "Happy": 92.35,
    "Neutral": 5.20,
    "Surprise": 1.15,
    "Sad": 0.65,
    "Angry": 0.30,
    "Fear": 0.20,
    "Disgust": 0.15,
}

EMOTION_COLORS = {
    "happy": (87, 222, 99),
    "neutral": (226, 188, 25),
    "surprise": (41, 166, 245),
    "sad": (70, 112, 255),
    "angry": (235, 61, 105),
    "fear": (181, 74, 224),
    "disgust": (238, 112, 36),
}

def _init_camera_state():
    defaults = {
        "camera_running": False,
        "current_emotion": "Happy",
        "emotion_confidence": 92.35,
        "emotion_scores": DEFAULT_SCORES.copy(),
        "emotion_history": ["Happy"] * 8 + ["Neutral"] + ["Happy"] * 8,
        "camera_fps": 18.7,
        "snapshot_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def init_camera_state():
    _init_camera_state()


def _normalise_scores(raw_scores):
    scores = {}
    for emotion in DEFAULT_SCORES:
        scores[emotion] = float(raw_scores.get(emotion.lower(), 0))
    return scores


def _draw_detection(frame, faces, emotion):
    color = EMOTION_COLORS.get(emotion.lower(), (87, 222, 99))

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = emotion.title()
        cv2.rectangle(frame, (x, max(0, y - 45)), (x + 220, y), color, -1)
        cv2.putText(
            frame,
            label,
            (x + 12, y - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (7, 10, 21),
            3,
            cv2.LINE_AA,
        )


def _analyse_face(frame, faces):
    if len(faces) == 0:
        return

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    padding = 35
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, frame.shape[1])
    y2 = min(y + h + padding, frame.shape[0])
    face = frame[y1:y2, x1:x2]

    try:
        global DeepFace
        if DeepFace is None:
            from deepface import DeepFace

        result = DeepFace.analyze(
            face,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        emotion_data = result[0] if isinstance(result, list) else result
        emotion = emotion_data["dominant_emotion"].title()
        scores = _normalise_scores(emotion_data["emotion"])
        confidence = max(scores.values()) if scores else 0

        st.session_state.current_emotion = emotion
        st.session_state.emotion_scores = scores
        st.session_state.emotion_confidence = confidence
        st.session_state.emotion_history = (
            st.session_state.emotion_history + [emotion]
        )[-24:]
    except Exception:
        return


def run_live_camera(show_header=True, auto_start=False):
    _init_camera_state()

    if auto_start and not st.session_state.camera_running:
        st.session_state.camera_running = True

    if show_header:
        st.markdown(
            """
            <div class="page-title">
                <div class="title-left">
                    <h1>Live Camera Emotion Detection</h1>
                    <p class="muted">Real-time emotion detection using AI</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    frame_slot = st.empty()

    if not st.session_state.camera_running:
        frame_slot.markdown(
            """
            <div style='
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 3rem;
                text-align: center;
                color: #cbd5e1;
                font-size: 1rem;
            '>
                <strong>Start the camera to see AI emotion detection.</strong><br>
                The live feed will appear here with detected emotion labels.
            </div>
            """,
            unsafe_allow_html=True,
        )

    controls = st.container()
    with controls:
        stop_col, capture_col, fps_col = st.columns([1.1, 1.6, 2])

        with stop_col:
            if st.session_state.camera_running:
                if st.button("Stop Camera", key="stop_camera", type="primary"):
                    st.session_state.camera_running = False
                    st.rerun()
            else:
                if st.button("Start Camera", key="start_camera", type="primary"):
                    st.session_state.camera_running = True
                    st.rerun()

        with capture_col:
            if st.button("Capture Snapshot", key="capture_snapshot"):
                frame = st.session_state.get("last_camera_frame")
                if frame is not None:
                    cv2.imwrite("snapshot.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    st.session_state.snapshot_message = "Snapshot saved as snapshot.jpg"
                else:
                    st.session_state.snapshot_message = "Start the camera before saving a snapshot."

        with fps_col:
            st.markdown(
                f"""
                <div class="fps-readout">
                    FPS: <span>{st.session_state.camera_fps:.1f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.snapshot_message:
        st.caption(st.session_state.snapshot_message)

    if not st.session_state.camera_running:
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.session_state.camera_running = False
        st.error("Camera not detected. Check webcam permissions and try again.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    frame_count = 0
    start_time = time.time()

    while st.session_state.camera_running:
        success, frame = camera.read()
        if not success:
            st.error("Camera frame could not be read.")
            break

        frame_count += 1
        frame = cv2.resize(frame, (900, 560))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if frame_count % 24 == 0:
            _analyse_face(frame, faces)

        _draw_detection(frame, faces, st.session_state.current_emotion)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.last_camera_frame = rgb_frame
        frame_slot.image(rgb_frame, use_container_width=True)

        elapsed = max(time.time() - start_time, 0.001)
        st.session_state.camera_fps = frame_count / elapsed
        time.sleep(0.03)

    camera.release()
