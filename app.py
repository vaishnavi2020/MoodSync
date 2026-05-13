import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import altair as alt
import pandas as pd
import streamlit as st

from emotion_detector import detect_emotion
from fusion import detect_mismatch
import live_camera
from music_recommender import get_recommendations
from sentiment_detector import detect_sentiment
from summary_generator import generate_summary
from audio_analyzer import analyze_audio


st.set_page_config(page_title="MoodSyncAI Dashboard", layout="wide", initial_sidebar_state="expanded")


DEFAULT_SCORES = {
    "Happy": 92.35,
    "Neutral": 5.20,
    "Surprise": 1.15,
    "Sad": 0.65,
    "Angry": 0.30,
    "Fear": 0.20,
    "Disgust": 0.15,
}


def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #030712;
    --panel: rgba(8, 13, 29, 0.88);
    --panel-strong: rgba(10, 16, 34, 0.96);
    --line: rgba(148, 163, 184, 0.14);
    --muted: #a7afc0;
    --text: #f8fafc;
    --pink: #f03bd2;
    --purple: #7c3aed;
    --green: #67e56f;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 30% 10%, rgba(240, 59, 210, 0.10), transparent 28rem),
        radial-gradient(circle at 86% 20%, rgba(103, 229, 111, 0.07), transparent 24rem),
        #020511;
    color: var(--text);
    font-family: Inter, system-ui, sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    padding: 2rem 1.5rem 2rem 1.2rem;
    max-width: 1680px;
}

section[data-testid="stSidebar"] {
    width: 300px !important;
    background: linear-gradient(180deg, rgba(6, 10, 26, 0.98), rgba(3, 7, 18, 0.98));
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.25rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.25rem 0 2rem;
    font-size: 1.7rem;
    font-weight: 800;
}

.brand-mark {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--purple), var(--pink));
    color: white;
    font-size: 1.45rem;
}

.brand span:last-child,
.gradient-text {
    background: linear-gradient(90deg, #f447d6, #ffffff 58%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.profile-card, .status-card, .quote-card, .glass-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.24);
}

.profile-card {
    display: flex;
    gap: 0.9rem;
    align-items: center;
    padding: 1rem;
    margin-bottom: 1.8rem;
}

.avatar {
    display: grid;
    place-items: center;
    width: 54px;
    height: 54px;
    border-radius: 50%;
    background: linear-gradient(135deg, #8b5cf6, #4c1d95);
    font-size: 1.7rem;
}

.profile-name {
    font-weight: 700;
    color: white;
}

.profile-role, .muted {
    color: var(--muted);
}

.nav-title {
    display: none;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.45rem;
}

section[data-testid="stSidebar"] label[data-baseweb="radio"] {
    min-height: 54px;
    padding: 0 1rem;
    border-radius: 10px;
    color: #d7dce8;
}

section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(90deg, rgba(240, 59, 210, 0.85), rgba(124, 58, 237, 0.88));
    color: white;
}

section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

.status-card {
    padding: 1rem;
    margin-top: 1.4rem;
}

.status-row {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: center;
    margin-top: 0.8rem;
    font-size: 0.9rem;
}

.pill {
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
    background: rgba(34, 197, 94, 0.16);
    color: #73e77c;
    font-size: 0.78rem;
    font-weight: 700;
}

.quote-card {
    padding: 1.2rem;
    margin-top: 1.25rem;
    background: linear-gradient(145deg, rgba(55, 26, 118, 0.46), rgba(14, 17, 40, 0.96));
}

.quote-mark {
    color: var(--pink);
    font-size: 2.4rem;
    line-height: 1;
    font-weight: 800;
}

.quote-card p {
    color: #ff4fda;
    font-weight: 600;
    line-height: 1.65;
}

.page-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.7rem 0 2rem;
}

.title-left h1 {
    font-size: clamp(2rem, 3vw, 2.7rem);
    line-height: 1.05;
    margin: 0;
    color: white;
}

.title-left p {
    margin: 0.65rem 0 0;
    color: var(--muted);
    font-size: 1.05rem;
}

.page-status {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    color: #a7afc0;
    font-size: 0.95rem;
}

.live-dot {
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 16px rgba(52, 211, 153, 0.35);
}

.glass-card {
    padding: 1.35rem;
    margin-bottom: 1.2rem;
}

.camera-card {
    padding: 1.3rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    margin-bottom: 1.25rem;
}

.camera-card .stButton {
    margin-top: 1rem;
}

.stButton > button {
    width: 100%;
    min-height: 45px;
    border: 0;
    border-radius: 8px;
    color: white;
    font-weight: 700;
    background: rgba(25, 31, 49, 0.92);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #ef4444, #f0526b);
}

.panel-title {
    margin: 0 0 1rem;
    font-size: 1.1rem;
    font-weight: 800;
    color: white;
}

.divider {
    height: 1px;
    background: var(--line);
    margin: 0.75rem 0 1.7rem;
}

.emotion-display {
    display: grid;
    grid-template-columns: 130px 1fr;
    gap: 1.35rem;
    align-items: center;
}

.face-icon {
    width: 112px;
    height: 112px;
    border-radius: 50%;
    border: 7px solid var(--green);
    color: var(--green);
    display: grid;
    place-items: center;
    font-size: 4.2rem;
    font-weight: 700;
}

.emotion-name {
    color: var(--green);
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1;
}

.confidence-pill {
    display: inline-block;
    margin-top: 0.85rem;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.16);
    color: #b9f7c0;
    font-weight: 600;
}

.prob-row {
    display: grid;
    grid-template-columns: 76px 1fr 62px;
    gap: 0.75rem;
    align-items: center;
    margin: 0.9rem 0;
    color: #dce2ee;
    font-size: 0.92rem;
}

.track {
    height: 14px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.10);
    overflow: hidden;
}

.bar {
    height: 100%;
    border-radius: 999px;
}

.instructions {
    margin: 0;
    padding-left: 1.15rem;
    color: #dce2ee;
    line-height: 2.05;
}

.song-card {
    padding: 0.85rem 0;
    border-top: 1px solid var(--line);
}

.song-card:first-of-type {
    border-top: 0;
    padding-top: 0;
}

.song-title {
    color: white;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.song-meta {
    color: var(--muted);
    font-size: 0.88rem;
    margin-bottom: 0.55rem;
}

.feature-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
}

.feature-chip {
    border-radius: 999px;
    padding: 0.24rem 0.55rem;
    background: rgba(240, 59, 210, 0.12);
    color: #f9a8e9;
    font-size: 0.74rem;
    font-weight: 700;
}

.spotify-link {
    display: inline-block;
    margin-top: 0.75rem;
    color: #a5f3fc;
    font-size: 0.92rem;
    font-weight: 700;
    text-decoration: none;
    border: 1px solid rgba(165, 243, 252, 0.18);
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    transition: all 0.15s ease-in-out;
}

.spotify-link:hover {
    background: rgba(165, 243, 252, 0.08);
    border-color: rgba(165, 243, 252, 0.35);
    color: #7dd3fc;
}

.upload-preview-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 1rem;
    display: grid;
    gap: 0.9rem;
}

.upload-preview-card img {
    width: 100%;
    border-radius: 16px;
    object-fit: cover;
}

.upload-meta {
    color: var(--muted);
    font-size: 0.92rem;
}


.upload-input-card {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 1.4rem;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.95rem;
}

.upload-input-card strong {
    display: block;
    margin-bottom: 0.8rem;
    color: white;
}

.upload-placeholder {
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.75;
}

.analysis-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.result-card {
    padding: 1.55rem;
    min-height: 290px;
    border-radius: 18px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
}

.result-card h3 {
    margin-top: 0;
    font-size: 1.05rem;
    color: #f8fafc;
}

.result-note {
    color: var(--muted);
    margin-top: 0.85rem;
    font-size: 0.95rem;
    line-height: 1.7;
}

.summary-card {
    padding: 1.45rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148,163,184,0.12);
    margin-top: 1rem;
}

.summary-card p {
    margin: 0;
    line-height: 1.8;
    color: #dce2ee;
}

.summary-card strong {
    color: #f3a5f7;
}

.upload-details {
    color: #a7afc0;
    font-size: 0.92rem;
    margin-top: 0.85rem;
}

.upload-details span {
    display: inline-block;
    margin-right: 1.1rem;
}

.analysis-metric {
    color: white;
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0.4rem 0 0.5rem;
}

.analysis-label {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.75rem;
    border-radius: 999px;
    background: rgba(103, 229, 111, 0.12);
    color: #a7f3d0;
    font-size: 0.84rem;
    margin-right: 0.55rem;
    margin-bottom: 0.85rem;
}

.row-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    align-items: start;
    margin-top: 1rem;
}

.row-grid .panel-title {
    margin-bottom: 0.9rem;
}

.upload-right-text textarea {
    min-height: 290px;
    width: 100%;
    box-sizing: border-box;
    padding: 1rem !important;
    font-size: 0.95rem !important;
    border-radius: 18px !important;
    border: 1px solid rgba(244, 63, 94, 0.35) !important;
    background: rgba(9, 14, 26, 0.92) !important;
    color: #e5e7eb !important;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 0.5rem;
}

[data-testid="stTabs"] [role="tab"] {
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.03);
    color: #cbd5e1;
    transition: all 0.2s ease;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, rgba(240, 59, 210, 0.18), rgba(124, 58, 237, 0.16));
    border-color: rgba(240, 59, 210, 0.35);
    color: white;
}

[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(255,255,255,0.06);
}

@media (max-width: 1100px) {
    .analysis-grid {
        grid-template-columns: 1fr;
    }
}

.emotion-display {
    grid-template-columns: 1fr;
}
</style>
""",
        unsafe_allow_html=True,
    )


def sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <span class="brand-mark">m</span>
                <span>MoodSyncAI</span>
            </div>
            <div class="profile-card">
                <div class="avatar">👤</div>
                <div>
                    <div class="profile-name">Vaishnavi</div>
                    <div class="profile-role">AI Explorer</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            [
                "⌂   Home",
                "🎥   Live Camera",
                "📊   Analyze",
                "↺   History",
                "▥   Insights",
                "♡   Favorites",
                "⚙   Settings",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="status-card">
                <div class="panel-title">System Status</div>
                <div class="status-row"><span>CNN Emotion Model</span><span class="pill">Active</span></div>
                <div class="status-row"><span>NLP Sentiment Model</span><span class="pill">Active</span></div>
                <div class="status-row"><span>Audio Analyzer</span><span class="pill">Active</span></div>
                <div class="status-row"><span>AI Summary Engine</span><span class="pill">Active</span></div>
            </div>
            <div class="quote-card">
                <div class="quote-mark">“</div>
                <p>Emotions are data.<br>We just help you<br>understand them better.</p>
                <div class="muted">- MoodSyncAI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


def probability_panel(scores):
    colors = {
        "Happy": "#57de63",
        "Neutral": "#e5bf1f",
        "Surprise": "#29a6f5",
        "Sad": "#4670ff",
        "Angry": "#eb3d69",
        "Fear": "#b54ae0",
        "Disgust": "#ee7024",
    }
    rows = []
    for emotion, value in scores.items():
        rows.append(
            '<div class="prob-row">'
            f"<span>{emotion}</span>"
            '<div class="track">'
            f'<div class="bar" style="width:{max(0, min(value, 100))}%; background:{colors.get(emotion, "#57de63")};"></div>'
            "</div>"
            f"<span>{value:.2f}%</span>"
            "</div>"
        )
    return "".join(rows)


def emotion_panel():
    emotion = st.session_state.get("current_emotion", "Happy").title()
    confidence = st.session_state.get("emotion_confidence", 92.35)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="panel-title">Current Emotion</div>
            <div class="divider"></div>
            <div class="emotion-display">
                <div class="face-icon">⌣</div>
                <div>
                    <div class="emotion-name">{emotion}</div>
                    <div class="confidence-pill">Confidence: {confidence:.2f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def emotion_probabilities_panel():
    scores = st.session_state.get("emotion_scores", DEFAULT_SCORES)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="panel-title">Emotion Probabilities</div>
            {probability_panel(scores)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def instructions_panel():
    st.markdown(
        """
        <div class="glass-card">
            <div class="panel-title">💡 Tips for Best Results</div>
            <ul class="instructions">
                <li>Upload high-quality images with clear faces</li>
                <li>Use complete sentences for better text analysis</li>
                <li>Ensure good audio quality for voice analysis</li>
                <li>Combine multiple analyses for fusion insights</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mood_selector(default_mood=None):
    moods = ["Happy", "Neutral", "Surprise", "Sad", "Angry", "Fear", "Disgust"]
    current = (default_mood or st.session_state.get("current_emotion", "Happy")).title()
    if current not in moods:
        current = "Happy"

    return st.selectbox(
        "Mood for Spotify Picks",
        moods,
        index=moods.index(current),
        key="spotify_mood_select",
        label_visibility="visible",
    )


def music_panel(mood=None):
    mood = (mood or st.session_state.get("current_emotion", "Happy")).title()
    recommendations = get_recommendations(mood)
    cards = []

    for _, song in recommendations.iterrows():
        spotify_url = song.get("spotify_url", "#")
        cards.append(
            '<div class="song-card">'
            f'<div class="song-title">{song["track"]}</div>'
            f'<div class="song-meta">{song["artist"]} • {song["reason"]}</div>'
            '<div class="feature-row">'
            f'<span class="feature-chip">Valence {song["valence"]:.2f}</span>'
            f'<span class="feature-chip">Energy {song["energy"]:.2f}</span>'
            f'<span class="feature-chip">Dance {song["danceability"]:.2f}</span>'
            "</div>"
            f'<a class="spotify-link" href="{spotify_url}" target="_blank">Open on Spotify</a>'
            "</div>"
        )

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="panel-title">Spotify Mood Picks</div>
            <div class="muted" style="margin-bottom:1rem;">Suggested for {mood}</div>
            {"".join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def trend_chart():
    history = st.session_state.get("emotion_history", ["Happy", "Happy", "Neutral", "Happy"])
    if not history:
        history = ["Happy", "Happy", "Neutral", "Happy"]

    order = ["Disgust", "Fear", "Angry", "Sad", "Neutral", "Surprise", "Happy"]
    emotion_to_level = {emotion: index for index, emotion in enumerate(order)}
    data = pd.DataFrame(
        {
            "Time": list(range(len(history))),
            "Emotion": [emotion.title() for emotion in history],
            "Level": [emotion_to_level.get(emotion.title(), 4) for emotion in history],
        }
    )

    chart = (
        alt.Chart(data)
        .mark_line(color="#f03bd2", strokeWidth=3, interpolate="monotone")
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title=None, labelColor="#a7afc0", gridColor="rgba(148,163,184,0.08)")),
            y=alt.Y(
                "Level:Q",
                scale=alt.Scale(domain=[0, 6]),
                axis=alt.Axis(
                    title=None,
                    values=list(range(len(order))),
                    labelExpr="['Disgust','Fear','Angry','Sad','Neutral','Surprise','Happy'][datum.value]",
                    labelColor="#dce2ee",
                    gridColor="rgba(148,163,184,0.10)",
                ),
            ),
            tooltip=[alt.Tooltip("Emotion:N"), alt.Tooltip("Level:Q")],
        )
        .properties(height=230)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=13)
        .configure(background="transparent")
    )
    st.altair_chart(chart, use_container_width=True)


def home_page():
    live_camera.init_camera_state()
    st.markdown(
        """
        <div class="page-title">
            <div class="title-left">
                <h1><span class="gradient-text">⌂</span> Live Camera Emotion Detection</h1>
                <p>Real-time emotion detection using AI</p>
            </div>
            <div class="page-status">
                <span class="live-dot"></span>
                Live
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    main_col, side_col = st.columns([2.3, 0.9], gap="large")

    with side_col:
        emotion_panel()
        emotion_probabilities_panel()
        music_panel(st.session_state.get("current_emotion", "Happy"))

    with main_col:
        st.markdown('<div class="camera-card">', unsafe_allow_html=True)
        live_camera.run_live_camera(show_header=False, auto_start=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="panel-title">Live Emotion Trend</div>', unsafe_allow_html=True)
        trend_chart()
        st.markdown('</div>', unsafe_allow_html=True)


def render_result_card(title, metric_text, label_text, chip_text, note_text):
    st.markdown(
        f'<div class="result-card">'
        f'<h3>{title}</h3>'
        f'<div class="analysis-metric">{metric_text}</div>'
        f'<div class="analysis-label">{label_text}</div>'
        f'<div class="stat-chip">{chip_text}</div>'
        f'<div class="result-note">{note_text}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def upload_page():
    st.markdown(
        """
        <div class="page-title">
            <div class="title-left">
                <h1><span class="gradient-text">📊</span> Emotion & Sentiment Analysis</h1>
                <p>Upload images, text, and audio for multimodal emotion analysis</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "photo_analysis_result" not in st.session_state:
        st.session_state.photo_analysis_result = None
    if "text_analysis_result" not in st.session_state:
        st.session_state.text_analysis_result = None
    if "audio_analysis_result" not in st.session_state:
        st.session_state.audio_analysis_result = None
    if "audio_transcript_value" not in st.session_state:
        st.session_state.audio_transcript_value = ""

    photo_tab, text_tab, audio_tab, fusion_tab = st.tabs([
        "📸 Photo Analysis",
        "📝 Text Analysis",
        "🎵 Audio Analysis",
        "🔗 Fusion Insights",
    ])

    with photo_tab:
        st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Face Emotion Detection</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="photo_upload", label_visibility="collapsed")

        if uploaded_file:
            st.markdown('<div class="upload-preview-card">', unsafe_allow_html=True)
            st.image(uploaded_file, width=520)
            st.markdown(
                f'<div class="upload-meta">{uploaded_file.name} • {round(uploaded_file.size / 1024, 1)} KB</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="upload-input-card">'
                '<strong>📸 Upload your face image</strong>'
                '<div class="upload-placeholder">Drag and drop a photo here, or click to browse. The AI will analyze facial emotions.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        if st.button("Analyze Photo", key="analyze_photo_tab", use_container_width=True):
            if uploaded_file:
                with open("temp.jpg", "wb") as file:
                    file.write(uploaded_file.getbuffer())
                emotion, emotion_scores = detect_emotion("temp.jpg")
                st.session_state.photo_analysis_result = {
                    "emotion": emotion,
                    "scores": emotion_scores,
                }
            else:
                st.warning("Please upload an image before analyzing.")

        if st.session_state.photo_analysis_result:
            emotion = st.session_state.photo_analysis_result["emotion"]
            score_text = st.session_state.photo_analysis_result["scores"].get(emotion, 0)
            render_result_card(
                "🎭 Visual Emotion",
                emotion.upper(),
                f"Confidence: {score_text:.0f}%",
                "Face-based detection",
                "Analyzed using CNN-based facial emotion recognition.",
            )
        else:
            st.markdown(
                '<div class="result-card">'
                '<h3>🎭 Visual Emotion</h3>'
                '<div class="result-note">Upload an image and click Analyze to see results.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with text_tab:
        st.markdown('<div class="upload-panel upload-right-text">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Text Sentiment Analysis</div>', unsafe_allow_html=True)
        with st.form(key="text_analysis_form"):
            text = st.text_area("", height=240, key="text_input", label_visibility="collapsed", placeholder="Enter your message or sentence...")
            st.markdown(
                '<div class="upload-details">Write anything you want to analyze the sentiment and emotional tone.</div>',
                unsafe_allow_html=True,
            )
            analyze_text = st.form_submit_button("Analyze Text", use_container_width=True)

        if analyze_text:
            if text and text.strip():
                sentiment, confidence = detect_sentiment(text)
                st.session_state.text_analysis_result = {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "text": text.strip(),
                }
            else:
                st.warning("Please enter some text before analyzing.")

        if st.session_state.text_analysis_result:
            sentiment = st.session_state.text_analysis_result["sentiment"]
            confidence = st.session_state.text_analysis_result["confidence"]
            render_result_card(
                "💬 Text Sentiment",
                sentiment.upper(),
                f"Confidence: {round(confidence * 100, 0)}%",
                "NLP-based sentiment",
                "Analyzes semantic meaning, tone, and emotional language.",
            )
        else:
            st.markdown(
                '<div class="result-card">'
                '<h3>💬 Text Sentiment</h3>'
                '<div class="result-note">Enter text and click Analyze to see results.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with audio_tab:
        st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Audio Sentiment Detection</div>', unsafe_allow_html=True)
        with st.form(key="audio_analysis_form"):
            uploaded_audio = st.file_uploader("", type=["wav", "mp3"], key="audio_upload", label_visibility="collapsed")
            if uploaded_audio:
                st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
                st.markdown(
                    f'<div class="upload-meta">{uploaded_audio.name} • {round(uploaded_audio.size / 1024, 1)} KB</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="upload-input-card">'
                    '<strong>🎵 Upload your audio</strong>'
                    '<div class="upload-placeholder">Drag and drop an audio file (WAV, MP3), or click to browse. Your voice will be transcribed and analyzed.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            analyze_audio_submit = st.form_submit_button("Analyze Audio", use_container_width=True)

        if analyze_audio_submit:
            if uploaded_audio:
                ext = uploaded_audio.name.split(".")[-1]
                with open(f"temp_audio.{ext}", "wb") as file:
                    file.write(uploaded_audio.getbuffer())
                try:
                    transcript, audio_sentiment, audio_confidence = analyze_audio(f"temp_audio.{ext}")
                    st.session_state.audio_analysis_result = {
                        "transcript": transcript,
                        "sentiment": audio_sentiment,
                        "confidence": audio_confidence,
                    }
                    st.session_state.audio_transcript_value = transcript
                except Exception as error:
                    st.error(f"Audio analysis failed: {error}")
            else:
                st.warning("Please upload an audio file before analyzing.")

        if st.session_state.audio_analysis_result:
            sentiment = st.session_state.audio_analysis_result["sentiment"]
            confidence = st.session_state.audio_analysis_result["confidence"]
            render_result_card(
                "🎙️ Audio Sentiment",
                sentiment.upper(),
                f"Confidence: {round(confidence * 100, 0)}%",
                "Voice sentiment analysis",
                "Transcribed and analyzed using advanced NLP models.",
            )
            st.markdown(
                '<div class="glass-card summary-card">'
                '<div class="panel-title">Transcript</div>'
                f'<p>{st.session_state.audio_analysis_result["transcript"] or "No transcript available."}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card">'
                '<h3>🎙️ Audio Sentiment</h3>'
                '<div class="result-note">Upload audio and click Analyze to see results.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with fusion_tab:
        st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Multimodal Fusion Analysis</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.photo_analysis_result:
                emotion = st.session_state.photo_analysis_result["emotion"]
                render_result_card(
                    "🎭 Face Emotion",
                    emotion.upper(),
                    "",
                    "Photo-based",
                    "Detected from facial expression.",
                )
            else:
                st.markdown(
                    '<div class="result-card"><h3>🎭 Face Emotion</h3><div class="result-note">Pending analysis</div></div>',
                    unsafe_allow_html=True,
                )
        
        with col2:
            if st.session_state.text_analysis_result:
                sentiment = st.session_state.text_analysis_result["sentiment"]
                render_result_card(
                    "💬 Text Sentiment",
                    sentiment.upper(),
                    "",
                    "Text-based",
                    "Detected from language analysis.",
                )
            else:
                st.markdown(
                    '<div class="result-card"><h3>💬 Text Sentiment</h3><div class="result-note">Pending analysis</div></div>',
                    unsafe_allow_html=True,
                )
        
        with col3:
            if st.session_state.audio_analysis_result:
                sentiment = st.session_state.audio_analysis_result["sentiment"]
                render_result_card(
                    "🎙️ Audio Sentiment",
                    sentiment.upper(),
                    "",
                    "Audio-based",
                    "Detected from voice analysis.",
                )
            else:
                st.markdown(
                    '<div class="result-card"><h3>🎙️ Audio Sentiment</h3><div class="result-note">Pending analysis</div></div>',
                    unsafe_allow_html=True,
                )

        fusion_messages = []
        if st.session_state.photo_analysis_result and st.session_state.text_analysis_result:
            fusion_messages.append(
                generate_summary(
                    st.session_state.text_analysis_result["sentiment"],
                    st.session_state.photo_analysis_result["emotion"],
                    detect_mismatch(
                        st.session_state.text_analysis_result["sentiment"],
                        st.session_state.photo_analysis_result["emotion"],
                    ),
                )
            )
        if st.session_state.photo_analysis_result and st.session_state.audio_analysis_result:
            fusion_messages.append(
                generate_summary(
                    st.session_state.audio_analysis_result["sentiment"],
                    st.session_state.photo_analysis_result["emotion"],
                    detect_mismatch(
                        st.session_state.audio_analysis_result["sentiment"],
                        st.session_state.photo_analysis_result["emotion"],
                    ),
                )
            )

        if fusion_messages:
            for message in fusion_messages:
                st.markdown(
                    '<div class="glass-card summary-card">'
                    '<div class="panel-title">✨ Fusion Insight</div>'
                    f'<p>{message}</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="glass-card summary-card">'
                '<div class="panel-title">✨ Fusion Insight</div>'
                '<p>Complete at least two analyses (Photo, Text, or Audio) to generate fusion insights.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
def placeholder_page(title):
    st.markdown(
        f"""
        <div class="page-title">
            <div class="title-left">
                <h1>{title}</h1>
                <p>This section is ready for your next MoodSyncAI feature.</p>
            </div>
        </div>
        <div class="glass-card">
            <div class="panel-title">Coming Soon</div>
            <p class="muted">Use the Live Camera or Upload + Text Analysis pages for the working AI flows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def live_camera_page():
    live_camera.init_camera_state()

    # Page header
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

    main_col, side_col = st.columns([3, 1.05])

    with side_col:
        current = st.session_state.get("current_emotion", "Happy")
        emotion_panel()
        emotion_probabilities_panel()

        # Spotify recommendations
        music_panel(current)

        history = st.session_state.get("emotion_history", [])
        if history:
            st.markdown('<div class="glass-card"><div class="panel-title">Live Emotion Trend</div>', unsafe_allow_html=True)
            trend_chart()
            st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        st.markdown('<div class="glass-card"><div class="panel-title">Live Emotion Trend</div>', unsafe_allow_html=True)
        trend_chart()
        st.markdown('</div>', unsafe_allow_html=True)

        # Run the camera UI and controls from live_camera module
        live_camera.run_live_camera()


inject_css()
selected_page = sidebar()

if "Home" in selected_page:
    home_page()
elif "Live Camera" in selected_page:
    live_camera_page()
elif "Analyze" in selected_page:
    upload_page()
else:
    placeholder_page(selected_page.split("   ")[-1])
