# MoodSyncAI

A Streamlit dashboard for multimodal emotion & sentiment analysis — live webcam facial emotion detection, plus photo, text, and audio analysis, fused together with mood-matched Spotify recommendations.

> Emotions are data. We just help you understand them better.

## Features

- **Live Camera Emotion Detection** — real-time facial emotion recognition from your webcam feed (OpenCV face detection + DeepFace emotion classification), with on-frame bounding boxes, a live emotion trend chart, and FPS readout
- **Photo Analysis** — upload an image and get a facial emotion breakdown
- **Text Analysis** — sentiment analysis on free-form text via a transformer model
- **Audio Analysis** — upload a `.wav`/`.mp3` clip, get it transcribed (Whisper) and sentiment-scored
- **Fusion Insights** — combines results across modalities (face + text, face + audio) and flags mismatches (e.g. positive wording paired with a negative facial expression)
- **Spotify Mood Picks** — mood-matched song recommendations with valence/energy/danceability stats and a direct Spotify search link
- **Live Emotion Trend** — rolling chart of the dominant detected emotion over a session

## How It Works

| Module | Role |
|---|---|
| `emotion_detector.py` | Facial emotion classification via **DeepFace** |
| `live_camera.py` | Webcam capture, OpenCV Haar-cascade face detection, live overlay + session state |
| `sentiment_detector.py` | Text sentiment via a Hugging Face `transformers` pipeline (DistilBERT, SST-2) |
| `audio_analyzer.py` | Speech-to-text via **Whisper** (`openai/whisper-small`), then sentiment on the transcript |
| `fusion.py` | Flags mismatches between text/audio sentiment and detected facial emotion |
| `summary_generator.py` | Generates the natural-language fusion insight summary |
| `music_recommender.py` | Mood-matched Spotify track picks |
| `app.py` | Streamlit UI — page layout, styling, navigation |

## Tech Stack

- **UI:** Streamlit, Altair (charts)
- **Computer vision:** OpenCV, DeepFace (TensorFlow backend)
- **NLP:** Hugging Face Transformers, PyTorch
- **Speech-to-text:** Whisper (`openai/whisper-small`)
- **Data:** pandas, numpy
- Python 3.10 (see `runtime.txt`)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/vaishnavi2020/MoodSync.git
cd MoodSync

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. A webcam is required for the Live Camera page; grant camera permission when prompted by your OS/browser.

## Usage Tips

- Upload high-quality images with clear, well-lit faces
- Use complete sentences for better text sentiment results
- Ensure clean audio (minimal background noise) for voice analysis
- Run at least two of Photo / Text / Audio analysis to unlock Fusion Insights
- Click **Stop Camera** to end a live detection session; **Capture Snapshot** saves the current frame as `snapshot.jpg`

## Project Structure

```
MoodSync/
├── app.py                   # Main Streamlit app & UI
├── live_camera.py           # Webcam capture + live emotion overlay
├── emotion_detector.py       # DeepFace facial emotion detection
├── sentiment_detector.py     # Text sentiment (transformers)
├── audio_analyzer.py         # Whisper transcription + sentiment
├── fusion.py                 # Cross-modal mismatch detection
├── summary_generator.py       # Fusion insight text generation
├── music_recommender.py       # Mood-based Spotify recommendations
├── requirements.txt
└── runtime.txt
```

## License

No license file is currently included — add one (e.g. MIT) to clarify usage terms for this public repo.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/vaishnavi2020/MoodSync/issues).
