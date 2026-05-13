import wave
import numpy as np
from transformers import pipeline
from sentiment_detector import detect_sentiment

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    language="en",
)

_SAMPLE_WIDTH_TO_DTYPE = {
    1: np.uint8,
    2: np.int16,
    4: np.int32,
}


def resample_audio(audio, original_rate, target_rate=16000):
    if original_rate == target_rate:
        return audio

    duration = audio.shape[0] / original_rate
    target_length = int(round(duration * target_rate))
    if target_length <= 0:
        return audio

    original_times = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
    target_times = np.linspace(0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, original_times, audio)


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()

    dtype = _SAMPLE_WIDTH_TO_DTYPE.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    max_value = np.iinfo(dtype).max
    if max_value != 0:
        audio = audio / max_value

    audio = resample_audio(audio, sample_rate, target_rate=16000)
    return audio, 16000


def transcribe_audio(path):
    audio, sample_rate = load_wav(path)
    result = asr_pipeline(audio)
    return result.get("text", "").strip()


def analyze_audio(path):
    transcript = transcribe_audio(path)
    if transcript and len(transcript.split()) >= 2:
        sentiment, score = detect_sentiment(transcript)
        if score < 0.60:
            sentiment = "Neutral"
    else:
        sentiment, score = "Neutral", 0.0

    return transcript, sentiment, score
