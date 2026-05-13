import pandas as pd
from urllib.parse import quote_plus


def spotify_search_url(track, artist):
    query = quote_plus(f"{track} {artist}")
    return f"https://open.spotify.com/search/{query}"


SPOTIFY_TRACKS = [
    {
        "track": "Good as Hell",
        "artist": "Lizzo",
        "mood": "Happy",
        "valence": 0.92,
        "energy": 0.73,
        "danceability": 0.69,
        "reason": "bright, confident, and upbeat",
    },
    {
        "track": "Levitating",
        "artist": "Dua Lipa",
        "mood": "Happy",
        "valence": 0.91,
        "energy": 0.82,
        "danceability": 0.70,
        "reason": "high-energy pop with a cheerful pulse",
    },
    {
        "track": "Sunflower",
        "artist": "Post Malone, Swae Lee",
        "mood": "Happy",
        "valence": 0.91,
        "energy": 0.52,
        "danceability": 0.76,
        "reason": "warm and relaxed without losing bounce",
    },
    {
        "track": "Weightless",
        "artist": "Marconi Union",
        "mood": "Neutral",
        "valence": 0.42,
        "energy": 0.21,
        "danceability": 0.35,
        "reason": "calm ambient texture for emotional balance",
    },
    {
        "track": "Bloom",
        "artist": "The Paper Kites",
        "mood": "Neutral",
        "valence": 0.52,
        "energy": 0.30,
        "danceability": 0.46,
        "reason": "soft acoustic sound for a steady mood",
    },
    {
        "track": "Holocene",
        "artist": "Bon Iver",
        "mood": "Neutral",
        "valence": 0.37,
        "energy": 0.31,
        "danceability": 0.43,
        "reason": "reflective, spacious, and gentle",
    },
    {
        "track": "Fix You",
        "artist": "Coldplay",
        "mood": "Sad",
        "valence": 0.28,
        "energy": 0.42,
        "danceability": 0.21,
        "reason": "comforting build with hopeful release",
    },
    {
        "track": "Someone Like You",
        "artist": "Adele",
        "mood": "Sad",
        "valence": 0.21,
        "energy": 0.33,
        "danceability": 0.56,
        "reason": "slow piano ballad for processing sadness",
    },
    {
        "track": "The Night We Met",
        "artist": "Lord Huron",
        "mood": "Sad",
        "valence": 0.18,
        "energy": 0.36,
        "danceability": 0.45,
        "reason": "soft melancholy with emotional space",
    },
    {
        "track": "Believer",
        "artist": "Imagine Dragons",
        "mood": "Angry",
        "valence": 0.43,
        "energy": 0.78,
        "danceability": 0.77,
        "reason": "intense rhythm that channels frustration",
    },
    {
        "track": "Stronger",
        "artist": "Kanye West",
        "mood": "Angry",
        "valence": 0.49,
        "energy": 0.72,
        "danceability": 0.62,
        "reason": "driving beat for converting tension into momentum",
    },
    {
        "track": "Numb",
        "artist": "Linkin Park",
        "mood": "Angry",
        "valence": 0.24,
        "energy": 0.86,
        "danceability": 0.50,
        "reason": "cathartic rock energy",
    },
    {
        "track": "Breathe Me",
        "artist": "Sia",
        "mood": "Fear",
        "valence": 0.19,
        "energy": 0.34,
        "danceability": 0.44,
        "reason": "gentle and grounding for anxious moments",
    },
    {
        "track": "Intro",
        "artist": "The xx",
        "mood": "Fear",
        "valence": 0.31,
        "energy": 0.52,
        "danceability": 0.58,
        "reason": "minimal rhythm that settles the atmosphere",
    },
    {
        "track": "Experience",
        "artist": "Ludovico Einaudi",
        "mood": "Fear",
        "valence": 0.25,
        "energy": 0.39,
        "danceability": 0.28,
        "reason": "cinematic calm with gradual emotional lift",
    },
    {
        "track": "On Top Of The World",
        "artist": "Imagine Dragons",
        "mood": "Surprise",
        "valence": 0.87,
        "energy": 0.78,
        "danceability": 0.64,
        "reason": "playful, bright, and energetic",
    },
    {
        "track": "Electric Feel",
        "artist": "MGMT",
        "mood": "Surprise",
        "valence": 0.77,
        "energy": 0.73,
        "danceability": 0.76,
        "reason": "colorful groove with a curious edge",
    },
    {
        "track": "Creep",
        "artist": "Radiohead",
        "mood": "Disgust",
        "valence": 0.14,
        "energy": 0.28,
        "danceability": 0.39,
        "reason": "uneasy and raw for uncomfortable emotions",
    },
    {
        "track": "Breathe Me",
        "artist": "Sia",
        "mood": "Disgust",
        "valence": 0.18,
        "energy": 0.32,
        "danceability": 0.36,
        "reason": "slow, intimate, and emotionally heavy",
    },
]


def spotify_dataset():
    dataset = pd.DataFrame(SPOTIFY_TRACKS)
    if "spotify_url" not in dataset.columns:
        dataset["spotify_url"] = dataset.apply(
            lambda row: spotify_search_url(row["track"], row["artist"]), axis=1
        )
    return dataset


def get_recommendations(mood, limit=3):
    dataset = spotify_dataset()
    mood = str(mood).title()
    if mood not in dataset["mood"].unique():
        mood = "Neutral"

    matches = dataset[dataset["mood"] == mood]
    if matches.empty:
        matches = dataset[dataset["mood"] == "Neutral"]

    return matches.sort_values(
        by=["valence", "energy", "danceability"],
        ascending=False,
    ).head(limit)
