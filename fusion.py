def detect_mismatch(sentiment, emotion):

    negative_emotions = ["sad", "angry", "fear"]

    if sentiment.lower() == "positive" and emotion.lower() in negative_emotions:
        return True

    return False
