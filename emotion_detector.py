DeepFace = None

def detect_emotion(img_path):
    global DeepFace
    if DeepFace is None:
        from deepface import DeepFace

    result = DeepFace.analyze(
        img_path=img_path,
        actions=['emotion'],
        enforce_detection=False
    )

    emotion = result[0]['dominant_emotion']
    scores = result[0]['emotion']

    return emotion, scores