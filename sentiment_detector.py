classifier = None


def detect_sentiment(text):
    global classifier
    if classifier is None:
        from transformers import pipeline

        classifier = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    result = classifier(text)[0]

    return result["label"], result["score"]
