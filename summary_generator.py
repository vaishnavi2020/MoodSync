def generate_summary(sentiment, emotion, mismatch):

    if mismatch:
        return (
            f"The speaker expressed {sentiment} sentiment verbally, "
            f"but facial emotion appears {emotion}. "
            f"This may indicate emotional incongruence."
        )

    return (
        f"Both verbal sentiment and facial emotion appear aligned."
    )