import nltk

nltk.download("punkt")


def tokenize_text(text: str) -> dict:
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    return {"sentences": sentences, "words": words}


if __name__ == "__main__":
    sample_text = (
        "Hello there! This is a simple NLP example. "
        "It shows sentence and word tokenization."
    )
    tokens = tokenize_text(sample_text)

    print("Sentences:")
    for sentence in tokens["sentences"]:
        print(f"- {sentence}")

    print("\nWord Tokens:")
    for idx, sentence_words in enumerate(tokens["words"], start=1):
        print(f"Sentence {idx}: {sentence_words}")
