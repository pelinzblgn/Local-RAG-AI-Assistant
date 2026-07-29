import math
from foundry_local_sdk import Configuration, FoundryLocalManager


EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts using Foundry Local."""

    cleaned_texts = [text.strip() for text in texts if text.strip()]

    if not cleaned_texts:
        raise ValueError("At least one non-empty text is required.")

    config = Configuration(app_name="local_rag_ai_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)

    try:
        print("Embedding modeli kontrol ediliyor...")

        model.download(
            lambda progress: print(
                f"\rEmbedding modeli: %{progress:.1f}",
                end="",
                flush=True,
            )
        )
        print()

        print("Embedding modeli belleğe yükleniyor...")
        model.load()

        client = model.get_embedding_client()

        print("Embedding'ler üretiliyor...")
        response = client.generate_embeddings(cleaned_texts)

        return [item.embedding for item in response.data]

    finally:
        if model.is_loaded:
            model.unload()