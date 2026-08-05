import math
from collections.abc import Sequence


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Calculate cosine similarity between two numerical vectors.

    Returns a value between -1.0 and 1.0.
    Returns 0.0 when at least one vector has zero magnitude.

    Raises:
        ValueError: If vectors are empty or have different lengths.
    """

    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty.")

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )
    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    similarity = dot_product / (magnitude_a * magnitude_b)

    # Floating-point calculations can produce values such as 1.0000000002.
    return max(-1.0, min(1.0, similarity))