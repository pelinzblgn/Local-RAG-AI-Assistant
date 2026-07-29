from src.config import get_foundry_manager

MODEL_ALIAS = "phi-4-mini"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local AI assistant. "
    "Answer clearly, accurately, and concisely."
)


def generate_response(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate a response with the local Phi-4 Mini model."""

    clean_prompt = prompt.strip()
    clean_system_prompt = system_prompt.strip()

    if not clean_prompt:
        raise ValueError("Prompt cannot be empty.")

    if not clean_system_prompt:
        raise ValueError("System prompt cannot be empty.")

    manager = get_foundry_manager()
    model = manager.catalog.get_model(MODEL_ALIAS)

    try:
        print("Chat modeli kontrol ediliyor...")

        model.download(
            lambda progress: print(
                f"\rChat modeli: %{progress:.1f}",
                end="",
                flush=True,
            )
        )
        print()

        print("Chat modeli belleğe yükleniyor...")
        model.load()

        client = model.get_chat_client()

        print("RAG isteği modele gönderiliyor...")

        response = client.complete_chat(
            [
                {
                    "role": "system",
                    "content": clean_system_prompt,
                },
                {
                    "role": "user",
                    "content": clean_prompt,
                },
            ]
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "The local model returned an empty response."
            )

        return answer.strip()

    finally:
        if model.is_loaded:
            model.unload()