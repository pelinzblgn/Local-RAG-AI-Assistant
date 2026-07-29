from foundry_local_sdk import Configuration, FoundryLocalManager


MODEL_ALIAS = "phi-4-mini"


def generate_response(prompt: str) -> str:
    """Generate a response with the local Phi-4 Mini model."""

    clean_prompt = prompt.strip()

    if not clean_prompt:
        raise ValueError("Prompt cannot be empty.")

    print("Foundry Local SDK başlatılıyor...")

    config = Configuration(app_name="local_rag_ai_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Model hazırlanıyor...")

    model = manager.catalog.get_model(MODEL_ALIAS)

    try:
        model.download(
            lambda progress: print(
                f"\rModel kontrolü: %{progress:.1f}",
                end="",
                flush=True,
            )
        )
        print()

        model.load()

        print("Modele istek gönderiliyor...")

        client = model.get_chat_client()

        response = client.complete_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful local AI assistant. "
                        "Answer clearly, accurately, and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": clean_prompt,
                },
            ]
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("The local model returned an empty response.")

        return answer.strip()

    finally:
        model.unload()