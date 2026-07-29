from foundry_local_sdk import Configuration, FoundryLocalManager


APP_NAME = "local_rag_ai_assistant"


def get_foundry_manager() -> FoundryLocalManager:
    """Initialize Foundry Local once and return its manager."""

    manager = FoundryLocalManager.instance

    if manager is None:
        config = Configuration(app_name=APP_NAME)
        FoundryLocalManager.initialize(config)
        manager = FoundryLocalManager.instance

    if manager is None:
        raise RuntimeError(
            "Foundry Local Manager başlatılamadı."
        )

    return manager