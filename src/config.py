from pathlib import Path

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
DATABASE_DIRECTORY = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIRECTORY / "rag.db"


# ==========================================================
# Application
# ==========================================================

APP_NAME = "local_rag_ai_assistant"


# ==========================================================
# Models
# ==========================================================

CHAT_MODEL_ALIAS = "phi-4-mini"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"


# ==========================================================
# Chunking
# ==========================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


# ==========================================================
# Retrieval
# ==========================================================

TOP_K = 3
MIN_SIMILARITY_SCORE = 0.20


# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"


# ==========================================================
# Foundry Local
# ==========================================================

def get_foundry_manager() -> FoundryLocalManager:
    """
    Initialize Foundry Local once and return its singleton manager.

    Returns:
        Initialized Foundry Local manager.

    Raises:
        RuntimeError: If the manager cannot be initialized.
    """

    manager = FoundryLocalManager.instance

    if manager is None:
        configuration = Configuration(
            app_name=APP_NAME,
        )

        FoundryLocalManager.initialize(
            configuration
        )

        manager = FoundryLocalManager.instance

    if manager is None:
        raise RuntimeError(
            "Foundry Local Manager başlatılamadı."
        )

    return manager