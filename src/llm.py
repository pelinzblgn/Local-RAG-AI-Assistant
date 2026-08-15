import logging
from typing import Any

from src.config import (
    CHAT_MODEL_ALIAS,
    get_foundry_manager,
)


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local AI assistant. "
    "Answer clearly, accurately, and concisely."
)


def _clean_prompt(
    prompt: str,
    field_name: str,
) -> str:
    """
    Validate and normalize a prompt value.
    """

    if not isinstance(prompt, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    clean_prompt = prompt.strip()

    if not clean_prompt:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return clean_prompt


class LocalLLM:
    """
    Manage the lifecycle of the local Foundry chat model.

    The model and client are loaded lazily and reused across
    multiple response-generation requests.
    """

    def __init__(
        self,
        model_alias: str = CHAT_MODEL_ALIAS,
    ) -> None:
        """Initialize the local LLM wrapper."""

        if not isinstance(model_alias, str):
            raise TypeError(
                "Model alias must be a string."
            )

        clean_model_alias = model_alias.strip()

        if not clean_model_alias:
            raise ValueError(
                "Model alias cannot be empty."
            )

        self._model_alias = clean_model_alias
        self._model: Any | None = None
        self._client: Any | None = None

    @property
    def model_alias(self) -> str:
        """Return the configured model alias."""

        return self._model_alias

    @property
    def is_loaded(self) -> bool:
        """Return whether the local model is loaded."""

        return bool(
            self._model is not None
            and self._model.is_loaded
        )

    def _ensure_client(self) -> Any:
        """
        Load the model when necessary and return its chat client.
        """

        if self._client is not None:
            return self._client

        manager = get_foundry_manager()

        model = manager.catalog.get_model(
            self._model_alias
        )

        logger.info(
            "Chat modeli kontrol ediliyor: %s",
            self._model_alias,
        )

        model.download(
            lambda progress: print(
                f"\rChat modeli: %{progress:.1f}",
                end="",
                flush=True,
            )
        )

        print()

        if not model.is_loaded:
            logger.info(
                "Chat modeli belleğe yükleniyor."
            )

            model.load()

        self._model = model
        self._client = model.get_chat_client()

        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """
        Generate a response with the local chat model.
        """

        clean_prompt = _clean_prompt(
            prompt,
            "Prompt",
        )

        clean_system_prompt = _clean_prompt(
            system_prompt,
            "System prompt",
        )

        client = self._ensure_client()

        logger.info(
            "RAG isteği yerel chat modeline gönderiliyor."
        )

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

        if not response.choices:
            raise RuntimeError(
                "The local model returned no response choices."
            )

        answer = response.choices[
            0
        ].message.content

        if not isinstance(answer, str):
            raise RuntimeError(
                "The local model returned an invalid response."
            )

        clean_answer = answer.strip()

        if not clean_answer:
            raise RuntimeError(
                "The local model returned an empty response."
            )

        return clean_answer

    def unload(self) -> None:
        """Unload the model and clear the cached client."""

        if self.is_loaded:
            logger.info(
                "Chat modeli bellekten kaldırılıyor."
            )

            self._model.unload()

        self._model = None
        self._client = None

    def __enter__(self) -> "LocalLLM":
        """Return the LLM instance."""

        return self
    def warm_up(self) -> None:
        """
        Load and prepare the local chat model before the first request.
        """

        self._ensure_client()

        logger.info(
            "Chat modeli warm-up tamamlandı."
        )
    
    def __exit__(
        self,
        exception_type: object,
        exception_value: object,
        traceback: object,
    ) -> None:
        """Unload the model when leaving the context block."""

        self.unload()