import logging
from functools import lru_cache
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class LLMService:
    """
    Adapter tương tác với Google Gemini API thông qua SDK google-genai.
    Hỗ trợ cấu hình System Instruction, Temperature thấp để giảm ảo giác.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL or DEFAULT_GEMINI_MODEL
        self._client = client

    @property
    def client(self) -> Any:
        """Lazy load Google GenAI client."""
        if self._client is None:
            if not self.api_key or not self.api_key.strip():
                raise RuntimeError(
                    "GEMINI_API_KEY is not configured. Please provide a valid Gemini API key in your .env file."
                )

            from google import genai

            logger.info("Initializing Google GenAI Client with model: %s", self.model)
            self._client = genai.Client(api_key=self.api_key.strip())
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Sinh văn bản từ Google Gemini.
        - prompt: Nội dung đầu vào
        - system_instruction: Chỉ thị hệ thống
        - temperature: 0.2 để câu trả lời có tính xác thực và bám sát ngữ cảnh cao nhất
        """
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=cleaned_prompt,
                config=config,
            )

            text_output = getattr(response, "text", "")
            return text_output.strip() if text_output else ""

        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("Gemini API generation error: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc


@lru_cache
def get_llm_service() -> LLMService:
    """Trả về thể hiện singleton của LLMService."""
    return LLMService()

