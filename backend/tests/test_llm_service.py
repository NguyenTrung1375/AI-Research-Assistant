from unittest.mock import MagicMock
import pytest

from app.rag.generation.llm_service import LLMService


def test_llm_service_generates_text_with_system_instruction_and_temperature():
    # Mock Google GenAI Client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Transformer là kiến trúc mạng nơ-ron dựa trên cơ chế self-attention [Trang 2]."
    mock_client.models.generate_content.return_value = mock_response

    service = LLMService(
        api_key="test-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    prompt = "Transformer là gì?"
    system_inst = "Bạn là trợ lý nghiên cứu."

    result = service.generate(
        prompt=prompt,
        system_instruction=system_inst,
        temperature=0.2,
    )

    assert result == "Transformer là kiến trúc mạng nơ-ron dựa trên cơ chế self-attention [Trang 2]."

    # Kiểm tra lời gọi đến Google GenAI client
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == prompt
    assert call_kwargs["config"].system_instruction == system_inst
    assert call_kwargs["config"].temperature == 0.2


def test_llm_service_rejects_empty_prompt():
    service = LLMService(api_key="test-key", client=MagicMock())
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        service.generate(prompt="   ")


def test_llm_service_raises_error_when_api_key_missing():
    service = LLMService(api_key="", client=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"):
        service.generate(prompt="Hello")


def test_llm_service_handles_api_exception():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Google GenAI Quota Exceeded")

    service = LLMService(api_key="test-key", client=mock_client)
    with pytest.raises(RuntimeError, match="LLM generation failed"):
        service.generate(prompt="Hello")

