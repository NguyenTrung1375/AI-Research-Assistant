import pytest

from app.rag.generation.prompt_template import (
    build_rag_user_prompt,
    build_system_instruction,
)


def test_system_instruction_contains_strict_grounding_and_page_citation_rules():
    instruction = build_system_instruction()

    # Phải có quy tắc chống ảo giác (chỉ dùng thông tin trong ngữ cảnh)
    assert "chỉ sử dụng" in instruction.lower() or "duy nhất" in instruction.lower()
    # Phải có quy tắc trích dẫn số trang
    assert "trang" in instruction.lower()
    # Phải có quy tắc xử lý khi tài liệu không có thông tin
    assert "không có" in instruction.lower() or "không đủ căn cứ" in instruction.lower()


def test_build_rag_user_prompt_combines_query_and_context_cleanly():
    query = "Mô hình transformer có ưu điểm gì?"
    context = "--- NGUỒN [1] (Trang: 4) ---\nTransformer xử lý dữ liệu song song hiệu quả."

    prompt = build_rag_user_prompt(query=query, context=context)

    # Prompt phải chứa context và câu hỏi
    assert context in prompt
    assert query in prompt
    # Phải có phân cách rõ ràng giữa context và câu hỏi
    assert "NGỮ CẢNH TÀI LIỆU" in prompt or "CONTEXT" in prompt
    assert "CÂU HỎI" in prompt or "QUESTION" in prompt


def test_build_rag_user_prompt_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        build_rag_user_prompt(query="   ", context="Some context")

