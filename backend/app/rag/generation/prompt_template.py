"""Module quản lý Prompt Template và System Instruction cho RAG Pipeline."""

DEFAULT_SYSTEM_INSTRUCTION = (
    "Bạn là một Trợ lý Nghiên cứu Khoa học (AI Research Assistant) chuyên nghiệp, "
    "có nhiệm vụ giải đáp các câu hỏi dựa trên tài liệu nghiên cứu được cung cấp.\n\n"
    "CÁC NGUYÊN TẮC BẮT BUỘC:\n"
    "1. CHỈ SỬ DỤNG thông tin có trong phần 'NGỮ CẢNH TÀI LIỆU' được cung cấp để trả lời. "
    "Tuyệt đối không tự suy diễn, phỏng đoán hoặc đưa vào các thông tin bên ngoài ngữ cảnh.\n"
    "2. BẮT BUỘC TRÍCH DẪN SỐ TRANG: Khi đưa ra bất kỳ luận điểm, số liệu hoặc kết luận nào, "
    "hãy đính kèm số trang tương ứng theo định dạng [Trang X] ngay sau câu đó.\n"
    "3. XỬ LÝ NGOÀI PHẠM VI (OUT-OF-SCOPE): Nếu ngữ cảnh không có thông tin hoặc không đủ căn cứ để trả lời, "
    "hãy trả lời rõ ràng: 'Tài liệu được cung cấp không có thông tin hoặc không đủ căn cứ để trả lời câu hỏi này.' "
    "Không cố tình bịa câu trả lời.\n"
    "4. Văn phong học thuật, ngắn gọn, súc tích, mạch lạc."
)


def build_system_instruction() -> str:
    """Trả về chỉ dẫn hệ thống (System Instruction) chống ảo giác cho LLM."""
    return DEFAULT_SYSTEM_INSTRUCTION


def build_rag_user_prompt(query: str, context: str) -> str:
    """Đóng gói câu hỏi và ngữ cảnh đã trích xuất thành User Prompt chuẩn."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    cleaned_context = context.strip() if context else "Không có ngữ cảnh."

    return (
        f"### NGỮ CẢNH TÀI LIỆU (CONTEXT):\n"
        f"{cleaned_context}\n\n"
        f"### CÂU HỎI CỦA NGƯỜI DÙNG (QUESTION):\n"
        f"{cleaned_query}\n\n"
        f"### YÊU CẦU:\n"
        f"Dựa trên các khối ngữ cảnh ở trên, hãy trả lời câu hỏi chi tiết, chính xác "
        f"và luôn đính kèm trích dẫn [Trang X] cho từng thông tin."
    )

