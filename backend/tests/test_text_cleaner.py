
import pytest

from app.rag.document_loader.text_cleaner import clean_text


def test_remove_excessive_spaces():
    text = "Hello     world"
    assert clean_text(text) == "Hello world"


def test_normalize_line_endings():
    text = "Hello\r\nworld\r"
    assert clean_text(text) == "Hello\nworld"


def test_remove_excessive_newlines():
    text = "Hello\n\n\n\nworld"
    assert clean_text(text) == "Hello\n\nworld"


def test_remove_null_characters():
    text = "Hello\x00 world"
    assert clean_text(text) == "Hello world"


def test_preserve_paragraphs():
    text = "Paragraph one.\n\nParagraph two."
    assert clean_text(text) == text


def test_empty_string():
    assert clean_text("") == ""


def test_invalid_input():
    with pytest.raises(TypeError):
        clean_text(123)
