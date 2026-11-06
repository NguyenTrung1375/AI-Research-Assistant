
import pytest

from app.rag.text_splitter.chunker import split_text, chunk_pages


def test_split_text_into_chunks():
    text = " ".join(f"word{i}" for i in range(1, 1201))

    chunks = split_text(text, chunk_size=500, chunk_overlap=80)

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 500
    assert len(chunks[1].split()) == 500
    assert len(chunks[2].split()) == 360


def test_chunk_overlap():
    text = " ".join(f"word{i}" for i in range(1, 601))

    chunks = split_text(text, chunk_size=500, chunk_overlap=80)

    first_words = chunks[0].split()
    second_words = chunks[1].split()

    assert first_words[-80:] == second_words[:80]


def test_empty_text():
    assert split_text("") == "" or split_text("") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello world", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("Hello world", chunk_size=100, chunk_overlap=100)


def test_chunk_pages_preserves_metadata():
    pages = [
        {
            "page_number": 1,
            "text": " ".join(f"word{i}" for i in range(1, 601)),
        },
        {
            "page_number": 2,
            "text": "Content from page two.",
        },
    ]

    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=80)

    assert len(chunks) == 3

    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0

    assert chunks[1]["page_number"] == 1
    assert chunks[1]["chunk_index"] == 1

    assert chunks[2]["page_number"] == 2
    assert chunks[2]["chunk_index"] == 0
