
def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    Split text into overlapping chunks based on word count.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    """
    Split extracted PDF pages into chunks while preserving metadata.
    """

    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = split_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })

    return all_chunks

def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    Split text into overlapping chunks based on word count.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = text.split()

    if not words:
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    """
    Split extracted PDF pages into chunks while preserving metadata.
    """

    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = split_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })

    return all_chunks
