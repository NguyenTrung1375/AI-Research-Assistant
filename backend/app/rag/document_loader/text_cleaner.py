
import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving paragraph structure.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Collapse 3 or more consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
