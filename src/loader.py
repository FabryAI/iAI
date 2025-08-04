import fitz  # PyMuPDF
from typing import List

def load_pdf_text(path: str) -> str:
    """
    Loads the entire text content from a PDF file.

    Args:
        path (str): Path to the PDF file.

    Returns:
        str: Full text content extracted from the PDF.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Splits a long text into smaller chunks of fixed size.

    Args:
        text (str): Full input text.
        chunk_size (int): Maximum length of each chunk.

    Returns:
        List[str]: List of text chunks.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
