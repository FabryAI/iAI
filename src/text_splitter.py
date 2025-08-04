from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def smart_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into chunks using RecursiveCharacterTextSplitter, preserving semantic boundaries.

    Args:
        text (str): The input text to split.
        chunk_size (int): Max number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    return splitter.split_text(text)
