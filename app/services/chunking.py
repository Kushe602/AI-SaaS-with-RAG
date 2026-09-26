"""Split raw document text into overlapping, word-aware chunks."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack words into ~``chunk_size``-character chunks.

    Each chunk carries ``overlap`` characters of tail context into the next one.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            # Carry over the tail words to preserve context across the boundary.
            tail: list[str] = []
            tail_len = 0
            for w in reversed(current):
                if tail_len >= overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + 1
            current = tail
            length = tail_len

    if current:
        chunks.append(" ".join(current))
    return chunks
