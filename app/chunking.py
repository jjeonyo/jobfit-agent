"""이력서 원문 청킹. LG ELLE에서 검증한 설정(1,000자/중첩 100자)을 기본값으로 쓴다."""

from __future__ import annotations

import re


def chunk_text(text: str, size: int = 1000, overlap: int = 100) -> list[dict]:
    """텍스트를 size자 단위로 overlap자 겹치게 나눈다.

    반환: [{"id": "chunk-001", "text": ...}] — id는 근거 인용의 단위다.
    """
    if size <= 0:
        raise ValueError("size는 0보다 커야 한다")
    if not 0 <= overlap < size:
        raise ValueError("overlap은 0 이상 size 미만이어야 한다")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[dict] = []
    step = size - overlap
    start = 0
    index = 1
    while start < len(text):
        piece = text[start : start + size]
        chunks.append({"id": f"chunk-{index:03d}", "text": piece})
        if start + size >= len(text):
            break
        start += step
        index += 1
    return chunks
