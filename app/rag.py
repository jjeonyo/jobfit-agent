"""하이브리드 검색의 스캐폴드 구현.

현재는 키워드 검색만 구현돼 있다. 벡터 검색(Supabase pgvector)은
2주 계획 Day 4-6 작업으로, VectorStore 프로토콜에 맞춰 추가한다.
키워드만으로도 파이프라인이 끝까지 동작하는 것이 목적이다.
"""

from __future__ import annotations

import re
from typing import Protocol


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[가-힣a-zA-Z0-9]+", text.lower()))


def keyword_retrieve(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """쿼리와 청크 간 토큰 겹침 점수로 상위 청크를 반환한다."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return chunks[:top_k]
    scored = []
    for chunk in chunks:
        chunk_tokens = _tokens(chunk["text"])
        score = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored if score > 0][:top_k]


class VectorStore(Protocol):
    """벡터 검색 백엔드가 구현할 인터페이스 (Day 4-6에 Supabase로 구현)."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ...


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    vector_store: VectorStore | None = None,
    top_k: int = 5,
) -> list[dict]:
    """키워드 결과와 벡터 결과를 합치고, 청크 id 기준 중복을 제거한다."""
    seen: dict[str, dict] = {}
    for chunk in keyword_retrieve(query, chunks, top_k=top_k):
        seen[chunk["id"]] = chunk
    if vector_store is not None:
        for chunk in vector_store.retrieve(query, top_k=top_k):
            seen.setdefault(chunk["id"], chunk)
    return list(seen.values())[:top_k]
