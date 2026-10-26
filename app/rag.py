"""하이브리드 검색: 키워드 점수 + 벡터 유사도 가중합.

가중치는 LG ELLE 하이브리드 RAG에서 검증한 벡터 0.9 / 키워드 0.1을 쓴다.
벡터 백엔드(Supabase pgvector)는 app/vectorstore.py에 있고, 설정이 없으면
키워드 검색만으로 파이프라인이 끝까지 동작한다.
"""

from __future__ import annotations

import re
from typing import Protocol

VECTOR_WEIGHT = 0.9
KEYWORD_WEIGHT = 0.1


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[가-힣a-zA-Z0-9]+", text.lower()))


def keyword_scores(query: str, chunks: list[dict]) -> dict[str, float]:
    """청크 id별 키워드 점수 = 쿼리 토큰 중 청크에 등장한 비율 (0~1)."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return {}
    return {
        chunk["id"]: len(query_tokens & _tokens(chunk["text"])) / len(query_tokens)
        for chunk in chunks
    }


def keyword_retrieve(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """쿼리와 청크 간 토큰 겹침 점수로 상위 청크를 반환한다."""
    if not _tokens(query):
        return chunks[:top_k]
    scores = keyword_scores(query, chunks)
    scored = [(scores[chunk["id"]], chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored if score > 0][:top_k]


class VectorStore(Protocol):
    """벡터 검색 백엔드가 구현할 인터페이스.

    retrieve는 [{"id": "chunk-001", "score": 0.83, ...}] 형식을 반환한다.
    score는 코사인 유사도(0~1)다.
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ...


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    vector_store: VectorStore | None = None,
    top_k: int = 5,
) -> list[dict]:
    """벡터 0.9 + 키워드 0.1 가중합으로 상위 청크를 반환한다.

    청크 본문은 항상 로컬 chunks에서 가져온다. 벡터 백엔드가 현재 이력서에 없는
    청크 id를 돌려주면 버린다 (인용 id는 현재 이력서 청크만 가리켜야 한다).
    """
    if vector_store is None:
        return keyword_retrieve(query, chunks, top_k=top_k)
    by_id = {chunk["id"]: chunk for chunk in chunks}
    keyword = keyword_scores(query, chunks)
    vector = {
        hit["id"]: float(hit.get("score", 0.0))
        for hit in vector_store.retrieve(query, top_k=top_k)
        if hit.get("id") in by_id
    }
    combined = {
        chunk_id: VECTOR_WEIGHT * vector.get(chunk_id, 0.0)
        + KEYWORD_WEIGHT * keyword.get(chunk_id, 0.0)
        for chunk_id in by_id
    }
    ranked = sorted(
        (chunk_id for chunk_id, score in combined.items() if score > 0),
        key=lambda chunk_id: combined[chunk_id],
        reverse=True,
    )
    return [by_id[chunk_id] for chunk_id in ranked[:top_k]]
