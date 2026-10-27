"""Supabase pgvector 벡터 검색 백엔드 (rag.VectorStore 구현).

SUPABASE_URL, SUPABASE_KEY가 모두 설정돼 있을 때만 쓴다. 테이블과 검색 함수는
db/supabase.sql로 만든다. SUPABASE_KEY는 service_role 키이며 서버에서만 쓴다.

이력서 청크는 프로세스당 한 번, 내용이 바뀌었을 때만 임베딩해 업서트한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable

import httpx
import openai

from app.rag import VectorStore

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536차원, db/supabase.sql과 맞춘다
TABLE = "profile_chunks"
MATCH_FUNCTION = "match_profile_chunks"

EmbedDocuments = Callable[[list[str]], list[list[float]]]
EmbedQuery = Callable[[str], list[float]]


class SupabaseVectorStore:
    def __init__(
        self,
        url: str,
        key: str,
        embed_documents: EmbedDocuments,
        embed_query: EmbedQuery,
        client: httpx.Client | None = None,
    ) -> None:
        self._rest = url.rstrip("/") + "/rest/v1"
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self._embed_documents = embed_documents
        self._embed_query = embed_query
        self._client = client or httpx.Client(timeout=30)
        self._synced_hash: str | None = None

    def sync(self, chunks: list[dict]) -> None:
        """현재 이력서 청크를 업서트하고, 더 이상 없는 청크 id는 지운다."""
        digest = hashlib.sha256(
            json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if digest == self._synced_hash:
            return
        if chunks:
            embeddings = self._embed_documents([chunk["text"] for chunk in chunks])
            rows = [
                {"id": chunk["id"], "content": chunk["text"], "embedding": embedding}
                for chunk, embedding in zip(chunks, embeddings)
            ]
            response = self._client.post(
                f"{self._rest}/{TABLE}",
                headers={**self._headers, "Prefer": "resolution=merge-duplicates"},
                json=rows,
            )
            response.raise_for_status()
        ids = ",".join(chunk["id"] for chunk in chunks)
        response = self._client.delete(
            f"{self._rest}/{TABLE}",
            headers=self._headers,
            params={"id": f"not.in.({ids})"},
        )
        response.raise_for_status()
        self._synced_hash = digest

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """실패하면 경고를 남기고 빈 목록을 반환한다 (해당 쿼리는 키워드 점수로만 순위)."""
        try:
            response = self._client.post(
                f"{self._rest}/rpc/{MATCH_FUNCTION}",
                headers=self._headers,
                json={"query_embedding": self._embed_query(query), "match_count": top_k},
            )
            response.raise_for_status()
        except (httpx.HTTPError, openai.OpenAIError) as exc:
            logger.warning("벡터 검색 실패, 키워드 점수로만 진행: %s", exc)
            return []
        return [
            {"id": row["id"], "text": row["content"], "score": row["similarity"]}
            for row in response.json()
        ]


_store: SupabaseVectorStore | None = None


def get_vector_store(chunks: list[dict]) -> VectorStore | None:
    """Supabase가 설정돼 있으면 청크를 동기화한 벡터 스토어를, 아니면 None을 반환한다.

    동기화에 실패하면 경고를 남기고 None을 반환해 키워드 검색으로 진행한다.
    인용 검증은 검색 방식과 무관하게 검증 원장이 한다.
    """
    global _store
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not (url and key):
        return None
    if _store is None:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _store = SupabaseVectorStore(
            url, key, embeddings.embed_documents, embeddings.embed_query
        )
    try:
        _store.sync(chunks)
    except (httpx.HTTPError, openai.OpenAIError) as exc:
        logger.warning("Supabase 동기화 실패, 키워드 검색으로 진행: %s", exc)
        return None
    return _store
