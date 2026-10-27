"""하이브리드 검색·Supabase 벡터 스토어 테스트. 네트워크 없이 실행된다."""

import json

import httpx

from app.rag import hybrid_retrieve, keyword_retrieve
from app.vectorstore import SupabaseVectorStore, get_vector_store

CHUNKS = [
    {"id": "chunk-001", "text": "Python FastAPI 백엔드"},
    {"id": "chunk-002", "text": "Flutter 클라이언트 화면"},
    {"id": "chunk-003", "text": "Docker Compose 배포"},
]


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k=5):
        return self.hits[:top_k]


def test_keyword_only_without_vector_store():
    assert [c["id"] for c in hybrid_retrieve("FastAPI", CHUNKS)] == ["chunk-001"]
    assert hybrid_retrieve("FastAPI", CHUNKS) == keyword_retrieve("FastAPI", CHUNKS)


def test_vector_weight_dominates_keyword():
    # 키워드는 chunk-001만 맞지만(0.1 × 1.0), 벡터 유사도가 높은 chunk-002가 앞선다(0.9 × 0.8).
    store = FakeVectorStore([{"id": "chunk-002", "score": 0.8}, {"id": "chunk-001", "score": 0.05}])
    ranked = hybrid_retrieve("FastAPI", CHUNKS, store, top_k=2)
    assert [c["id"] for c in ranked] == ["chunk-002", "chunk-001"]


def test_vector_hits_outside_current_profile_are_dropped():
    store = FakeVectorStore([{"id": "chunk-404", "score": 0.99, "text": "지워진 청크"}])
    ranked = hybrid_retrieve("FastAPI", CHUNKS, store)
    assert [c["id"] for c in ranked] == ["chunk-001"]
    assert ranked[0]["text"] == "Python FastAPI 백엔드"  # 본문은 로컬 청크에서 가져온다


def _store(handler, calls):
    def record(request):
        calls.append(request)
        return handler(request)

    return SupabaseVectorStore(
        "https://example.supabase.co",
        "service-key",
        embed_documents=lambda texts: [[float(i)] for i, _ in enumerate(texts)],
        embed_query=lambda text: [0.5],
        client=httpx.Client(transport=httpx.MockTransport(record)),
    )


def test_supabase_sync_upserts_once_and_prunes_stale_ids():
    calls = []
    store = _store(lambda request: httpx.Response(201), calls)
    store.sync(CHUNKS[:2])
    store.sync(CHUNKS[:2])  # 내용이 같으면 다시 임베딩하지 않는다

    assert [request.method for request in calls] == ["POST", "DELETE"]
    upsert, prune = calls
    assert upsert.url.path == "/rest/v1/profile_chunks"
    assert upsert.headers["Prefer"] == "resolution=merge-duplicates"
    assert upsert.headers["apikey"] == "service-key"
    rows = json.loads(upsert.content)
    assert [row["id"] for row in rows] == ["chunk-001", "chunk-002"]
    assert prune.url.params["id"] == "not.in.(chunk-001,chunk-002)"


def test_supabase_retrieve_maps_similarity_to_score():
    calls = []
    rows = [{"id": "chunk-002", "content": "Flutter 클라이언트 화면", "similarity": 0.7}]
    store = _store(lambda request: httpx.Response(200, json=rows), calls)

    assert store.retrieve("모바일 앱", top_k=3) == [
        {"id": "chunk-002", "text": "Flutter 클라이언트 화면", "score": 0.7}
    ]
    assert calls[0].url.path == "/rest/v1/rpc/match_profile_chunks"
    assert json.loads(calls[0].content) == {"query_embedding": [0.5], "match_count": 3}


def test_supabase_retrieve_failure_falls_back_to_empty():
    store = _store(lambda request: httpx.Response(500), [])
    assert store.retrieve("모바일 앱") == []


def test_vector_store_disabled_without_supabase_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert get_vector_store(CHUNKS) is None
