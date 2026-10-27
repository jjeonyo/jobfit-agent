"""API 테스트. OPENAI_API_KEY 없이도 실행되며, 키가 없으면 지어낸
리포트 대신 503을 반환하는지 확인한다."""

import os

os.environ.pop("OPENAI_API_KEY", None)

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_without_api_key_returns_503_not_fabrication():
    response = client.post("/analyze", json={"posting_text": "AI 개발자 채용 공고"})
    assert response.status_code == 503


def test_analyze_empty_text_returns_400():
    response = client.post("/analyze", json={"posting_text": "   "})
    assert response.status_code == 400


def test_index_serves_web_form():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="posting"' in response.text


def test_chunks_returns_profile_chunks():
    response = client.get("/chunks")
    assert response.status_code == 200
    chunks = response.json()
    assert chunks and chunks[0]["id"] == "chunk-001"


def test_analyze_rejects_oversized_posting():
    response = client.post("/analyze", json={"posting_text": "가" * 20_001})
    assert response.status_code == 422
