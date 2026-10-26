"""청킹 테스트. LLM 없이 실행된다."""

from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_overlap_keeps_continuity():
    text = "a" * 2500
    chunks = chunk_text(text, size=1000, overlap=100)
    assert len(chunks) == 3
    assert chunks[0]["id"] == "chunk-001"
    # 겹침 구간: 두 번째 청크는 첫 번째 청크의 끝 100자와 이어진다.
    assert chunks[1]["text"][:100] == chunks[0]["text"][-100:]


def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", size=100, overlap=100)
