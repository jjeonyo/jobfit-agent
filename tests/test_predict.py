"""평가 예측 스크립트 테스트. 가짜 LLM으로 실행된다."""

import json
from types import SimpleNamespace

from app import agents
from app.graph import build_graph
from eval.predict import predict_case

CHUNKS = [{"id": "chunk-001", "text": "Python FastAPI 백엔드 개발"}]


class FakeLLM:
    def __init__(self, outputs, usage=None):
        self.outputs = list(outputs)
        self.usage = usage

    def invoke(self, prompt):
        return SimpleNamespace(content=self.outputs.pop(0), usage_metadata=self.usage)


def test_predict_case_records_report_and_elapsed(monkeypatch):
    outputs = [
        json.dumps([{"id": "req-01", "category": "필수", "text": "FastAPI 개발 경험"}]),
        json.dumps([{"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"]}]),
    ]
    llm = FakeLLM(outputs, usage={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60})
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    result = predict_case(build_graph(), {"id": "case-01", "posting_text": "공고"}, CHUNKS)

    assert result["id"] == "case-01"
    assert result["matches"][0]["verdict"] == "매칭"
    assert result["llm_calls"] == 2 and result["blocked_claims"] == 0
    assert result["input_tokens"] == 100 and result["output_tokens"] == 20
    assert result["elapsed_sec"] >= 0


def test_predict_case_records_llm_error(monkeypatch):
    llm = FakeLLM(["JSON 아님", "JSON 아님"])
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    result = predict_case(build_graph(), {"id": "case-01", "posting_text": "공고"}, CHUNKS)
    assert "error" in result and result["requirements"] == []
