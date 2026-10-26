"""이력서 원문 로딩. 민감 정보(전화번호·상세주소)는 이 파일에 넣지 않는다."""

from __future__ import annotations

from pathlib import Path

from app.chunking import chunk_text

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "profile.md"


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> list[dict]:
    """data/profile.md를 읽어 근거 청크 목록으로 반환한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"이력서 원문이 없습니다: {path} — data/profile.md 템플릿에 "
            "STAR 이력서 원문을 붙여넣으세요. 전화번호·상세주소는 넣지 마세요."
        )
    return chunk_text(path.read_text(encoding="utf-8"))
