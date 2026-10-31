# n8n 분석 기록 (평가셋 수집)

웹 UI나 API로 분석한 공고와 결과를 Google Sheets에 자동으로 쌓는다. 같은 시트에서
정답 라벨을 달고, CSV로 내려받아 `eval/cases/gold.json`을 만든다.
평가셋 20건을 만드는 수고를 줄이는 것이 목적이다 ([개선-기록.md](개선-기록.md) P15).

## 흐름

```
POST /analyze (app) ─응답 후─▶ n8n Webhook ─┬─▶ Split Out ─▶ Google Sheets: postings 탭
                                            └─▶ Split Out ─▶ Google Sheets: predictions 탭

사람: gold 탭에 정답 라벨 작성 ─CSV 내려받기─▶ python -m eval.import_labels ─▶ eval/cases/gold.json
```

## 지키는 것

- n8n은 `/analyze`가 보낸 행을 시트에 추가만 한다. LLM 호출이나 판정 로직을 n8n에 두지
  않는다. 그러면 검증 원장을 거치지 않는 경로가 생긴다 (README 핵심 불변 규칙 2).
- 공고 링크를 받아 페이지를 가져오지 않는다. 공고 크롤링은 범위 밖이다.
- 이력서 본문은 보내지 않는다. 청크 id와 근거 문장만 보낸다.
- 시트 행 형식은 `app/recording.py`에서 만들고 테스트한다 (`tests/test_recording.py`).
  워크플로에는 변환 로직이 없다.

## 1. Google Sheets 준비

새 스프레드시트에 탭 3개를 만들고, 1행에 아래 머리글을 그대로 적는다.

| 탭 | 채우는 쪽 | 1행 머리글 |
| --- | --- | --- |
| `postings` | n8n | `case_id`, `analyzed_at`, `posting_text`, `requirements`, `unverified`, `blocked_claims`, `llm_calls`, `input_tokens`, `output_tokens` |
| `predictions` | n8n | `case_id`, `requirement_id`, `category`, `requirement`, `verdict`, `citations`, `quotes`, `rationale`, `gap_plan` |
| `gold` | 사람 | `case_id`, `gold_id`, `category`, `text`, `verdict`, `citations` |

## 2. n8n 실행

`.env`에 아래를 추가한다. 토큰은 아무도 추측할 수 없는 긴 문자열로 정한다.

```bash
RECORD_WEBHOOK_URL=http://n8n:5678/webhook/jobfit-record
RECORD_WEBHOOK_TOKEN=<임의의 긴 문자열>
```

n8n은 `record` 프로필로 묶여 있어, 프로필을 켤 때만 함께 뜬다.

```bash
docker compose --profile record up --build
```

`http://localhost:5678`에 접속해 n8n 소유자 계정을 만든다.

## 3. 워크플로 가져오기와 자격 증명

1. 워크플로를 가져온다. 둘 중 하나로 한다.
   - 편집기: Workflows → Import from File → `workflows/jobfit-record.json`
   - 명령: `docker compose exec n8n n8n import:workflow --input=/workflows/jobfit-record.json`
2. `Webhook` 노드에 Header Auth 자격 증명을 만들어 연결한다.
   - Name: `X-JobFit-Token`
   - Value: `.env`의 `RECORD_WEBHOOK_TOKEN`과 같은 값
3. Google Sheets 노드 2개(`postings 탭에 추가`, `predictions 탭에 추가`)에 Google 자격 증명을
   연결하고, Document에서 1단계의 스프레드시트를 고른다.
   - 서비스 계정을 쓰면, 스프레드시트를 서비스 계정 이메일에 편집자로 공유해야 한다.
   - Document를 고르지 않으면 n8n이 실행을 거부한다 ("Parameter "Document" is required").
4. 워크플로를 Publish한다 (n8n 2.x에서 활성화에 해당). Publish해야 `/webhook/jobfit-record`가
   열린다. 편집기에서 테스트할 때는 `/webhook-test/jobfit-record`를 쓴다.

## 4. 동작 확인

웹 UI(`http://localhost:8000/`)에서 공고 1건을 분석한다. 그 뒤 다음을 확인한다.

- `postings` 탭에 1행이 추가됐다.
- `predictions` 탭에 요건 수만큼 행이 추가됐다. 검증 원장이 막은 요건은 `verdict`가 `미검증`이다.

행이 안 들어오면 두 곳을 본다.

- `api` 로그의 "분석 기록 전송 실패" 경고
- n8n의 Executions 목록

## 5. 정답 라벨 작성 (`gold` 탭)

모델 결과를 보고 정답을 달면 정답이 모델 쪽으로 기운다. 그래서 순서를 지킨다.

1. `predictions` 탭을 숨긴다.
2. `postings` 탭의 `posting_text`만 보고 요건을 직접 뽑아 `gold` 탭에 적는다.
   - `case_id`: `postings` 탭의 값을 그대로 쓴다.
   - `gold_id`: 공고 안에서만 겹치지 않으면 된다 (예: `g-01`).
   - `text`: 공고 문구를 그대로 옮긴다. 예측 요건과 문장 유사도로 대응시키기 때문이다 (P10).
3. 요건마다 [판정-기준.md](판정-기준.md)대로 `verdict`(매칭/부분 매칭/없음)와
   `citations`(청크 id, 쉼표로 구분)를 적는다. 청크 id는 `GET /chunks`로 확인한다.
   판정을 아직 못 정했으면 `verdict`를 비운다. 그 요건은 판정 지표에서 빠진다.
4. 다 적은 뒤에 `predictions` 탭과 비교한다.

`predictions`의 요건을 `gold`로 복사하지 않는다. 그러면 정답이 예측과 같아져
요건 추출 정확도가 부풀려진다.

## 6. gold.json 만들기

`postings` 탭과 `gold` 탭을 각각 CSV로 내려받는다 (파일 → 다운로드 → CSV, 탭마다 한 번).

```bash
python -m eval.import_labels --postings postings.csv --gold gold.csv --out eval/cases/gold.json
```

- `gold` 행이 없는 공고는 빠진다.
- 판정 기준을 어긴 행이 있으면 행 번호와 이유를 모두 알리고, 파일을 쓰지 않는다.
  - 알 수 없는 판정
  - '매칭/부분 매칭'인데 인용이 없음
  - '없음'인데 인용이 있음
  - 청크 id 형식이 아님 (`chunk-001` 꼴)
  - `postings`에 없는 `case_id`, 중복된 `gold_id`

그다음 측정은 README "평가 방법"대로 `eval.predict` → `eval.run`을 실행한다.

## 한계

- `predictions` 탭은 분석 당시 코드로 낸 결과다. 지표는 `gold.json`으로 `eval.predict`를
  다시 실행해 잰다. `predictions`는 지원 이력과 비교용이다.
- 전송에 실패한 분석은 다시 보내지 않는다. 경고 로그만 남는다.
- 같은 공고를 다시 분석하면 `case_id`가 다른 행이 새로 생긴다 (분석 시각이 들어간다).
  라벨은 그중 하나에만 단다.
- 앱을 Render에 배포하면 `RECORD_WEBHOOK_URL`은 Render에서 닿는 n8n 주소여야 한다.
  로컬 Compose 밖의 n8n 운영은 다루지 않는다. 값을 비워두면 기록하지 않는다.
