# RAGAS-JEV

JEV(TypeSafe System One)를 Primary Judge로 쓰는 RAGAS 방식의 RAG 평가 도구입니다.

> LLM은 문제를 나누고, JEV는 판단하고, 프로그램은 점수를 계산한다.

- 설계: [docs/RAGAS-JEV_설계방향.md](docs/RAGAS-JEV_설계방향.md)
- 구현계획: [docs/구현계획서.md](docs/구현계획서.md)

## 데이터 보호

LLM 프록시(OpenAI)와 JEV(TypeSafe)는 모두 국외 서비스입니다. 외부로 나가는 모든 텍스트는 PII Guard를 거쳐 마스킹되지만, **실제 고객 데이터는 보안·법무 검토 전까지 사용하지 않습니다.** 공개/합성 데이터로만 평가하세요.

## 설치

```bash
uv sync
cp .env.example .env   # TYPESAFE_API_KEY 등 입력
```

## 사용법

```bash
# 프록시·JEV 연결 확인 (합성 입력만 전송)
uv run ragas-jev healthcheck

# 평가 (결과는 샘플당 1줄 JSONL로 append, 재실행 시 완료된 sample_id는 건너뜀)
uv run ragas-jev evaluate -i benchmark/datasets/smoke_ko.jsonl -o results/smoke.jsonl
uv run ragas-jev evaluate -i data.jsonl -o results/out.jsonl --metrics context_precision,faithfulness
uv run ragas-jev evaluate -i data.jsonl -o results/out.jsonl --judge mock   # API 호출 없이
```

입력 JSONL 한 줄의 형식:

```json
{"sample_id": "q-1", "question": "...", "answer": "...", "contexts": ["...", "..."], "reference": "..."}
```

`reference`가 없으면 Context Recall은 계산하지 않고 `status=partial`로 기록합니다.

## 테스트

```bash
uv run pytest            # 오프라인 테스트 (unit / contract / integration)
uv run pytest -m live    # 실제 JEV API 호출 (합성 데이터)
```

## 현재 상태 (Phase 4)

| 구성 요소 | 상태 |
|---|---|
| JEV Judge (`typesafe-sdk`, fan-out, 캐시) | 구현 |
| 점수 계산 4종 + confidence/uncertainty | 구현 |
| PII Guard | 구현 |
| CLI `healthcheck`, `evaluate` | 구현 |
| Preprocessor | LLM claim/statement 추출 (`--extractor llm`, 기본) + 오프라인 문장 분리 (`--extractor sentence`) |
| 수치 검사 | claim 숫자와 contexts 대조 (진단 정보) |
| LLM Judge (Phase 1 비교용) | 구현. Auditor/Router 연결은 Phase 5 |
| Phase 1 검증 | 완료: [리포트](docs/reports/phase1_context_precision.md) |
| Phase 2 검증 | 완료: [리포트](docs/reports/phase2_faithfulness.md) |
| Phase 3 검증 | 완료: [리포트](docs/reports/phase3_context_recall.md) |
| Phase 4 검증 | 완료: [리포트](docs/reports/phase4_answer_relevancy.md) |
