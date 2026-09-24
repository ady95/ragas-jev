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

## 사람 검토와 도메인 재보정

```bash
# LLM이 JEV와 반대로 판정한 unit을 CSV로 내보내고, 라벨을 반영해 재채점
uv run ragas-jev review export -i results/out.jsonl -o results/review.csv
uv run ragas-jev review import -i results/out.jsonl -l results/review.csv -o results/out_reviewed.jsonl

# 서비스 도메인 라벨로 JEV 확률 보정을 다시 학습 (자세한 절차: docs/도메인_재보정_가이드.md)
uv run ragas-jev calibration sample -r results/out.jsonl -s data.jsonl -o results/label_sheet.csv
uv run ragas-jev calibration fit -l results/label_sheet.csv -o results/domain_calibration.json
uv run ragas-jev evaluate -i data.jsonl -o results/out2.jsonl --calibration results/domain_calibration.json
```

## 벤치마크 의존성

```bash
uv sync --all-groups   # benchmark(ragas, pyarrow), embeddings(sentence-transformers, torch), dev
```

`ragas` 기준선은 Python 3.12 가상환경이 필요하다 (`uv venv --python 3.12`). 자세한 재현 방법은 각 Phase 리포트의 마지막 장에 있다.

## 테스트

```bash
uv run pytest            # 오프라인 테스트 (unit / contract / integration)
uv run pytest -m live    # 실제 JEV API 호출 (합성 데이터)
```

## 현재 상태 (Phase 6)

| 구성 요소 | 상태 |
|---|---|
| JEV Judge (`typesafe-sdk`, fan-out, 캐시) | 구현 |
| 점수 계산 4종 + confidence/uncertainty | 구현 |
| PII Guard | 구현 |
| CLI `healthcheck`, `evaluate` | 구현 |
| Preprocessor | LLM claim/statement 추출 (`--extractor llm`, 기본) + 오프라인 문장 분리 (`--extractor sentence`) |
| 수치 검사 | claim 숫자와 contexts 대조 (진단 정보) |
| 확률 보정 + Router | 구현. 보정 파일 `src/ragas_jev/data/calibration_jev-1.13.0.json`, metric별 routing 정책 |
| 사람 검토 | `ragas-jev review export` / `review import` |
| 도메인 재보정 | `ragas-jev calibration sample` / `calibration fit` ([가이드](docs/도메인_재보정_가이드.md)) |
| Phase 1 검증 | 완료: [리포트](docs/reports/phase1_context_precision.md) |
| Phase 2 검증 | 완료: [리포트](docs/reports/phase2_faithfulness.md) |
| Phase 3 검증 | 완료: [리포트](docs/reports/phase3_context_recall.md) |
| Phase 4 검증 | 완료: [리포트](docs/reports/phase4_answer_relevancy.md) |
| Phase 5 검증 | 완료: [리포트](docs/reports/phase5_routing_calibration.md) |
| Phase 6 벤치마크 | 완료: [리포트](docs/reports/phase6_benchmark.md): RAGAS / LLM Judge / JEV / Hybrid 비교 |
| 후속: 도메인 재보정 검증, reference 기반 Context Precision, RAGAS Answer Relevancy | 완료: [Phase 6 리포트](docs/reports/phase6_benchmark.md) 4.1, 5.1, 5.4절 |
