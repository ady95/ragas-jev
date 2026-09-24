# Phase 5 검증 리포트: 확률 보정과 LLM Judge 재검증

- 실행일: 2026-09-24
- JEV 모델: `jev-1.13.0`
- LLM Judge 역할: Auditor `gpt-6-luna`, Strong Judge `gpt-6-sol` (처음 계획한 `gpt-6-astra`에서 변경, 3.2절), 비교 기준선 `gpt-6-sol`
- 관련 문서: [구현계획서](../구현계획서.md) Phase 5, [Phase 1](phase1_context_precision.md) · [2](phase2_faithfulness.md) · [3](phase3_context_recall.md) 리포트

## 1. 요약

| 항목 | 결과 |
|---|---|
| 확률 보정 | 질문·언어별 isotonic 보정으로 ECE가 크게 줄었다 (Context Precision 0.184 → 0.008, Recall 0.102 → 0.010, Faithfulness 0.073 → 0.017). 교차 검증 결과다 |
| 보정만으로 | Context Precision κ 0.450 → 0.543으로 **`gpt-6-sol` 단독(0.492)보다 높다.** Context Recall κ 0.763 → 0.818로 `gpt-6-sol`(0.821)과 비슷하다. LLM 호출은 0회 |
| 계획서의 기본 routing (0.85 / 0.60) | **최선이 아니다.** Context Precision에서 Auditor 구간이 오히려 κ를 낮춘다 (보정만 0.543 → 기본 routing 0.531). 샘플당 LLM 호출도 0.7~1.2회로 많다 |
| metric별 권장 정책 | Context Precision: 보정 + confidence < 0.60만 `gpt-6-sol`. Faithfulness: 보정 + confidence < 0.65만 `gpt-6-luna`. Context Recall: 보정만 (routing 없음) |
| 권장 정책의 효과 | κ Context Precision 0.558 / Faithfulness 0.658 / Recall 0.818. **세 metric 모두 LLM 단독과 같거나 낫고**, 샘플당 LLM 호출은 0.40 / 0.34 / 0 |
| 사람 검토 | LLM이 JEV와 반대로 판정한 unit만 검토 대상으로 둔다. 권장 정책에서 unit의 2.1% / 5.6% / 0% |
| 주의 | isotonic 보정은 데이터의 **라벨 기본 비율**까지 학습한다. 같은 문구의 질문이 Faithfulness에서는 위로, Recall에서는 아래로 보정됐다. 서비스 데이터에서는 도메인 라벨로 다시 보정해야 한다 (5.3절) |

**판단**: 계획서의 Exit criteria("escalation 비율과 정확도 향상폭이 정량화되고, 운영 threshold가 근거와 함께 확정된다")를 충족한다. 다만 확정한 threshold는 계획서의 0.85 / 0.60이 아니라 metric별 정책이다.

## 2. 구현한 것

| 구성 요소 | 파일 | 내용 |
|---|---|---|
| 확률 보정 | [calibration.py](../../src/ragas_jev/scoring/calibration.py) | 질문 ID × 언어별 isotonic 보정(PAV). 보정된 p로 confidence를 다시 계산한다. 원래 JEV 값은 `Decision.jev_p`에 남는다 |
| 보정 파일 | [calibration_jev-1.13.0.json](../../src/ragas_jev/data/calibration_jev-1.13.0.json) | 이 리포트의 전체 데이터로 학습. `evaluate`가 기본으로 사용 (`--no-calibration`으로 끔) |
| 언어 판별 | [lang.py](../../src/ragas_jev/lang.py) | 한글 비율 20% 이상이면 `ko`, 아니면 `en` |
| Router | [router.py](../../src/ragas_jev/audit/router.py) | confidence 구간별로 JEV 유지 / Auditor / Strong Judge. metric별 정책(`DEFAULT_METRIC_POLICIES`). LLM 호출이 실패하면 JEV 판정을 유지하고 사람 검토 대상으로 표시 |
| 사람 검토 | [review_queue.py](../../src/ragas_jev/audit/review_queue.py) | `ragas-jev review export`로 CSV 내보내기(마스킹된 텍스트만), `review import`로 라벨 반영 후 재채점 |
| 파이프라인 | [pipeline.py](../../src/ragas_jev/pipeline.py) | JEV 판정 → 보정 → routing → 점수 계산. 결과에 재판정·사람 검토·routing 실패 건수와 LLM 사용량 기록 |
| CLI | [cli.py](../../src/ragas_jev/cli.py) | `evaluate --audit/--no-audit --calibration PATH --no-calibration`, `review export/import` |

사람 검토 기준은 계획서에서 바꿨다. 계획서는 confidence < 0.60을 모두 사람 검토로 보냈지만, 그러면 Context Precision에서만 unit의 6%가 대기열에 쌓인다. **LLM 판정이 JEV와 0.5 기준으로 반대편일 때만** 검토 대상으로 둔다.

## 3. 시뮬레이션 방법

[run_phase5.py](../../benchmark/run_phase5.py). Phase 1~3에서 저장한 판정만 쓰고 API는 호출하지 않는다.

### 3.1 데이터

| metric | 데이터 (라벨) | unit | JEV 문항 |
|---|---|---|---|
| Context Precision | MIRACL-ko dev + 전부 무관 세트, MIRACL-en 213문항 (사람 라벨) | 6,267 chunk | `chunk_relevance.v2` |
| Faithfulness | RAGTruth QA (전문가 라벨), 한국어 합성 hallucination 세트 | 4,193 claim | `claim_support.v1` |
| Context Recall | MIRACL-ko/en 합성 reference, full / partial / none (기대 라벨) | 4,533 claim | `ref_claim_coverage.v1` |

- 모든 unit에 JEV p와 `gpt-6-luna`, `gpt-6-sol` p가 있다. LLM Judge에도 JEV와 **같은 문항 버전**을 줬다. Faithfulness만 이를 위해 두 모델을 v1 문항으로 다시 돌렸다 (응답 512개).
- Answer Relevancy는 unit 단위 라벨이 불완전해 시뮬레이션에서 뺐다 (routing 없음으로 둔다).

### 3.2 Strong Judge를 `gpt-6-sol`로 바꾼 이유

계획서는 Strong Judge를 `gpt-6-astra`로 임시 배정했지만, 성능 근거가 없었다. 또 `gpt-6-astra`는 Phase 2~4의 합성 데이터를 만든 모델이라, 그 데이터에서는 자기가 쓴 답변을 판정하게 된다. 그래서 Phase 1~4에서 실제로 가장 강했던 기준선 `gpt-6-sol`로 바꿨다 (사용자 결정). 기준선과 Strong Judge가 같은 모델이지만, Hybrid에서 `gpt-6-sol`이 판정하는 비율은 6% 이하다.

### 3.3 절차

1. **보정 교차 검증**: 질문 단위 5-fold. 같은 질문의 unit(Recall은 같은 질문의 세 조건 전체)이 학습과 평가에 함께 들어가지 않는다. fold마다 언어별 isotonic 보정을 학습하고, 평가 fold에 적용한 값만 쓴다.
2. **routing grid**: accept ∈ {0, 0.60, …, 0.95, 1.01}, audit ∈ {0, 0.55, …, 0.70} (audit ≤ accept). confidence ≥ accept는 JEV, audit ≤ confidence < accept는 `gpt-6-luna`, confidence < audit는 `gpt-6-sol`의 판정을 쓴다. accept 0은 routing 없음, audit = accept는 Strong Judge만, audit 0은 Auditor만이다.
3. **비용**: 한 샘플의 한 metric에서 재판정할 unit이 하나라도 있으면 LLM 호출 1회로 센다 (Router는 구간별로 한 번에 묶어 보낸다).
4. **권장 정책 선택 규칙**: 보정한 경우의 조합 중, κ가 최고치에서 0.01 이내인 것 가운데 샘플당 LLM 호출이 가장 적은 것.
5. **F1의 기준 클래스**: 각 Phase 리포트와 같다. Context Precision은 "관련", Faithfulness는 "hallucination(미지원)", Recall은 "지원됨"이다. κ는 클래스와 무관하다.

## 4. 결과

### 4.1 보정 (교차 검증)

| metric | 언어 | unit | ECE 보정 전 → 후 | JEV F1 보정 전 → 후 | JEV κ 보정 전 → 후 | gpt-6-luna F1 | gpt-6-sol F1 | conf < 0.85 비율 보정 전 → 후 |
|---|---|---|---|---|---|---|---|---|
| context_precision | en | 2210 | 0.132 → 0.018 | 0.642 → 0.640 | 0.477 → 0.529 | 0.657 | 0.670 | 37% → 46% |
| context_precision | ko | 4057 | 0.212 → 0.005 | 0.530 → 0.592 | 0.422 → 0.536 | 0.530 | 0.555 | 35% → 23% |
| context_precision | all | 6267 | 0.184 → 0.008 | 0.580 → 0.617 | 0.450 → 0.543 | 0.589 | 0.607 | 36% → 31% |
| faithfulness | en | 3440 | 0.080 → 0.018 | 0.667 → 0.617 | 0.599 → 0.563 | 0.663 | 0.632 | 23% → 22% |
| faithfulness | ko | 753 | 0.044 → 0.023 | 0.851 → 0.838 | 0.824 → 0.810 | 0.853 | 0.863 | 9% → 7% |
| faithfulness | all | 4193 | 0.073 → 0.017 | 0.698 → 0.662 | 0.637 → 0.612 | 0.690 | 0.664 | 20% → 19% |
| context_recall | en | 2529 | 0.091 → 0.015 | 0.910 → 0.922 | 0.787 → 0.826 | 0.933 | 0.936 | 13% → 16% |
| context_recall | ko | 2004 | 0.116 → 0.011 | 0.891 → 0.915 | 0.734 → 0.808 | 0.907 | 0.907 | 15% → 21% |
| context_recall | all | 4533 | 0.102 → 0.010 | 0.901 → 0.919 | 0.763 → 0.818 | 0.922 | 0.923 | 14% → 19% |

Context Precision 한국어 수치에는 전부 무관인 세트(1,000 chunk)가 포함돼 Phase 1 리포트 수치와 다르다.

### 4.2 routing 정책 비교

"사람 검토"는 LLM 판정이 JEV와 0.5 기준으로 반대편이라 검토 대상이 되는 unit 비율이다.

#### context_precision (6267 units)

| 정책 (accept / audit) | F1 | κ | Auditor 재판정 | Strong 재판정 | 사람 검토 | 샘플당 LLM 호출 |
|---|---|---|---|---|---|---|
| JEV만 (보정 전) | 0.580 | 0.450 | 0.0% | 0.0% | 0.0% | 0.00 |
| JEV만 (보정 후) | 0.617 | 0.543 | 0.0% | 0.0% | 0.0% | 0.00 |
| 보정 + Strong Judge만 (conf < 0.60) | 0.633 | 0.558 | 0.0% | 5.8% | 2.1% | 0.40 |
| 보정 + 계획서 기본값 (0.85 / 0.60) | 0.626 | 0.531 | 25.3% | 5.8% | 10.1% | 1.18 |
| 보정 없이 계획서 기본값 (0.85 / 0.60) | 0.606 | 0.494 | 28.2% | 7.5% | 10.2% | 1.40 |
| 보정 + κ 최대 (0.70 / 0.70) | 0.646 | 0.565 | 0.0% | 14.3% | 5.6% | 0.64 |
| **권장: 보정 + (0.60 / 0.60)** | 0.633 | 0.558 | 0.0% | 5.8% | 2.1% | 0.40 |
| gpt-6-luna 단독 | 0.589 | 0.477 | 100% | – | – | 1.00 |
| gpt-6-sol 단독 | 0.607 | 0.492 | 100% | – | – | 1.00 |

#### faithfulness (4193 units)

| 정책 (accept / audit) | F1 | κ | Auditor 재판정 | Strong 재판정 | 사람 검토 | 샘플당 LLM 호출 |
|---|---|---|---|---|---|---|
| JEV만 (보정 전) | 0.698 | 0.637 | 0.0% | 0.0% | 0.0% | 0.00 |
| JEV만 (보정 후) | 0.662 | 0.612 | 0.0% | 0.0% | 0.0% | 0.00 |
| 보정 + Strong Judge만 (conf < 0.60) | 0.669 | 0.613 | 0.0% | 5.3% | 3.2% | 0.25 |
| 보정 + 계획서 기본값 (0.85 / 0.60) | 0.702 | 0.640 | 13.9% | 5.3% | 9.1% | 0.71 |
| 보정 없이 계획서 기본값 (0.85 / 0.60) | 0.696 | 0.630 | 16.4% | 3.9% | 5.7% | 0.77 |
| 보정 + κ 최대 (0.70 / 0.00) | 0.713 | 0.658 | 10.3% | 0.0% | 6.4% | 0.38 |
| **권장: 보정 + (0.65 / 0.00)** | 0.712 | 0.658 | 8.8% | 0.0% | 5.6% | 0.34 |
| gpt-6-luna 단독 | 0.690 | 0.621 | 100% | – | – | 1.00 |
| gpt-6-sol 단독 | 0.664 | 0.589 | 100% | – | – | 1.00 |

#### context_recall (4533 units)

| 정책 (accept / audit) | F1 | κ | Auditor 재판정 | Strong 재판정 | 사람 검토 | 샘플당 LLM 호출 |
|---|---|---|---|---|---|---|
| JEV만 (보정 전) | 0.901 | 0.763 | 0.0% | 0.0% | 0.0% | 0.00 |
| JEV만 (보정 후) | 0.919 | 0.818 | 0.0% | 0.0% | 0.0% | 0.00 |
| 보정 + Strong Judge만 (conf < 0.60) | 0.920 | 0.818 | 0.0% | 2.7% | 1.3% | 0.17 |
| 보정 + 계획서 기본값 (0.85 / 0.60) | 0.923 | 0.822 | 15.8% | 2.7% | 4.3% | 0.75 |
| 보정 없이 계획서 기본값 (0.85 / 0.60) | 0.920 | 0.813 | 11.1% | 2.8% | 3.4% | 0.63 |
| 보정 + κ 최대 (0.85 / 0.70) | 0.925 | 0.825 | 11.7% | 6.8% | 4.2% | 0.85 |
| **권장: 보정 + (0.00 / 0.00)** | 0.919 | 0.818 | 0.0% | 0.0% | 0.0% | 0.00 |
| gpt-6-luna 단독 | 0.922 | 0.817 | 100% | – | – | 1.00 |
| gpt-6-sol 단독 | 0.923 | 0.821 | 100% | – | – | 1.00 |

## 5. 해석

### 5.1 보정이 routing보다 효과가 크다

- Context Precision과 Recall에서는 보정만으로 κ가 +0.09, +0.06 올랐다. 가장 좋은 routing이 여기에 더하는 것은 +0.02, +0.01 이하다.
- JEV p가 한쪽으로 치우쳐 있으면 0.5 기준 판정과 `max(p, 1−p)` confidence가 모두 어긋난다. 보정은 두 가지를 한 번에 바로잡는다.

### 5.2 Auditor 구간은 metric에 따라 이득이 다르다

- **Context Precision**: 보정된 JEV가 중간 confidence 구간에서 `gpt-6-luna`보다 정확하다. 그래서 Auditor 구간을 넓힐수록 κ가 떨어진다. 가장 불확실한 구간만 `gpt-6-sol`로 보내는 것이 가장 좋다.
- **Faithfulness**: 반대로 `gpt-6-luna`가 불확실한 claim을 잘 가려낸다. Strong Judge 구간(`gpt-6-sol`)은 이득이 없었다. Phase 2에서도 `gpt-6-sol`이 hallucination 탐지에서 가장 낮았다.
- **Context Recall**: 보정된 JEV가 이미 LLM 단독과 같은 수준이라 재판정할 이유가 없다.
- 계획서처럼 모든 metric에 한 가지 threshold(0.85 / 0.60)를 쓰면, Context Precision에서는 손해이고 LLM 호출은 권장 정책의 2~3배다.

### 5.3 보정은 라벨의 기본 비율을 학습한다

- 질문 문구가 같은 `claim_support.v1`(Faithfulness)과 `ref_claim_coverage.v1`(Recall)이 반대로 보정됐다. JEV p = 0.5가 Faithfulness에서는 한국어 0.79, Recall에서는 0.13으로 바뀐다.
- 원인은 데이터의 기본 비율이다. Faithfulness 데이터는 claim의 85%가 지원되고, Recall 데이터는 none 조건 때문에 미지원 claim이 많다. isotonic 보정은 "이 p에서 실제로 yes인 비율"을 배우므로 기본 비율이 그대로 들어간다.
- 그래서 **Faithfulness에서는 보정 후 0.5 기준 F1이 떨어졌다** (0.698 → 0.662). 소수 클래스(hallucination)가 "지원됨" 쪽으로 밀렸기 때문이다. 하지만 confidence가 정확해져서 Auditor로 보낼 claim을 더 잘 고르게 되고, 보정 + Auditor는 보정 없는 최선(κ 0.638)보다 높은 0.658이 됐다.
- **서비스 데이터의 hallucination 비율이나 관련 passage 비율이 벤치마크와 다르면 이 보정은 맞지 않는다.** 서비스 도메인의 사람 라벨(예: `review export/import`로 모은 라벨)로 다시 보정하는 절차가 필요하다 (7장 제안).

### 5.4 비용과 사람 검토

- 권장 정책에서 샘플당 LLM 호출은 Context Precision 0.40, Faithfulness 0.34, Recall 0이다. LLM 단독(1.0)과 비교하면 호출이 60% 이상 줄고, 판정 품질은 같거나 낫다.
- 호출이 발생한 샘플은 LLM 지연(4~6초)이 더해진다. JEV 판정 자체는 0.3초다.
- 사람 검토 대상은 unit의 2.1%(Context Precision), 5.6%(Faithfulness)다.

### 5.5 end-to-end 확인

합성 스모크 세트 5개로 `ragas-jev evaluate`를 보정과 routing을 켠 채 실행했다. 보정 파일이 자동으로 적용됐고, JEV p 0.94인 한국어 chunk 하나가 보정 후 불확실 구간으로 들어가 `gpt-6-sol`로 재판정됐다 (0.99). 결과 JSON에 `secondary_judge`, `strong_judge`, `calibration`과 escalation·LLM 사용량이 기록됐다.

### 5.6 한계

- LLM Judge 판정은 각 1회 실행이고 결정적이지 않다. 1~2%p 차이는 다시 돌리면 달라질 수 있다.
- JEV도 재요청하면 값이 바뀐다 (Phase 1). 시뮬레이션은 저장된 1회 실행 기준이다.
- Faithfulness 한국어, Recall 전체는 합성 라벨이다. 보정 파일도 이 데이터로 학습했다.
- Answer Relevancy는 검증하지 않았다 (routing 없음).

## 6. 결정 사항

1. 확률 보정을 기본으로 켠다. 보정 파일은 [calibration_jev-1.13.0.json](../../src/ragas_jev/data/calibration_jev-1.13.0.json)이다. JEV 모델 버전이 바뀌면 다시 학습해야 한다.
2. routing은 metric별 정책을 기본으로 쓴다 (`DEFAULT_METRIC_POLICIES`).

| metric | accept / audit | 의미 |
|---|---|---|
| context_precision | 0.60 / 0.60 | confidence < 0.60만 Strong Judge(`gpt-6-sol`) |
| faithfulness | 0.65 / 0 | confidence < 0.65만 Auditor(`gpt-6-luna`) |
| context_recall | 0 / 0 | routing 없음 |
| answer_relevancy | 0 / 0 | routing 없음 (검증 안 됨) |

`RAGAS_JEV_CONF_ACCEPT`와 `RAGAS_JEV_CONF_AUDIT`를 명시적으로 설정하면 모든 metric에 그 값을 쓴다.
3. Strong Judge는 `gpt-6-sol`이다.
4. 사람 검토는 LLM이 JEV와 반대로 판정한 unit만 대상으로 한다.

## 7. 제안 (확인 필요)

1. **Phase 5 통과 판정**: Exit criteria를 충족한다.
2. **도메인 재보정 절차**: 서비스 데이터에서 무작위 샘플을 뽑아 `review export/import`로 사람 라벨을 모으고, 그 라벨로 보정을 다시 학습하는 스크립트를 만든다. 5.3절의 기본 비율 문제를 푸는 가장 직접적인 방법이다.
3. **Phase 6 (Benchmark)**: 계획서의 비교 대상 3종을 같은 데이터로 한 번에 비교한다. (1) 기존 RAGAS / LLM Judge, (2) JEV만, (3) Hybrid(보정 + 권장 routing). 이 리포트의 시뮬레이션 결과가 (2)와 (3)의 unit 단위 추정치다.

## 8. 재현 방법

```bash
# Phase 1~3 결과가 .cache/phase1~3에 있어야 한다. Faithfulness는 LLM Judge를 v1 문항으로 추가 실행
uv run python benchmark/run_phase2.py --datasets ragtruth_qa_test ko_hallu_miracl \
  --configs llm:gpt-6-luna/claim_support.v1/joint llm:gpt-6-sol/claim_support.v1/joint \
  --summary .cache/phase2/summary_v1_luna_sol.json
uv run python benchmark/run_phase5.py            # 시뮬레이션 + 보정 파일 생성
uv run python benchmark/run_phase5.py --no-write-calibration
```
