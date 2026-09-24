# Phase 4 검증 리포트: Answer Relevancy

- 실행일: 2026-09-24
- JEV 모델: `jev-1.13.0`
- statement 추출: `gpt-6-luna` + `statement_extraction.v1`. 모든 구성이 **같은 statement**를 판정한다
- 비교 LLM Judge: 기준선 `gpt-6-sol`, Auditor `gpt-6-luna` (statement 판정만. LLM Judge는 Score 문항을 지원하지 않아 scale 점수는 JEV만 있다)
- 관련 문서: [구현계획서](../구현계획서.md) Phase 4

Answer Relevancy는 두 점수로 낸다.

| 점수 | 계산 | 결과 필드 |
|---|---|---|
| statement (주 점수) | 답변을 statement로 나눈 뒤 statement마다 "질문과 관련 있는가" Noul, 평균 | `answer_relevancy` |
| scale (보조) | 답변 전체에 4단계 Score 한 번, `score / 3` | `answer_relevancy_scale` |

## 1. 요약

| 항목 | 결과 |
|---|---|
| 좋은 답변 > 부분 답변 (WikiEval) | JEV statement v2 **0.92**, scale **1.00**. `gpt-6-sol` 0.76~0.78, `gpt-6-luna` 0.82. 참고: RAGAS 논문의 answer relevance 사람 일치도 0.78 |
| 무관 문장 찾기 (MIRACL 합성) | 모든 구성이 삽입된 무관 문장을 **100%** 찾았다. 원본 > 무관 문장 삽입 쌍 정확도: JEV 0.95~0.98, `gpt-6-sol` 0.78~0.84 |
| 답하지 않는 답변 | 모든 구성이 0.93~1.00으로 원본보다 낮게 매긴다 |
| **문제 1: statement 문항이 너무 엄격함** | v1은 원본 답변 statement의 50~55%를 "무관"으로 판정했다. **v2에서 21~22%로 줄었다** (LLM v2는 33~39%) |
| **문제 2: 관련성 점수가 정답 여부를 반영함** | scale v1은 틀린 답변을 한국어 0.29, 영어 0.11, WikiEval 0.12 낮게 매겼다. **scale v2에서 0.08 / 0.05 / 0.05로 줄었지만 남아 있다.** statement 점수는 차이가 0.01~0.03으로 정답 여부에 거의 영향받지 않는다 |
| 결정 | 기본 문항을 `statement_relevance.v2`, `answer_relevance_scale.v2`로 바꿨다. 주 점수는 statement 점수를 유지한다 |
| 속도·비용 | JEV p50 0.28초, LLM 3.6~4.7초. statement 추출은 LLM p50 6.5~7.8초 |

**판단**: 기준선 `gpt-6-sol` 대비 JEV가 모든 판별 지표에서 같거나 낫다 (WikiEval 쌍 정확도 +0.14, 무관 문장 삽입 쌍 정확도 +0.14~0.17). Phase 1~3과 같은 "기준선 대비" 기준으로 **통과**다. 계획서의 검증 항목("off-topic statement 주입 샘플에서 해당 statement 식별")도 충족한다.

## 2. 데이터셋

[prepare_relevancy.py](../../benchmark/prepare_relevancy.py)로 만들었다.

| 데이터셋 | 언어 | 샘플 | 변형 | 기대 |
|---|---|---|---|---|
| `relevancy_wikieval` | 영어 | 150 (50문항 × 3) | good / poor / wrong | good > poor, good ≈ wrong |
| `relevancy_miracl_ko` | 한국어 | 369 | base 100 / offtopic 100 / nonanswer 100 / wrong 69 | base > offtopic, base > nonanswer, base ≈ wrong |
| `relevancy_miracl_en` | 영어 | 399 | base 100 / offtopic 100 / nonanswer 100 / wrong 99 | 위와 같음 |

- **WikiEval** ([explodinggradients/WikiEval](https://huggingface.co/datasets/explodinggradients/WikiEval)): RAGAS 논문이 answer relevance 검증에 쓴 데이터다. `answer`(좋은 답변), `poor_answer`(일부만 답하거나 주제에서 벗어난 답변), `ungrounded_answer`(주제는 맞지만 사실이 틀린 답변)를 썼다. **라이선스가 명시돼 있지 않아** 내부 평가에만 쓰고 저장소에 커밋하지 않는다.
- **MIRACL 합성 세트**: Phase 3의 reference 답변을 base로 썼다.
  - offtopic: 다른 질문의 reference 첫 문장을 문장 경계에 삽입 (LLM 호출 없음)
  - wrong: `gpt-6-astra`가 질문에 직접 답하는 사실 하나만 틀리게 바꿈. 바뀐 위치를 확인할 수 없는 경우는 뺐다 (한국어 31개)
  - nonanswer: `gpt-6-astra`가 무관 passage로 같은 주제의 배경 정보만 씀
- base 답변은 Phase 3에서 "모든 passage에서 한 문장씩" 쓰도록 만든 것이다. 그래서 질문에 직접 답하지 않는 **보조 정보 문장**이 섞여 있다. 문제 1(엄격함)을 시험하는 데 적합하다.

## 3. 비교 구성

| 구성 | Judge | statement 문항 | scale 문항 |
|---|---|---|---|
| `jev...v1...v1` | JEV | v1: Does `statement` directly address `question`? | v1: 4단계 (Unrelated … Direct and complete) |
| `jev...v2...v2` | JEV | v2: Is `statement` relevant to answering `question`? Judge relevance only, not whether it is true. + criteria | v2: v1 + "사실 정확성은 평가하지 않는다" + 단계별 설명 |
| `llm-gpt-6-sol...v1` | `gpt-6-sol` | v1 | – |
| `llm-gpt-6-sol...v2` | `gpt-6-sol` | v2 | – |
| `llm-gpt-6-luna...v2` | `gpt-6-luna` | v2 | – |

v2 criteria: true = "질문에 답하거나, 답하는 데 도움이 되는 질문 대상에 대한 정보", false = "다른 대상에 대한 내용이거나 답에 도움이 되지 않는 군더더기".

## 4. 결과

"쌍 정확도"는 같은 질문의 두 변형 중 기대한 쪽을 더 높게 매긴 비율이다 (동점은 0.5). "평균 차"는 기준 변형 점수 − 비교 변형 점수의 평균이다. wrong 비교에서는 쌍 정확도가 0.5 근처이고 평균 차가 0에 가까워야 좋다.

#### relevancy_wikieval (150 samples, statement 6.69개/답변)

| 구성 | 점수 | 평균 (good / poor / wrong) | good_vs_poor: 쌍 정확도 / 평균 차 / 차이 0.1 이내 | good_vs_wrong: 쌍 정확도 / 평균 차 / 차이 0.1 이내 |
|---|---|---|---|---|
| jev.statement_relevance.v1.answer_relevance_scale.v1 | statement | 0.819 / 0.561 / 0.813 | 0.900 / 0.258 / 18.0% | 0.480 / 0.006 / 76.0% |
| jev.statement_relevance.v1.answer_relevance_scale.v1 | scale | 0.942 / 0.461 / 0.824 | 1.000 / 0.481 / 6.0% | 0.690 / 0.119 / 44.0% |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | statement | 0.923 / 0.635 / 0.916 | 0.920 / 0.288 / 22.0% | 0.690 / 0.008 / 80.0% |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | scale | 0.950 / 0.573 / 0.902 | 1.000 / 0.377 / 2.0% | 0.620 / 0.048 / 70.0% |
| llm-gpt-6-sol.statement_relevance.v1.noscale | statement | 0.755 / 0.527 / 0.758 | 0.760 / 0.229 / 20.0% | 0.521 / -0.009 / 43.8% |
| llm-gpt-6-sol.statement_relevance.v2.noscale | statement | 0.860 / 0.575 / 0.868 | 0.780 / 0.284 / 14.0% | 0.650 / -0.009 / 66.0% |
| llm-gpt-6-luna.statement_relevance.v2.noscale | statement | 0.912 / 0.628 / 0.908 | 0.816 / 0.283 / 14.3% | 0.570 / 0.004 / 76.0% |

| 구성 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|
| jev.statement_relevance.v1.answer_relevance_scale.v1 | 0.28 | 0.0050 | 0 |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | 0.27 | 0.0079 | 0 |
| llm-gpt-6-sol.statement_relevance.v1.noscale | 4.31 | – | 2 |
| llm-gpt-6-sol.statement_relevance.v2.noscale | 4.29 | – | 0 |
| llm-gpt-6-luna.statement_relevance.v2.noscale | 3.64 | – | 1 |

#### relevancy_miracl_ko (369 samples, statement 4.84개/답변)

| 구성 | 점수 | 평균 (base / nonanswer / offtopic / wrong) | base_vs_nonanswer: 쌍 정확도 / 평균 차 / 차이 0.1 이내 | base_vs_offtopic: 쌍 정확도 / 평균 차 / 차이 0.1 이내 | base_vs_wrong: 쌍 정확도 / 평균 차 / 차이 0.1 이내 |
|---|---|---|---|---|---|
| jev.statement_relevance.v1.answer_relevance_scale.v1 | statement | 0.503 / 0.090 / 0.379 / 0.480 | 0.980 / 0.413 / 0.0% | 0.970 / 0.124 / 48.0% | 0.638 / 0.027 / 82.6% |
| jev.statement_relevance.v1.answer_relevance_scale.v1 | scale | 0.920 / 0.173 / 0.817 / 0.662 | 0.990 / 0.747 / 1.0% | 0.980 / 0.103 / 52.0% | 0.971 / 0.289 / 17.4% |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | statement | 0.751 / 0.223 / 0.569 / 0.753 | 0.970 / 0.528 / 3.0% | 0.980 / 0.182 / 20.0% | 0.565 / 0.013 / 87.0% |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | scale | 0.816 / 0.206 / 0.672 / 0.748 | 1.000 / 0.611 / 1.0% | 0.980 / 0.144 / 33.0% | 0.935 / 0.084 / 60.9% |
| llm-gpt-6-sol.statement_relevance.v1.noscale | statement | 0.435 / 0.007 / 0.328 / 0.493 | 1.000 / 0.428 / 1.0% | 0.810 / 0.107 / 57.0% | 0.326 / -0.040 / 66.7% |
| llm-gpt-6-sol.statement_relevance.v2.noscale | statement | 0.662 / 0.122 / 0.534 / 0.690 | 0.950 / 0.540 / 5.0% | 0.840 / 0.128 / 39.0% | 0.478 / -0.008 / 71.0% |
| llm-gpt-6-luna.statement_relevance.v2.noscale | statement | 0.685 / 0.157 / 0.537 / 0.726 | 0.950 / 0.528 / 1.0% | 0.855 / 0.148 / 26.0% | 0.442 / -0.030 / 62.3% |

| 구성 | 삽입된 무관 문장 탐지율 | 원본 statement 오탐률 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|---|---|
| jev.statement_relevance.v1.answer_relevance_scale.v1 | 100.0% | 55.0% | 0.28 | 0.0118 | 0 |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | 100.0% | 21.6% | 0.28 | 0.0173 | 0 |
| llm-gpt-6-sol.statement_relevance.v1.noscale | 100.0% | 58.8% | 4.02 | – | 0 |
| llm-gpt-6-sol.statement_relevance.v2.noscale | 100.0% | 33.6% | 4.65 | – | 0 |
| llm-gpt-6-luna.statement_relevance.v2.noscale | 100.0% | 32.6% | 4.13 | – | 0 |

#### relevancy_miracl_en (399 samples, statement 5.57개/답변)

| 구성 | 점수 | 평균 (base / nonanswer / offtopic / wrong) | base_vs_nonanswer: 쌍 정확도 / 평균 차 / 차이 0.1 이내 | base_vs_offtopic: 쌍 정확도 / 평균 차 / 차이 0.1 이내 | base_vs_wrong: 쌍 정확도 / 평균 차 / 차이 0.1 이내 |
|---|---|---|---|---|---|
| jev.statement_relevance.v1.answer_relevance_scale.v1 | statement | 0.500 / 0.102 / 0.385 / 0.485 | 1.000 / 0.398 / 2.0% | 0.960 / 0.115 / 48.0% | 0.652 / 0.020 / 89.9% |
| jev.statement_relevance.v1.answer_relevance_scale.v1 | scale | 0.904 / 0.193 / 0.780 / 0.797 | 1.000 / 0.711 / 1.0% | 0.990 / 0.124 / 36.0% | 0.848 / 0.114 / 63.6% |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | statement | 0.748 / 0.261 / 0.585 / 0.741 | 0.980 / 0.487 / 6.0% | 0.950 / 0.162 / 27.0% | 0.581 / 0.012 / 94.9% |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | scale | 0.830 / 0.227 / 0.668 / 0.788 | 1.000 / 0.603 / 1.0% | 0.960 / 0.162 / 26.0% | 0.753 / 0.047 / 83.8% |
| llm-gpt-6-sol.statement_relevance.v1.noscale | statement | 0.357 / 0.010 / 0.295 / 0.358 | 1.000 / 0.347 / 3.0% | 0.828 / 0.059 / 68.7% | 0.480 / 0.002 / 75.5% |
| llm-gpt-6-sol.statement_relevance.v2.noscale | statement | 0.597 / 0.149 / 0.493 / 0.604 | 0.930 / 0.448 / 9.0% | 0.780 / 0.104 / 47.0% | 0.419 / -0.002 / 67.7% |
| llm-gpt-6-luna.statement_relevance.v2.noscale | statement | 0.664 / 0.150 / 0.557 / 0.678 | 0.940 / 0.514 / 9.0% | 0.800 / 0.106 / 35.0% | 0.470 / -0.008 / 66.7% |

| 구성 | 삽입된 무관 문장 탐지율 | 원본 statement 오탐률 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|---|---|
| jev.statement_relevance.v1.answer_relevance_scale.v1 | 100.0% | 50.1% | 0.28 | 0.0119 | 0 |
| jev.statement_relevance.v2.answer_relevance_scale.v2 | 100.0% | 20.5% | 0.31 | 0.0185 | 0 |
| llm-gpt-6-sol.statement_relevance.v1.noscale | 100.0% | 64.6% | 4.43 | – | 2 |
| llm-gpt-6-sol.statement_relevance.v2.noscale | 100.0% | 38.7% | 4.70 | – | 0 |
| llm-gpt-6-luna.statement_relevance.v2.noscale | 100.0% | 32.5% | 4.15 | – | 0 |

LLM Judge 오류 5건(WikiEval 3, MIRACL 영어 2)은 LLM이 또 다른 JSON 형식(`{"answers": {"<id>": x}}`, `{"results": [...]}`)으로 답해서 생겼다. 파서를 고쳤고([llm_judge.py](../../src/ragas_jev/audit/llm_judge.py)) 결과는 다시 돌리지 않았다. 전체 대비 1% 미만이라 결론에 영향은 없다.

## 5. 해석

### 5.1 statement 문항 v2가 엄격함 문제를 풀었다

- v1("directly address")은 답에 도움이 되는 보조 정보까지 무관으로 판정했다. 원본 답변 statement의 절반 이상이 무관으로 판정됐고, 그 결과 원본의 statement 점수가 0.50에 그쳤다.
- v2는 "답하는 데 도움이 되는가"를 묻고 참·거짓은 따지지 않게 했다. 원본 오탐이 21~22%로 줄었고, 삽입된 무관 문장은 여전히 100% 찾았다. WikiEval 쌍 정확도도 0.90 → 0.92로 올랐다.
- v2에서도 남은 21%의 오탐은 대부분 질문과 느슨하게 연결된 보조 정보다 (예: 질문 "그리스의 수도는?"에 대한 "아테네에서는 1896년과 2004년에 하계 올림픽이 열렸습니다."). RAGAS의 answer relevancy도 불필요한 정보에 감점하므로, 이 정도의 감점은 metric 정의에 맞는 동작으로 본다.

### 5.2 scale 점수의 정답 편향

- scale v1은 틀린 답변을 뚜렷하게 낮게 매겼다 (한국어 평균 차 0.289, 틀린 답변을 더 낮게 매긴 비율 97%). Phase 0 스모크 실행에서 본 문제가 데이터로 확인됐다.
- scale v2(정확성은 평가하지 말라는 문구 추가)로 평균 차가 0.084 / 0.047 / 0.048로 줄었다. 하지만 한국어에서는 여전히 94%의 질문에서 틀린 답변을 더 낮게 매긴다. JEV 문서의 "literal reading" 설명대로, 문구만으로 판단 기준을 완전히 바꾸기는 어렵다.
- statement 점수는 정답 여부에 거의 영향받지 않는다 (평균 차 0.01~0.03). 따라서 **주 점수는 statement 점수로 유지하고, scale은 보조 지표로 둔다.** scale은 좋은 답변과 부분 답변을 가르는 데는 가장 강하다 (WikiEval 1.00).

### 5.3 LLM 대비 JEV

- 무관 문장 삽입 답변을 원본보다 낮게 매기는 비율은 JEV가 0.95~0.98, `gpt-6-sol`이 0.78~0.84로 차이가 크다. LLM은 statement 확률을 거의 0/1로 주기 때문에, 원본에도 오탐 statement가 있으면 원본과 삽입본의 평균이 비슷해진다.
- JEV는 확률이 연속적이라 삽입된 무관 문장 하나가 평균을 확실히 낮춘다. Phase 1~3에서 본 "순위 판별력은 JEV가 좋다"와 같은 경향이다.

## 6. 결정 사항과 제안

**적용한 결정**

1. 기본 문항을 `statement_relevance.v2`, `answer_relevance_scale.v2`로 바꿨다 ([pipeline.py](../../src/ragas_jev/pipeline.py)).
2. 주 점수는 statement 점수(`answer_relevancy`)를 유지하고, scale(`answer_relevancy_scale`)은 보조 지표로 둔다.
3. LLM 응답 파서가 새로 발견된 두 형식을 받도록 고쳤다.

**제안 (확인 필요)**

1. **Phase 4 통과 판정**: 기준선 `gpt-6-sol` 대비 모든 판별 지표에서 같거나 낫다.
2. **scale 정답 편향**: 보조 지표로만 쓰면 당장은 문제가 없다. 없애려면 "답변이 사실이라고 가정하고 평가하라" 같은 문구로 v3를 A/B하거나, scale을 정답 여부와 분리된 두 질문(질문의 모든 부분을 다루는가 / 주제에서 벗어난 내용이 있는가)으로 나누는 방법이 있다.

## 7. 재현 방법

```bash
uv run --group benchmark python benchmark/prepare_relevancy.py wikieval
uv run python benchmark/prepare_relevancy.py miracl --lang ko   # Phase 3의 recall_miracl_ko 필요, gpt-6-astra로 생성
uv run python benchmark/prepare_relevancy.py miracl --lang en
uv run python benchmark/run_phase4.py --datasets relevancy_wikieval --summary .cache/phase4/summary_wikieval.json
uv run python benchmark/run_phase4.py --datasets relevancy_miracl_ko relevancy_miracl_en --summary .cache/phase4/summary_miracl.json
```

원시 결과와 추출 캐시는 `.cache/phase4/`에 저장된다 (git 제외).
