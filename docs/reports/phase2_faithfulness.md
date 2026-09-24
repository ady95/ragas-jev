# Phase 2 검증 리포트: Faithfulness

- 실행일: 2026-09-24
- JEV 모델: `jev-1.13.0`
- claim 추출: `gpt-6-luna` + `claim_extraction.v1` ([prompts.py](../../src/ragas_jev/preprocess/prompts.py)). 모든 구성이 **같은 claim**을 판정하도록 추출 결과를 캐시로 공유했다
- 비교 LLM Judge: 기준선 `gpt-6-sol`, Auditor `gpt-6-luna` (`seed=7`, `temperature` 미지정이라 결정적이지 않음)
- 관련 문서: [구현계획서](../구현계획서.md) Phase 2, [Phase 1 리포트](phase1_context_precision.md)

## 1. 요약

| 항목 | 결과 |
|---|---|
| hallucination 탐지 (RAGTruth, 영어 전문가 라벨) | **JEV가 LLM 기준선과 같거나 약간 낫다.** 응답 단위 F1: JEV v1 0.781, `gpt-6-sol` 0.761, `gpt-6-luna` 0.778. claim 단위 AUROC: JEV 0.913, `gpt-6-sol` 0.904 |
| 보정(calibration) | claim 지원 확률의 ECE가 JEV 0.080으로 LLM(0.119~0.130)보다 좋다 |
| 한국어 (합성 라벨) | 모두 높다. 응답 F1: JEV v1 0.958, `gpt-6-sol` 0.952. 합성 세트라 실제보다 쉬운 것으로 보인다 |
| 문항 v1 vs v2 | v2(criteria 추가)는 **나아지지 않았다** (RAGTruth F1 0.781 → 0.772). 입력 토큰은 60% 늘었다. **v1을 기본으로 유지한다** |
| joint vs per_chunk | per_chunk는 **나아지지 않았고** (F1 0.772 → 0.749) 비용은 2.6배, 지연은 약 3배다. **joint를 기본으로 유지한다** |
| 수치 검사 | 숫자가 contexts에 없는 claim은 실제 hallucination 비율이 높다 (RAGTruth 57%, 전체 평균 15%). 하지만 JEV가 이미 대부분 잡아서 판정에 더해도 F1이 오르지 않았다. **진단 정보로만 쓴다** |
| claim 추출 | 응답당 claim 10.8개(RAGTruth) / 4.0개(한국어). 원문 구간 일치 99%. 라벨된 hallucination 구간의 92%(RAGTruth) / 100%(한국어)가 claim으로 잡혔다 |
| 알려진 약점 | ① 출처에 대한 문장("passage에 없다")을 claim에서 빼서, RAGTruth의 틀린 "근거 없음" 주장을 놓친다. ② passage에 없지만 **실제로는 사실인** 문장을 JEV가 "지원됨"으로 판정하는 경우가 있다 |
| 속도 | 판정은 JEV p50 0.3초. **전체 지연은 claim 추출(LLM p50 6~10초)이 결정한다** |

**판단**: Faithfulness에서 JEV는 LLM 기준선(`gpt-6-sol`) 대비 응답 F1 +0.020, claim AUROC +0.009다. Phase 1 기준("기준선 F1 − 0.05 이내, AUROC ≥ 0.85")을 그대로 적용하면 두 데이터셋 모두 통과다. 계획서의 수동 검수 기준(claim 누락률·환각 claim 비율)은 정량 목표가 없어 5장의 관찰로 대신한다.

## 2. 데이터셋

| 데이터셋 | 언어 | 응답 수 | hallucination | 라벨 | 만든 방법 |
|---|---|---|---|---|---|
| `ragtruth_qa_test` | 영어 | 320 | 160 (구간 235개) | 전문가가 답변의 hallucination 문자 구간을 표시 | [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) (MIT) QA test에서 hallucination 있는 응답 160개, 없는 응답 160개 무작위 추출. [prepare_ragtruth.py](../../benchmark/prepare_ragtruth.py) |
| `ko_hallu_miracl` | 한국어 | 192 | 96 | **합성**: 원본 답변은 정확하다고 가정, 수정본에 바뀐 구간 1개 | MIRACL-ko 질문 96개. `gpt-6-astra`가 관련 passage만으로 답변을 쓰고, 사본에서 사실 하나를 바꿈 (숫자 33 / 개체 30 / 문장 추가 33). contexts는 관련 passage + 무관 passage 최대 5개. [prepare_ko_hallu.py](../../benchmark/prepare_ko_hallu.py) |

주의:

- 한국어 세트는 사람이 검수하지 않았다. 원본 답변 자체가 부정확한 사례가 일부 있다 (6.4절).
- "문장 추가" 유형은 생성 모델이 **실제로는 사실인** 문장을 넣는 경우가 많았다 (예: 빌 클린턴 출생일). passage에는 없으므로 hallucination으로 라벨했다. JEV가 contexts만 근거로 판단하는지 시험하는 문항이 된다.
- 공개 데이터(RAGTruth, Wikipedia)와 그 파생물만 사용했다. 국외 API 전송 제약은 없다.

## 3. 방법

- 파이프라인: `Evaluator`의 `faithfulness`만 실행 ([run_phase2.py](../../benchmark/run_phase2.py)).
- claim 라벨: claim의 `source_span`이 라벨된 hallucination 구간과 겹치면 hallucinated. 원문에서 구간을 찾지 못한 claim(1%)은 claim 단위 지표에서 뺐다.
- 응답 단위 판정: claim 중 하나라도 p < 0.5이면 "hallucination 있음". 연속 점수는 `1 − faithfulness`(mean)와 `1 − min p`(min)를 봤다.
- 비교 구성 (모두 같은 claim):

| 구성 | Judge | 문항 | contexts |
|---|---|---|---|
| `jev.claim_support.v1.joint` | JEV | v1: Is `claim` fully supported by `contexts`? | 한 요청에 전체 |
| `jev.claim_support.v2.joint` | JEV | v2: v1 + criteria (숫자·날짜·이름까지 모두 뒷받침돼야 true) | 한 요청에 전체 |
| `jev.claim_support.v2.per_chunk` | JEV | v2 | chunk마다 요청, claim별 max |
| `llm-gpt-6-sol...` | `gpt-6-sol` | v2 | 한 요청에 전체 |
| `llm-gpt-6-luna...` | `gpt-6-luna` | v2 | 한 요청에 전체 |

## 4. 결과

열 설명: "응답 P / R / F1"은 claim 중 하나라도 p < 0.5면 hallucination으로 본 응답 단위 판정이다. "+수치검사 F1"은 숫자 불일치 claim도 hallucination으로 친 경우다. "claim P / R / F1"은 p < 0.5를 hallucinated claim 예측으로 본 claim 단위 판정이다.

#### ragtruth_qa_test (320 responses)

추출: claim 10.84개/응답, 원문 구간 일치 0.992, 라벨 구간 커버리지 0.919, claim 0개 응답 6개

| 구성 | 응답 AUROC (mean) | 응답 AUROC (min) | 응답 P / R / F1 (claim p<0.5 존재) | +수치검사 F1 | claim AUROC | claim P / R / F1 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|---|---|---|---|---|---|
| jev.claim_support.v1.joint | 0.812 | 0.843 | 0.709 / 0.869 / 0.781 | 0.777 | 0.913 | 0.597 / 0.754 / 0.667 | 0.30 | 0.0142 | 0 |
| jev.claim_support.v2.joint | 0.804 | 0.842 | 0.670 / 0.912 / 0.772 | 0.770 | 0.913 | 0.540 / 0.785 / 0.640 | 0.31 | 0.0228 | 0 |
| jev.claim_support.v2.per_chunk | 0.804 | 0.832 | 0.648 / 0.887 / 0.749 | 0.749 | 0.912 | 0.533 / 0.793 / 0.638 | 0.86 | 0.0601 | 0 |
| llm-gpt-6-sol.claim_support.v2.joint | 0.810 | 0.821 | 0.641 / 0.938 / 0.761 | 0.759 | 0.904 | 0.488 / 0.843 / 0.618 | 6.13 | – | 0 |
| llm-gpt-6-luna.claim_support.v2.joint | 0.818 | 0.797 | 0.674 / 0.919 / 0.778 | 0.776 | 0.907 | 0.531 / 0.857 / 0.655 | 4.68 | – | 0 |

#### ko_hallu_miracl (192 responses)

추출: claim 3.96개/응답, 원문 구간 일치 0.989, 라벨 구간 커버리지 1.000, claim 0개 응답 0개

| 구성 | 응답 AUROC (mean) | 응답 AUROC (min) | 응답 P / R / F1 (claim p<0.5 존재) | +수치검사 F1 | claim AUROC | claim P / R / F1 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|---|---|---|---|---|---|
| jev.claim_support.v1.joint | 0.961 | 0.982 | 0.958 / 0.958 / 0.958 | 0.939 | 0.944 | 0.813 / 0.893 / 0.851 | 0.29 | 0.0125 | 0 |
| jev.claim_support.v2.joint | 0.960 | 0.984 | 0.939 / 0.958 / 0.948 | 0.929 | 0.942 | 0.783 / 0.902 / 0.838 | 0.31 | 0.0143 | 0 |
| jev.claim_support.v2.per_chunk | 0.965 | 0.985 | 0.968 / 0.948 / 0.958 | 0.948 | 0.945 | 0.840 / 0.893 / 0.866 | 1.48 | 0.0375 | 0 |
| llm-gpt-6-sol.claim_support.v2.joint | 0.967 | 0.984 | 0.968 / 0.938 / 0.952 | 0.944 | 0.947 | 0.846 / 0.884 / 0.865 | 3.15 | – | 0 |
| llm-gpt-6-luna.claim_support.v2.joint | 0.961 | 0.975 | 0.957 / 0.938 / 0.947 | 0.933 | 0.942 | 0.838 / 0.875 / 0.856 | 2.55 | – | 1 |

| 구성 | 탐지율: entity | 탐지율: fabricated | 탐지율: numeric |
|---|---|---|---|
| jev.claim_support.v1.joint | 100.0% | 87.9% | 100.0% |
| jev.claim_support.v2.joint | 100.0% | 87.9% | 100.0% |
| jev.claim_support.v2.per_chunk | 96.7% | 90.9% | 97.0% |
| llm-gpt-6-sol.claim_support.v2.joint | 96.7% | 90.9% | 93.9% |
| llm-gpt-6-luna.claim_support.v2.joint | 96.7% | 87.9% | 97.0% |

#### 추가 지표

| 데이터셋 | 구성 | claim ECE (지원 확률) | 응답 AUROC (수치 보정 점수) | 입력 토큰 | p95 지연(s) |
|---|---|---|---|---|---|
| ragtruth_qa_test | jev v1 joint | **0.080** | 0.801 | 338,913 | 0.57 |
| ragtruth_qa_test | jev v2 joint | 0.104 | 0.794 | 543,525 | 0.41 |
| ragtruth_qa_test | jev v2 per_chunk | 0.099 | 0.794 | 1,430,295 | 1.09 |
| ragtruth_qa_test | gpt-6-sol | 0.130 | 0.803 | 479,676 | 12.10 |
| ragtruth_qa_test | gpt-6-luna | 0.119 | 0.810 | 479,676 | 10.57 |
| ko_hallu_miracl | jev v1 joint | 0.044 | 0.942 | 296,506 | 0.45 |
| ko_hallu_miracl | jev v2 joint | 0.055 | 0.942 | 341,405 | 0.47 |
| ko_hallu_miracl | jev v2 per_chunk | 0.038 | 0.948 | 892,673 | 1.73 |
| ko_hallu_miracl | gpt-6-sol | 0.036 | 0.947 | 274,071 | 7.50 |
| ko_hallu_miracl | gpt-6-luna | 0.041 | 0.946 | 272,621 | 7.13 |

| 데이터셋 | 숫자 불일치 claim | 그중 실제 hallucinated | 전체 claim 중 hallucinated 비율 |
|---|---|---|---|
| ragtruth_qa_test | 98 | 56 (57%) | 517 / 3,468 (15%) |
| ko_hallu_miracl | 53 | 45 (85%) | 112 / 761 (15%) |

claim 추출 비용: RAGTruth LLM 호출 314회(p50 9.8초), 한국어 192회(p50 6.5초). RAGTruth는 스모크 실행 때 캐시된 6개를 재사용했다. RAGTruth 응답 2개는 claim이 30개를 넘었다(44, 34개).

## 5. claim 추출 품질

### 5.1 잘 되는 것

- 설계 문서 예시("A사는 2025년에 매출 3조원을 기록했고 영업이익은 5,000억원으로 전년 대비 20% 증가했다")가 문서에 적힌 그대로 3개 claim으로 나뉜다.
- 대명사와 생략된 주어를 채운다 ("이후 스위스 특허청에서 근무했다" → "아인슈타인은 스위스 특허청에서 근무했다").
- 인사, 답변 거절, "Based on the passages" 같은 사실 없는 문장은 뺀다.
- 원문 구간(`source_span`)이 답변에서 그대로 찾아지는 비율은 99%다.

### 5.2 놓친 hallucination 구간 (RAGTruth 19 / 235개)

| 원인 | 개수 | 예 |
|---|---|---|
| 출처에 대한 문장을 claim에서 뺌 (프롬프트 규칙 5) | 5 | "Passage 3 does not provide instructions for folding a quilt." (실제로는 있음, Evident Conflict) |
| "Therefore, …" 추론 문장을 hedge로 보고 뺌 | 3 | "Therefore, the current temperature in Bucharest is 22°C (72°F)." |
| 세부 정보·목록 항목 누락 | 약 11 | 전화번호 "1-800-829-1954", 목록으로 나열된 효과들 |

- claim 0개 응답 6개 중 2개가 hallucination 응답이었다. 둘 다 "passage에 없다"는 식의 틀린 출처 주장이 hallucination이었다.
- 절차형 답변 하나("Check if the door stops are properly aligned…")에서 claim이 0개였다. 다른 절차형 답변에서는 단계를 잘 뽑았으므로 추출이 일관되지 않은 사례다.

`claim_extraction.v2`를 제안한다 (7장). 출처에 대한 주장("passage X에는 Y가 없다")도 contexts로 검증할 수 있으므로 claim에 포함하고, "Therefore" 추론과 목록 항목·연락처 같은 세부 정보를 빠뜨리지 않도록 규칙을 추가한다.

### 5.3 추출 오류로 생긴 오탐 (한국어)

- "줄기세포는 암 줄기세포를 구별할 방법을 찾기 위해 연구한다": 주어를 잘못 채웠다 (원래 주어는 연구자). JEV가 0.38로 판정해 정상 답변을 hallucination으로 잡았다.

## 6. 해석

### 6.1 Phase 1과 달리 JEV가 LLM보다 낫다

- Context Precision(Phase 1)에서는 JEV가 `gpt-6-sol`보다 F1이 0.035 낮았다. Faithfulness에서는 반대로 F1 +0.020, claim AUROC +0.009이고 ECE는 크게 좋다.
- claim 하나를 contexts와 대조하는 판단은 JEV 문서가 말하는 "좁고 구조화된 판단"에 가깝다. passage의 관련성처럼 정의가 넓은 판단보다 JEV에 잘 맞는 것으로 보인다.

### 6.2 문항 v2와 per_chunk는 이득이 없다

- v2 criteria는 판정을 엄격하게 만들어 recall을 올리고(0.869 → 0.912) precision을 떨어뜨렸다(0.709 → 0.670). F1은 소폭 낮아졌고, criteria가 question마다 붙어 입력 토큰이 60% 늘었다.
- per_chunk는 여러 chunk에 걸친 claim을 지원하지 못하고(claim별 max만 봄), 요청 수가 chunk 수만큼 늘어난다. 두 데이터셋 모두 contexts가 짧아(RAGTruth 약 1,200자) joint에서 distractor 문제가 드러나지 않았다. 긴 contexts에서는 다시 확인해야 한다.

### 6.3 수치 검사

- 숫자가 contexts에 없는 claim은 hallucination일 가능성이 높다 (RAGTruth 57%, 한국어 85%). 신호로서는 유효하다.
- 하지만 JEV가 그런 claim을 이미 대부분 낮게 판정해서, 판정에 더하면 오탐만 늘었다 (F1 −0.004 ~ −0.019). 표기 차이(단위 환산, 반올림 등)로 생기는 오탐이 원인으로 보인다.
- 따라서 점수에는 반영하지 않고, unit별 `checks.numeric`과 `*_numeric_mismatch_ratio`로 진단 정보만 남긴다.

### 6.4 JEV가 세상 지식으로 판단하는 경우

- 한국어 세트에서 JEV(v1)가 놓친 4건은 모두 "문장 추가" 유형이었고, 추가된 문장이 실제로 사실이었다 ("다저 스타디움은 1962년에 개장했습니다" 0.98, "독일의 폴란드 침공을 계기로 영국과 프랑스는 1939년 9월 3일 독일에 선전포고했습니다" 0.97).
- faithfulness는 "contexts에 근거가 있는가"를 묻는 것이므로 이 경우는 오판이다. LLM Judge도 같은 유형의 탐지율이 88~91%로 비슷한 한계를 보였다.
- 보완 후보: "Your own knowledge is not evidence" 같은 criteria를 넣은 v3를 시험한다. 단 v2처럼 criteria가 precision을 떨어뜨릴 수 있어 A/B가 필요하다.
- 반대로 오탐 4건 중 일부는 데이터 문제였다. "증기기관차를 처음 발명한 사람은 스티븐슨이다"는 contexts에 트레비식도 나와 "처음"이 불확실한데, 합성 과정에서 정확한 답변으로 라벨됐다.

### 6.5 지연 시간

- JEV 판정은 응답당 0.3초(p50)이고, LLM 판정은 5~6초다.
- 하지만 claim 추출이 응답당 6~10초 걸려 **전체 지연은 추출기가 결정한다.** 추출 결과는 캐시되므로 같은 답변을 다시 평가할 때는 판정 시간만 든다.

## 7. 결정 사항과 제안

**적용한 결정**

1. claim 문항은 `claim_support.v1`, contexts 방식은 `joint`를 기본으로 유지한다 (코드 기본값 그대로).
2. 수치 검사는 진단 정보로만 쓰고 점수에 반영하지 않는다.

**제안 (확인 필요)**

1. **Phase 2 통과 판정**: Phase 1과 같은 기준(기준선 F1 − 0.05 이내, AUROC ≥ 0.85)을 적용하면 통과다.
2. **`claim_extraction.v2`**: 5.2절의 누락 원인 3가지를 반영한 프롬프트를 만들고 RAGTruth로 A/B한다. 추출만 다시 하면 되므로 비용이 작다 (LLM 약 320회, JEV 약 $0.02).
3. **`claim_support.v3`**: 세상 지식을 근거로 쓰지 않게 하는 문구를 시험한다 (6.4절).
4. **긴 contexts 검증**: 두 데이터셋 모두 contexts가 짧다. Phase 3 이후 긴 문서 데이터로 joint와 per_chunk를 다시 비교한다.

## 8. 재현 방법

```bash
uv run python benchmark/prepare_miracl.py --lang ko            # 한국어 합성 세트의 원천
uv run --group benchmark python benchmark/prepare_ragtruth.py --per-class 160
uv run python benchmark/prepare_ko_hallu.py --queries 100      # gpt-6-astra로 생성 (실행마다 결과가 다름)
uv run --group benchmark python benchmark/run_phase2.py --datasets ragtruth_qa_test --summary .cache/phase2/summary_ragtruth.json
uv run python benchmark/run_phase2.py --datasets ko_hallu_miracl --summary .cache/phase2/summary_ko.json
uv run python benchmark/run_phase2.py --render-only --summary .cache/phase2/summary_ragtruth.json
```

원시 결과와 추출 캐시는 `.cache/phase2/`에 저장된다 (git 제외). 한국어 합성 세트는 생성 모델 출력이 결정적이지 않아 다시 만들면 달라진다.
