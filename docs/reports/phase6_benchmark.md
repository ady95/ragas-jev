# Phase 6 벤치마크 리포트: RAGAS vs LLM Judge vs JEV vs Hybrid

- 실행일: 2026-09-24
- 관련 문서: [구현계획서](../구현계획서.md) Phase 6, [설계방향](../RAGAS-JEV_설계방향.md) 17장, Phase [1](phase1_context_precision.md) · [2](phase2_faithfulness.md) · [3](phase3_context_recall.md) · [4](phase4_answer_relevancy.md) · [5](phase5_routing_calibration.md) 리포트

## 1. 요약

설계방향 문서가 정한 비교 대상을 같은 데이터로 비교했다. 기존 RAGAS와 LLM 기반 Judge, JEV만 쓰는 평가, Hybrid RAGAS-JEV다.

| metric | 가장 좋은 시스템 | Hybrid의 위치 | 비고 |
|---|---|---|---|
| Context Precision | **Hybrid** (사람 라벨과의 Spearman 한국어 0.681 / 영어 0.567) | 1위 | RAGAS는 정의가 다른 `ContextRelevance`로만 비교 가능 (0.111 / 0.253) |
| Faithfulness | **Hybrid** (RAGTruth AUROC 0.830, F1 0.790) | 1위 | RAGAS 0.790 / 0.752, LLM Judge 0.821 / 0.764. 한국어는 네 시스템 모두 0.94 이상 |
| Context Recall | **RAGAS** (Pearson 0.909 / 0.944) | 2~3위 (0.903 / 0.938) | 정답을 reference 문장 단위로 만들어 RAGAS 방식에 유리하다 |
| Answer Relevancy | **JEV = Hybrid** (WikiEval 쌍 정확도 0.92) | 1위 | RAGAS는 임베딩 API가 없어 제외, LLM Judge 0.78 |
| 재현성 (Faithfulness) | **JEV / Hybrid** | 1위 | 같은 입력 2회에서 응답 판정이 뒤집힌 비율 JEV 0%, LLM Judge 1.2%, RAGAS 6.2% |
| 비용 | **Hybrid** | 1위 | 판정용 LLM 호출이 샘플당 0~0.40회. RAGAS는 1~2회 |

**결론**: 이번 데이터에서 Hybrid RAGAS-JEV는 네 metric 중 세 개에서 1위이고, Context Recall에서도 1위와의 차이가 작다 (Pearson −0.006). 판정에 필요한 LLM 호출은 RAGAS와 LLM Judge의 0~40% 수준이다 (Context Recall은 0회). 설계방향 문서의 목표("LLM은 문제를 나누고, JEV는 판단하고, 프로그램은 점수를 계산한다")가 품질과 비용 양쪽에서 성립한다.

단, **Hybrid의 보정은 이번 벤치마크와 같은 종류의 데이터로 학습했다** (교차 검증이라 같은 질문은 학습에 쓰지 않았다). 서비스 도메인에서는 재보정이 필요하다 (5.3절).

## 2. 비교 대상

| 시스템 | 분해 (평가 단위 생성) | 판정 | 점수 |
|---|---|---|---|
| **RAGAS** | ragas 0.4.3 자체 프롬프트 | `gpt-6-sol` (ragas 자체 프롬프트) | ragas 공식 계산 |
| **LLM Judge** | 이 파이프라인 (`gpt-6-luna` 추출기) | `gpt-6-sol`이 JEV와 같은 문항에 답함 | 이 파이프라인 |
| **JEV만** | 이 파이프라인 | JEV (보정·routing 없음) | 이 파이프라인 |
| **Hybrid** | 이 파이프라인 | JEV + 확률 보정 + metric별 routing (Phase 5 권장 정책) | 이 파이프라인 |

- 모든 LLM 판정자는 `gpt-6-sol`(벤치마크 기준선)이다. RAGAS와 LLM Judge는 판정 모델이 같고, 분해와 프롬프트만 다르다.
- RAGAS는 `temperature=1.0`을 보내는데, GPT-5 이후 모델은 이를 거부해 클라이언트에서 뺐다. 그 밖의 설정은 ragas 기본값이다 ([run_ragas_baseline.py](../../benchmark/run_ragas_baseline.py)).
- Hybrid 점수는 Phase 5의 **교차 검증 보정값**과 저장된 LLM 판정으로 계산했다. 배포용 보정 파일은 전체 데이터로 학습했기 때문에 벤치마크에는 쓰지 않았다.
- LLM Judge, JEV, Hybrid는 같은 추출 결과(claim/statement)를 판정한다. RAGAS만 자체적으로 분해한다.

### 2.1 metric 대응과 제약

| 이 프로젝트 | RAGAS 대응 | 제약 |
|---|---|---|
| Context Precision | `ContextRelevance` | RAGAS의 `ContextPrecision`은 reference 답변이 필요한데 MIRACL에는 없다. `ContextRelevance`는 contexts **전체**가 질문에 관련되는지를 0/0.5/1로 매기는 metric이라 정의가 다르다 |
| Faithfulness | `Faithfulness` | 같은 개념 |
| Context Recall | `ContextRecall` | 같은 개념. RAGAS는 reference **문장**마다 판정한다 |
| Answer Relevancy | `AnswerRelevancy` | 임베딩으로 점수를 매기는데 프록시에 임베딩 API가 없어 **실행하지 못했다** |

## 3. 데이터와 정답

모두 Phase 1~4에서 쓴 데이터다. 샘플 단위 정답은 다음과 같다.

| metric | 데이터 | 샘플 | 샘플 단위 정답 | 지표 |
|---|---|---|---|---|
| Context Precision | MIRACL-ko dev 213, MIRACL-en 213, 전부 무관 세트 100 | 526 | 판정된 passage 중 사람이 관련으로 본 비율 | Spearman, Kendall, MAE. 무관 세트는 평균 점수 |
| Faithfulness | RAGTruth QA 320 (전문가 라벨), 한국어 합성 192 | 512 | 응답에 hallucination이 있는가 | AUROC(1 − 점수). "점수 < 1"(RAGAS) 또는 "claim 중 p < 0.5 존재"(나머지)를 hallucination 판정으로 보고 P / R / F1 |
| Context Recall | MIRACL-ko/en 합성 reference × full / partial / none | 600 | reference 문장 중 출처 passage가 남아 있는 비율 | Pearson, MAE, full vs none AUROC, 단조 감소 비율 |
| Answer Relevancy | WikiEval 150, MIRACL-ko 369, MIRACL-en 399 | 918 | 변형 쌍 (좋은 답 > 부분 답 등) | 쌍 정확도 |

## 4. 결과

#### Context Precision (사람 라벨 관련 비율과의 상관)

| 시스템 | 한국어 Spearman / Kendall / MAE | 영어 Spearman / Kendall / MAE | 전부 무관 세트 평균 점수 (낮을수록 좋음) |
|---|---|---|---|
| RAGAS (ragas 0.4, gpt-6-sol) | 0.111 / 0.092 / 0.808 | 0.253 / 0.218 / 0.688 | 0.586 |
| LLM Judge (gpt-6-sol) | 0.565 / 0.414 / 0.158 | 0.499 / 0.378 / 0.157 | 0.140 |
| JEV만 | 0.621 / 0.448 / 0.228 | 0.544 / 0.410 / 0.178 | 0.188 |
| Hybrid (보정 + routing) | 0.681 / 0.503 / 0.079 | 0.567 / 0.429 / 0.121 | 0.056 |

#### Faithfulness (응답 단위 hallucination 탐지)

| 시스템 | RAGTruth AUROC | RAGTruth P / R / F1 | 한국어 AUROC | 한국어 P / R / F1 |
|---|---|---|---|---|
| RAGAS (ragas 0.4, gpt-6-sol) | 0.790 | 0.623 / 0.950 / 0.752 | 0.944 | 0.948 / 0.958 / 0.953 |
| LLM Judge (gpt-6-sol) | 0.821 | 0.648 / 0.930 / 0.764 | 0.969 | 0.978 / 0.938 / 0.957 |
| JEV만 | 0.809 | 0.706 / 0.867 / 0.778 | 0.962 | 0.958 / 0.958 / 0.958 |
| Hybrid (보정 + routing) | 0.830 | 0.750 / 0.835 / 0.790 | 0.970 | 0.989 / 0.917 / 0.951 |

#### Context Recall (기대 recall과의 상관, context 제거 반응)

| 시스템 | 한국어 Pearson / MAE / full vs none AUROC / 단조 | 영어 Pearson / MAE / full vs none AUROC / 단조 |
|---|---|---|
| RAGAS (ragas 0.4, gpt-6-sol) | 0.909 / 0.109 / 1.000 / 0.990 | 0.944 / 0.075 / 1.000 / 0.980 |
| LLM Judge (gpt-6-sol) | 0.893 / 0.128 / 1.000 / 0.940 | 0.944 / 0.087 / 1.000 / 1.000 |
| JEV만 | 0.878 / 0.161 / 1.000 / 0.930 | 0.920 / 0.134 / 1.000 / 0.980 |
| Hybrid (보정 + routing) | 0.903 / 0.143 / 0.999 / 0.930 | 0.938 / 0.117 / 1.000 / 0.980 |

#### Answer Relevancy (쌍 정확도; RAGAS는 임베딩이 없어 제외, Hybrid = JEV)

| 시스템 | WikiEval good > poor | 한국어 base > offtopic | 한국어 base > nonanswer | 영어 base > offtopic | 영어 base > nonanswer |
|---|---|---|---|---|---|
| LLM Judge (gpt-6-sol) | 0.780 | 0.840 | 0.950 | 0.780 | 0.930 |
| JEV만 | 0.920 | 0.980 | 0.970 | 0.950 | 0.980 |

#### 시스템 간 점수 순위 일치 (Kendall τ, 샘플 단위)

| metric | ragas~llm | ragas~jev | ragas~hybrid | llm~jev | llm~hybrid | jev~hybrid |
|---|---|---|---|---|---|---|
| context_precision | 0.420 | 0.441 | 0.481 | 0.701 | 0.616 | 0.709 |
| faithfulness | 0.710 | 0.631 | 0.650 | 0.746 | 0.762 | 0.875 |
| context_recall | 0.846 | 0.811 | 0.798 | 0.853 | 0.831 | 0.912 |

#### 반복 실행 안정성 (RAGTruth 80개 응답, 같은 입력 2회)

| 시스템 | unit 값 동일 | unit 판정 뒤집힘 | unit 평균 / 최대 Δp | 샘플 점수 평균 / 최대 Δ | 응답 판정(hallucination 여부) 뒤집힘 |
|---|---|---|---|---|---|
| RAGAS (ragas 0.4, gpt-6-sol) | – | – | – / – | 0.040 / 0.333 | 6.2% |
| LLM Judge (gpt-6-sol) | 65.3% | 3.1% | 0.025 / 0.880 | 0.022 / 0.201 | 1.2% |
| JEV만 | 56.6% | 0.4% | 0.007 / 0.090 | 0.003 / 0.017 | 0.0% |

#### Hybrid routing 비용과 RAGAS 지연

| metric | Hybrid 재판정 unit 비율 | Hybrid 샘플당 LLM 호출 | RAGAS 샘플당 p50 / p95 지연(s) | RAGAS 오류 |
|---|---|---|---|---|
| context_precision | 5.8% | 0.40 | 5.0 / 10.8 | 0 |
| faithfulness | 8.8% | 0.34 | 16.8 / 30.8 | 0 |
| context_recall | 0.0% | 0.00 | 9.1 / 16.8 | 0 |

### 4.1 효율

| metric | 시스템 | 판정용 LLM 호출/샘플 | 분해용 LLM 호출/샘플 | JEV 요청/샘플 | 샘플당 지연 (p50) |
|---|---|---|---|---|---|
| Context Precision | RAGAS | 2 (두 판정의 평균) | 0 | 0 | 5.0초 (측정) |
| | LLM Judge | 1 | 0 | 0 | 약 7~8초 (Phase 1 요청당 p50) |
| | JEV만 | 0 | 0 | 1 | 0.3초 |
| | Hybrid | 0.40 | 0 | 1 | 0.3초 + 호출 시 약 8초 |
| Faithfulness | RAGAS | 1 (판정) | 1 (문장 분해) | 0 | 16.8초 (측정) |
| | LLM Judge | 1 | 1 | 0 | 약 16초 (추출 9.8초 + 판정 6.1초, Phase 2) |
| | JEV만 | 0 | 1 | 1 | 약 10초 (추출 9.8초 + JEV 0.3초) |
| | Hybrid | 0.34 | 1 | 1 | 약 10초 + 호출 시 약 5초 |
| Context Recall | RAGAS | 1 | 0 (판정과 함께) | 0 | 9.1초 (측정) |
| | LLM Judge | 1 | 1 | 0 | 약 13초 (추출 8~9초 + 판정 4.4~4.8초, Phase 3) |
| | JEV만 / Hybrid | 0 | 1 | 1 | 약 9초 (추출 + JEV 0.3초) |

- RAGAS 지연은 이번 실행에서 샘플 단위로 잰 값이다. 나머지는 Phase 1~5에서 잰 요청 단위 지연을 더한 추정치다.
- 이 프로젝트의 추출 결과는 캐시되므로, 같은 답변을 다시 평가할 때는 분해 비용이 없다. RAGAS는 매번 다시 분해한다.
- JEV 비용은 입력 토큰당 $0.042/Mtok으로, 1,000샘플당 $0.04~0.19 수준이다 (Phase 1~4 실측). LLM 비용은 OAuth 프록시라 측정하지 못해 호출 수로 비교했다.

## 5. 해석

### 5.1 Context Precision

- Hybrid가 사람 라벨과의 순위 상관(Spearman 0.681 / 0.567)과 절대 오차(MAE 0.079 / 0.121) 모두 가장 좋다. 전부 무관한 세트에서도 평균 점수가 0.056으로 가장 정확하다 (정답은 0).
- JEV만으로도 순위 상관은 LLM Judge보다 높다 (0.621 vs 0.565). 다만 JEV p가 "관련" 쪽으로 치우쳐 절대 오차는 LLM Judge보다 크다 (0.228 vs 0.158). 이 치우침을 보정이 없앤다 (Phase 5).
- RAGAS `ContextRelevance`는 사람의 precision과 거의 상관이 없다. 관련 passage가 하나라도 있으면 높은 점수를 주는 metric이라, 무관 passage 비율을 반영하지 못한다. **metric 정의가 달라서 생긴 차이**이므로 RAGAS의 판정 능력이 낮다는 뜻은 아니다. RAGAS로 precision을 재려면 reference 답변이 있는 데이터가 필요하다.

### 5.2 Faithfulness

- RAGTruth(영어 전문가 라벨)에서 Hybrid가 AUROC와 F1 모두 1위다. RAGAS는 recall이 가장 높지만(0.950) precision이 가장 낮아(0.623), hallucination이 없는 응답도 많이 잡는다.
- LLM Judge와 RAGAS는 판정 모델(`gpt-6-sol`)이 같다. LLM Judge가 AUROC 0.821로 RAGAS(0.790)보다 높은 것은 **이 파이프라인의 claim 분해가 RAGAS의 문장 분해보다 판정에 유리하다**는 뜻으로 읽을 수 있다.
- 한국어 합성 세트는 네 시스템 모두 0.94 이상이라 변별력이 낮다 (Phase 2에서 본 것처럼 합성 세트가 쉽다).

### 5.3 Context Recall

- RAGAS가 가장 좋다. 이번 정답을 "reference 문장 중 출처 passage가 남은 비율"로 정했는데, RAGAS도 reference **문장** 단위로 판정한다. 그래서 RAGAS의 판정 단위가 정답과 정확히 맞는다.
- 이 프로젝트는 문장을 더 잘게 claim으로 나눈다. 한 문장에서 나온 claim 일부만 지원되는 경우 정답(문장 단위)과 어긋날 수 있다. Hybrid(Pearson 0.903 / 0.938)와 RAGAS의 차이는 0.006 이하로 작다.
- 네 시스템 모두 full과 none을 완벽히 구분한다 (AUROC 1.000).

### 5.4 Answer Relevancy

- JEV(= Hybrid)가 모든 쌍 비교에서 LLM Judge보다 높다. Phase 4에서 본 것처럼 LLM은 statement 확률을 거의 0/1로 줘서, 원본의 오탐 하나가 무관 문장 삽입만큼 점수를 떨어뜨린다.
- RAGAS는 비교하지 못했다. 임베딩 API가 생기면 추가할 수 있다 (7장).

### 5.5 재현성

- 같은 80개 응답을 두 번 평가했을 때, JEV는 claim 판정이 0.4%만 뒤집혔고 응답 단위 판정(hallucination 여부)은 한 건도 바뀌지 않았다. LLM Judge는 claim 판정의 3.1%, 응답 판정의 1.2%가 바뀌었다.
- RAGAS는 응답 판정의 6.2%가 바뀌었고, 샘플 점수는 최대 0.333까지 달라졌다. 매번 문장 분해부터 다시 하므로 분해 결과가 달라지는 만큼 변동이 더 크다.
- Phase 1(Context Precision)에서는 JEV도 최대 0.83까지 변했다. 판단 유형에 따라 JEV의 안정성이 다르다. 어느 경우든 이 프로젝트는 추출·판정 결과를 캐시하므로, 캐시를 쓰면 점수가 비트 단위로 재현된다.

### 5.6 시스템 간 일치

- JEV와 Hybrid의 샘플 점수 순위 일치(Kendall τ)는 0.71~0.91로 높다. Hybrid는 JEV의 판정을 대부분 유지하고 일부만 바꾸기 때문이다.
- RAGAS와 나머지 시스템의 일치는 Context Precision에서 특히 낮다 (0.42~0.48). 5.1절의 정의 차이 때문이다.

## 6. 한계

1. **Hybrid의 보정 데이터와 벤치마크 데이터가 같은 분포다.** 교차 검증으로 같은 질문을 학습과 평가에 함께 쓰지는 않았지만, 서비스 데이터에서는 라벨 비율이 달라 보정이 맞지 않을 수 있다 (Phase 5, 5.3절).
2. **합성 라벨**: Faithfulness 한국어, Context Recall 전체, Answer Relevancy의 MIRACL 세트는 `gpt-6-astra`로 만든 합성 데이터다.
3. **Context Recall 정답 정의가 RAGAS에 유리하다** (5.3절).
4. **RAGAS Context Precision과 Answer Relevancy는 실제로 비교하지 못했다** (2.1절).
5. **LLM 판정은 결정적이지 않다.** 1~2%p 차이는 다시 돌리면 바뀔 수 있다.
6. **도메인**: 모두 Wikipedia·웹 기반 일반 도메인이다. 금융·의료 같은 전문 도메인에서는 다시 검증해야 한다.

## 7. 제안

1. **도메인 재보정 절차** (Phase 5 제안과 같음): 서비스 데이터에서 샘플을 뽑아 `review export/import`로 사람 라벨을 모으고, 보정을 다시 학습하는 스크립트를 만든다. 6장 한계 1의 해법이다.
2. **reference가 있는 한국어 데이터로 Context Precision 재비교**: RAGAS `ContextPrecision`과 정의가 같은 조건에서 비교하려면 reference 답변이 있는 데이터가 필요하다. 예: Allganize RAG-Evaluation-Dataset-KO.
3. **RAGAS AnswerRelevancy 추가**: 임베딩 API가 생기거나 로컬 임베딩 모델을 쓰면 비교할 수 있다.

## 8. 재현 방법

```bash
# Phase 1~5 결과가 .cache/에 있어야 한다
uv run --group benchmark python benchmark/run_ragas_baseline.py     # 공식 ragas, 약 1,300샘플
uv run --group benchmark python benchmark/run_ragas_baseline.py --only faithfulness --datasets ragtruth_qa_test --limit 80 --tag rep1
uv run python benchmark/run_phase2.py --datasets ragtruth_qa_test --limit 80 \
  --configs jev/claim_support.v1/joint llm:gpt-6-sol/claim_support.v1/joint --tag rep1 --summary .cache/phase2/summary_rep1.json
uv run python benchmark/run_phase6.py                                 # 집계 (API 호출 없음)
```

`ragas`를 쓰려면 Python 3.12 가상환경이 필요하다 (`scikit-network`가 Windows + Python 3.14용 wheel을 제공하지 않는다). `benchmark` 그룹에서 `langchain-community<0.4`로 고정했다 (ragas 0.4.3이 0.4에서 제거된 모듈을 import한다).
