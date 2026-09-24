# Phase 6 벤치마크 리포트: RAGAS vs LLM Judge vs JEV vs Hybrid

- 실행일: 2026-09-24 (후속 비교 2건 추가: reference 기반 Context Precision, RAGAS Answer Relevancy)
- 관련 문서: [구현계획서](../구현계획서.md) Phase 6, [설계방향](../RAGAS-JEV_설계방향.md) 17장, Phase [1](phase1_context_precision.md) · [2](phase2_faithfulness.md) · [3](phase3_context_recall.md) · [4](phase4_answer_relevancy.md) · [5](phase5_routing_calibration.md) 리포트, [도메인 재보정 가이드](../도메인_재보정_가이드.md)

## 1. 요약

설계방향 문서가 정한 비교 대상을 같은 데이터로 비교했다. 기존 RAGAS와 LLM 기반 Judge, JEV만 쓰는 평가, Hybrid RAGAS-JEV다.

| metric / 항목 | 가장 좋은 시스템 | Hybrid | 비고 |
|---|---|---|---|
| Context Precision: 질문만 보는 판정 (MIRACL, 관련 18%) | **Hybrid** (Spearman 0.681 / 0.567) | 1위 | RAGAS는 정의가 다른 `ContextRelevance`만 가능 (0.111 / 0.253) |
| Context Precision: reference 기반 (관련 44%) | **JEV v3 / Hybrid v3** (AP Kendall 0.518~0.537 / 0.523~0.545) | v3 + 재보정이 순위 일치·오차·chunk κ 1위 | reference를 보는 v3 문항으로 RAGAS(0.459 / 0.518)를 따라잡았다. 영어 AP Pearson만 RAGAS가 약간 높다 (0.696 vs 0.672). v2 + 기본 보정은 분포 차이로 최하위 (5.1절) |
| Faithfulness (RAGTruth) | **Hybrid** (AUROC 0.830, F1 0.790) | 1위 | RAGAS 0.790 / 0.752, LLM Judge 0.821 / 0.764 |
| Context Recall | **RAGAS** (Pearson 0.909 / 0.944) | 2~3위 (0.903 / 0.938) | 정답을 reference 문장 단위로 만들어 RAGAS 방식에 유리하다 |
| Answer Relevancy | **JEV = Hybrid** (WikiEval 0.92, 무관 문장 삽입 0.98 / 0.95) | 1위 | LLM Judge 0.78 / 0.84 / 0.78, RAGAS 0.72 / 0.61 / 0.49 |
| 재현성 (Faithfulness) | **JEV / Hybrid** | 1위 | 같은 입력 2회에서 응답 판정이 뒤집힌 비율 JEV 0%, LLM Judge 1.2%, RAGAS 6.2% |
| 비용 | **Hybrid** | 1위 | 판정용 LLM 호출이 샘플당 0~0.43회. RAGAS 1~5회, LLM Judge 1회 |

**결론**

- Hybrid RAGAS-JEV는 Faithfulness, Answer Relevancy, Context Precision(질문만 보는 경우와 reference를 보는 경우 모두)에서 1위이고, 판정용 LLM 호출은 RAGAS·LLM Judge의 0~40%다. 설계방향 문서의 구조("LLM은 문제를 나누고, JEV는 판단하고, 프로그램은 점수를 계산한다")가 품질과 비용 양쪽에서 성립한다.
- reference 기반 Context Precision에서 RAGAS가 앞섰던 것은 **reference라는 정보 차이** 때문이었다. JEV에게도 reference를 보여 주는 v3 문항을 쓰면, 한국어는 LLM 호출 없이 RAGAS보다 좋다. v3를 Context Precision 기본 문항으로 바꿨다 (reference가 없으면 v2).
- Context Recall은 RAGAS가 앞선다. 정답 정의(reference 문장 단위)가 RAGAS 방식에 가깝다.
- **보정은 데이터 분포에 의존한다는 것이 실제로 확인됐다.** 관련 passage 비율이 다른 세트에 기본 보정을 쓰면 Hybrid가 가장 나빴고, 그 세트의 라벨로 재보정하면 LLM Judge 수준으로 회복했다. 서비스 도메인에서는 [도메인 재보정](../도메인_재보정_가이드.md)이 필수다.

## 2. 비교 대상

| 시스템 | 분해 (평가 단위 생성) | 판정 | 점수 |
|---|---|---|---|
| **RAGAS** | ragas 0.4.3 자체 프롬프트 | `gpt-6-sol` (ragas 자체 프롬프트) | ragas 공식 계산 |
| **LLM Judge** | 이 파이프라인 (`gpt-6-luna` 추출기) | `gpt-6-sol`이 JEV와 같은 문항에 답함 | 이 파이프라인 |
| **JEV만** | 이 파이프라인 | JEV (보정·routing 없음) | 이 파이프라인 |
| **Hybrid** | 이 파이프라인 | JEV + 확률 보정 + metric별 routing (Phase 5 권장 정책) | 이 파이프라인 |

- 모든 LLM 판정자는 `gpt-6-sol`(벤치마크 기준선)이다. RAGAS와 LLM Judge는 판정 모델이 같고, 분해와 프롬프트만 다르다.
- RAGAS는 `temperature=1.0`을 보내는데, GPT-5 이후 모델은 이를 거부해 클라이언트에서 뺐다. 그 밖의 설정은 ragas 기본값이다 ([run_ragas_baseline.py](../../benchmark/run_ragas_baseline.py)).
- Hybrid 점수는 **이 벤치마크의 평가 대상을 보정 학습에 쓰지 않도록** 계산했다. 1차 비교는 Phase 5의 교차 검증 보정값을 썼고, reference 기반 Context Precision은 그 세트의 질문을 뺀 Phase 1 데이터로 보정을 학습했다. 배포용 보정 파일은 쓰지 않았다.
- LLM Judge, JEV, Hybrid는 같은 추출 결과(claim/statement)를 판정한다. RAGAS만 자체적으로 분해한다.

### 2.1 metric 대응과 제약

| 이 프로젝트 | RAGAS 대응 | 조건 |
|---|---|---|
| Context Precision (MIRACL) | `ContextRelevance` | MIRACL에는 reference가 없다. `ContextRelevance`는 contexts **전체**가 질문에 관련되는지를 0/0.5/1로 매기는 metric이라 precision과 정의가 다르다 |
| Context Precision (reference 세트) | `ContextPrecision` (reference 사용) | 정의가 같다. 이 프로젝트는 질문만 보는 v2와 reference도 보는 v3를 모두 비교했다 |
| Faithfulness | `Faithfulness` | 같은 개념 |
| Context Recall | `ContextRecall` | 같은 개념. RAGAS는 reference **문장**마다 판정한다 |
| Answer Relevancy | `AnswerRelevancy` | 임베딩이 필요한데 프록시에 없어 **로컬 다국어 임베딩**(`paraphrase-multilingual-MiniLM-L12-v2`)으로 실행했다. 공식 구성(OpenAI 임베딩)보다 불리할 수 있다 |

## 3. 데이터와 정답

| metric | 데이터 | 샘플 | 샘플 단위 정답 | 지표 |
|---|---|---|---|---|
| Context Precision (MIRACL) | MIRACL-ko dev 213, MIRACL-en 213, 전부 무관 세트 100 | 526 | 판정된 passage 중 사람이 관련으로 본 비율 | Spearman, Kendall, MAE. 무관 세트는 평균 점수 |
| Context Precision (reference) | Phase 3 recall 세트의 full·partial 조건 (한국어·영어 각 200) | 400 | context 순서대로 사람 라벨을 놓고 계산한 average precision (RAGAS 공식) | AP와의 Pearson / Spearman / Kendall, MAE, chunk F1 / κ |
| Faithfulness | RAGTruth QA 320 (전문가 라벨), 한국어 합성 192 | 512 | 응답에 hallucination이 있는가 | AUROC(1 − 점수). "점수 < 1"(RAGAS) 또는 "claim 중 p < 0.5 존재"(나머지)를 hallucination 판정으로 보고 P / R / F1 |
| Context Recall | MIRACL-ko/en 합성 reference × full / partial / none | 600 | reference 문장 중 출처 passage가 남아 있는 비율 | Pearson, MAE, full vs none AUROC, 단조 감소 비율 |
| Answer Relevancy | WikiEval 150, MIRACL-ko 369, MIRACL-en 399 | 918 | 변형 쌍 (좋은 답 > 부분 답 등) | 쌍 정확도, 틀린 답과의 평균 차 |

reference 기반 Context Precision 세트([prepare_cp_reference.py](../../benchmark/prepare_cp_reference.py))는 Phase 3 recall 세트를 재사용했다. reference는 합성이지만, context의 관련 여부는 MIRACL 원어민 라벨이고 context 순서는 고정돼 있다. 처음 제안한 Allganize 한국어 데이터는 문서가 PDF가 아니라 기관 게시판 링크로만 제공돼(64개) 쓰지 않았다.

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

#### Answer Relevancy (쌍 정확도; Hybrid = JEV)

RAGAS AnswerRelevancy는 로컬 다국어 임베딩(paraphrase-multilingual-MiniLM-L12-v2)으로 실행했다. wrong 비교는 쌍 정확도가 0.5 근처, 평균 차가 0에 가까워야 좋다 (관련성은 정답 여부와 무관해야 한다).

| 시스템 | WikiEval good > poor | 한국어 base > offtopic | 한국어 base > nonanswer | 영어 base > offtopic | 영어 base > nonanswer | WikiEval good vs wrong 평균 차 | 한국어 base vs wrong 평균 차 | 영어 base vs wrong 평균 차 |
|---|---|---|---|---|---|---|---|---|
| RAGAS (ragas 0.4, gpt-6-sol) | 0.720 | 0.605 | 0.900 | 0.490 | 0.960 | 0.038 | 0.136 | 0.047 |
| LLM Judge (gpt-6-sol) | 0.780 | 0.840 | 0.950 | 0.780 | 0.930 | -0.009 | -0.008 | -0.002 |
| JEV만 | 0.920 | 0.980 | 0.970 | 0.950 | 0.980 | 0.008 | 0.013 | 0.012 |

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

### 4.1 reference 기반 Context Precision

정답은 사람 라벨로 계산한 AP다. RAGAS는 context마다 판정해 AP를 낸다. 나머지 시스템은 p ≥ 0.5를 판정으로 보고 같은 공식으로 AP를 계산했다 ([run_cp_reference.py](../../benchmark/run_cp_reference.py)).

- **v2 문항** (`chunk_relevance.v2`): 질문만 보고 "이 passage가 질문에 답하는 정보를 담고 있는가"를 묻는다.
- **v3 문항** (`chunk_relevance.v3`): reference도 함께 보고 "이 passage에 reference 답변에 쓰인 정보가 있는가"를 묻는다. RAGAS `ContextPrecision`과 같은 정보를 쓴다.
- **Hybrid (기본 보정)**: Phase 1 MIRACL 데이터 중 이 세트의 질문을 뺀 unit으로 학습한 보정을 썼다 (v2).
- **재보정**: 이 세트의 라벨로 질문 단위 2-fold 재보정을 했다. 한쪽 절반의 질문으로 학습한 보정을 다른 절반에 적용한다. [도메인 재보정](../도메인_재보정_가이드.md) 절차(`ragas_jev.scoring.recalibration`)를 그대로 썼다.

##### 한국어 (200 samples)

| 시스템 | AP Pearson / Spearman / Kendall | AP MAE | soft precision Pearson / MAE | chunk F1 / κ |
|---|---|---|---|---|
| RAGAS ContextPrecision (reference 사용) | 0.638 / 0.560 / 0.459 | 0.128 | – / – | – |
| LLM Judge (gpt-6-sol) | 0.559 / 0.554 / 0.450 | 0.138 | 0.528 / 0.125 | 0.798 / 0.623 |
| JEV만 | 0.550 / 0.544 / 0.448 | 0.141 | 0.527 / 0.143 | 0.795 / 0.610 |
| Hybrid (기본 보정 + routing) | 0.390 / 0.408 / 0.332 | 0.223 | 0.474 / 0.166 | 0.735 / 0.576 |
| Hybrid (이 도메인 라벨로 재보정 + routing) | 0.558 / 0.579 / 0.475 | 0.137 | 0.529 / 0.116 | 0.808 / 0.653 |
| LLM Judge v3 (reference 사용) | 0.666 / 0.594 / 0.502 | 0.119 | 0.550 / 0.151 | 0.851 / 0.701 |
| JEV만 v3 (reference 사용) | 0.674 / 0.645 / 0.518 | 0.123 | 0.614 / 0.132 | 0.843 / 0.697 |
| Hybrid v3 (reference 사용, 재보정 + routing) | 0.644 / 0.650 / 0.537 | 0.117 | 0.610 / 0.109 | 0.851 / 0.723 |
- Hybrid (기본 보정 + routing): 재판정 unit 12.2%, 샘플당 LLM 호출 0.43
- Hybrid (이 도메인 라벨로 재보정 + routing): 재판정 unit 7.6%, 샘플당 LLM 호출 0.28
- Hybrid v3 (reference 사용, 재보정 + routing): 재판정 unit 7.4%, 샘플당 LLM 호출 0.28

Hybrid: 재판정 unit 12.2%, 샘플당 LLM 호출 0.43 (보정 학습 unit 2596개, 이 세트의 질문 제외). RAGAS: 샘플당 LLM 호출 5 (context마다 1), p50 지연 21.2초, 오류 0

##### 영어 (200 samples)

| 시스템 | AP Pearson / Spearman / Kendall | AP MAE | soft precision Pearson / MAE | chunk F1 / κ |
|---|---|---|---|---|
| RAGAS ContextPrecision (reference 사용) | 0.696 / 0.645 / 0.518 | 0.126 | – / – | – |
| LLM Judge (gpt-6-sol) | 0.654 / 0.638 / 0.497 | 0.148 | 0.348 / 0.171 | 0.749 / 0.560 |
| JEV만 | 0.574 / 0.562 / 0.431 | 0.183 | 0.352 / 0.168 | 0.721 / 0.505 |
| Hybrid (기본 보정 + routing) | 0.516 / 0.502 / 0.392 | 0.202 | 0.319 / 0.161 | 0.709 / 0.543 |
| Hybrid (이 도메인 라벨로 재보정 + routing) | 0.614 / 0.615 / 0.477 | 0.169 | 0.329 / 0.151 | 0.740 / 0.561 |
| LLM Judge v3 (reference 사용) | 0.622 / 0.566 / 0.474 | 0.138 | 0.412 / 0.170 | 0.829 / 0.666 |
| JEV만 v3 (reference 사용) | 0.672 / 0.639 / 0.523 | 0.122 | 0.508 / 0.137 | 0.828 / 0.684 |
| Hybrid v3 (reference 사용, 재보정 + routing) | 0.672 / 0.646 / 0.545 | 0.115 | 0.382 / 0.136 | 0.830 / 0.694 |
- Hybrid (기본 보정 + routing): 재판정 unit 9.0%, 샘플당 LLM 호출 0.35
- Hybrid (이 도메인 라벨로 재보정 + routing): 재판정 unit 9.3%, 샘플당 LLM 호출 0.33
- Hybrid v3 (reference 사용, 재보정 + routing): 재판정 unit 12.0%, 샘플당 LLM 호출 0.42

Hybrid: 재판정 unit 9.0%, 샘플당 LLM 호출 0.35 (보정 학습 unit 1170개, 이 세트의 질문 제외). RAGAS: 샘플당 LLM 호출 5 (context마다 1), p50 지연 19.0초, 오류 0

### 4.2 효율

| metric | 시스템 | 판정용 LLM 호출/샘플 | 분해용 LLM 호출/샘플 | JEV 요청/샘플 | 샘플당 지연 (p50) |
|---|---|---|---|---|---|
| Context Precision (MIRACL) | RAGAS `ContextRelevance` | 2 (두 판정의 평균) | 0 | 0 | 5.0초 (측정) |
| | LLM Judge | 1 | 0 | 0 | 약 7~8초 (Phase 1 요청당 p50) |
| | JEV만 | 0 | 0 | 1 | 0.3초 |
| | Hybrid | 0.40 | 0 | 1 | 0.3초 + 호출 시 약 8초 |
| Context Precision (reference) | RAGAS `ContextPrecision` | 5 (context마다 1) | 0 | 0 | 19~21초 (측정) |
| | Hybrid (재보정) | 0.28~0.33 | 0 | 1 | 0.3초 + 호출 시 약 8초 |
| Faithfulness | RAGAS | 1 (판정) | 1 (문장 분해) | 0 | 16.8초 (측정) |
| | LLM Judge | 1 | 1 | 0 | 약 16초 (추출 9.8초 + 판정 6.1초, Phase 2) |
| | JEV만 | 0 | 1 | 1 | 약 10초 (추출 9.8초 + JEV 0.3초) |
| | Hybrid | 0.34 | 1 | 1 | 약 10초 + 호출 시 약 5초 |
| Context Recall | RAGAS | 1 | 0 (판정과 함께) | 0 | 9.1초 (측정) |
| | LLM Judge | 1 | 1 | 0 | 약 13초 (추출 8~9초 + 판정 4.4~4.8초, Phase 3) |
| | JEV만 / Hybrid | 0 | 1 | 1 | 약 9초 (추출 + JEV 0.3초) |
| Answer Relevancy | RAGAS | 질문 생성 LLM + 로컬 임베딩 | 0 | 0 | 9.1초 (측정) |
| | LLM Judge | 1 | 1 | 0 | 약 11초 (추출 6.5~7.8초 + 판정 4.3초, Phase 4) |
| | JEV만 / Hybrid | 0 | 1 | 1 | 약 7~8초 (추출 + JEV 0.3초) |

- RAGAS 지연은 이번 실행에서 샘플 단위로 잰 값이다. 나머지는 Phase 1~5에서 잰 요청 단위 지연을 더한 추정치다.
- 이 프로젝트의 추출 결과는 캐시되므로, 같은 답변을 다시 평가할 때는 분해 비용이 없다. RAGAS는 매번 다시 분해한다.
- JEV 비용은 입력 토큰당 $0.042/Mtok으로, 1,000샘플당 $0.04~0.19 수준이다 (Phase 1~4 실측). LLM 비용은 OAuth 프록시라 측정하지 못해 호출 수로 비교했다.

## 5. 해석

### 5.1 Context Precision

**질문만 보는 판정 (MIRACL)**

- Hybrid가 사람 라벨과의 순위 상관(Spearman 0.681 / 0.567)과 절대 오차(MAE 0.079 / 0.121) 모두 가장 좋다. 전부 무관한 세트에서도 평균 점수가 0.056으로 가장 정확하다 (정답은 0).
- JEV만으로도 순위 상관은 LLM Judge보다 높다 (0.621 vs 0.565). JEV p가 "관련" 쪽으로 치우쳐 절대 오차가 큰 문제는 보정이 없앤다 (Phase 5).
- RAGAS `ContextRelevance`가 사람의 precision과 거의 상관이 없는 것은 **metric 정의가 달라서**다. 관련 passage가 하나라도 있으면 높은 점수를 주므로 무관 passage 비율을 반영하지 못한다.

**reference 기반 판정**

- **v2 문항(질문만)에서는 RAGAS가 가장 좋다** (AP Pearson 0.638 / 0.696). RAGAS는 reference를 보고 "이 context가 reference를 만드는 데 쓰였는가"를 판정하므로 정보가 더 많다.
- **v3 문항으로 JEV에게도 reference를 보여 주면 격차가 사라진다.** JEV만 v3의 AP Pearson은 0.674 / 0.672로, 한국어는 RAGAS보다 높고 영어는 0.024 낮다. 순위 일치(Kendall 0.518 / 0.523)와 오차(MAE 0.123 / 0.122)는 두 언어 모두 RAGAS와 같거나 낫다. JEV만이라 판정용 LLM 호출은 0회다 (RAGAS는 5회).
- **Hybrid v3(재보정 + routing)**는 순위 일치(Kendall 0.537 / 0.545), 오차(MAE 0.117 / 0.115), chunk κ(0.723 / 0.694)에서 모든 시스템 중 1위다. 샘플당 LLM 호출은 0.28 / 0.42다.
- LLM Judge도 v3로 좋아진다 (0.559 → 0.666 / 0.654 → 0.622). reference가 있으면 쓰는 것이 판정자와 무관하게 유리하다.
- **v2 + 기본 보정을 쓴 Hybrid가 가장 나쁘다** (0.390 / 0.516). 보정을 학습한 MIRACL 데이터는 관련 passage가 18%인데, 이 세트는 44%다. 보정이 JEV p를 크게 낮춰(한국어 p 0.9 → 약 0.36) 관련 passage 상당수가 무관으로 판정됐다. Phase 5 리포트 5.3절에서 경고한 "보정이 라벨 기본 비율을 학습한다"는 문제가 실제로 드러났다. 이 세트 라벨로 재보정하면 0.558 / 0.614로 회복한다.

**결정**: Context Precision 기본 문항을 `chunk_relevance.v3`로 바꿨다. reference가 없는 샘플은 v2로 판정하고, 결과의 `question_id`에 실제 문항이 기록된다. 기본 보정 파일에는 v3 보정이 없으므로 v3는 JEV 원래 확률을 쓴다. 보정하려면 도메인 재보정이 필요하다.

### 5.2 Faithfulness

- RAGTruth(영어 전문가 라벨)에서 Hybrid가 AUROC와 F1 모두 1위다. RAGAS는 recall이 가장 높지만(0.950) precision이 가장 낮아(0.623), hallucination이 없는 응답도 많이 잡는다.
- LLM Judge와 RAGAS는 판정 모델(`gpt-6-sol`)이 같다. LLM Judge가 AUROC 0.821로 RAGAS(0.790)보다 높은 것은 **이 파이프라인의 claim 분해가 RAGAS의 문장 분해보다 판정에 유리하다**는 뜻으로 읽을 수 있다.
- 한국어 합성 세트는 네 시스템 모두 0.94 이상이라 변별력이 낮다.

### 5.3 Context Recall

- RAGAS가 가장 좋다. 정답을 "reference 문장 중 출처 passage가 남은 비율"로 정했는데, RAGAS도 reference **문장** 단위로 판정하므로 판정 단위가 정답과 정확히 맞는다.
- 이 프로젝트는 문장을 더 잘게 claim으로 나눈다. 한 문장에서 나온 claim 일부만 지원되면 정답과 어긋날 수 있다. Hybrid(Pearson 0.903 / 0.938)와 RAGAS의 차이는 0.006 이하다.
- 네 시스템 모두 full과 none을 완벽히 구분한다 (AUROC 1.000).

### 5.4 Answer Relevancy

- JEV(= Hybrid)가 모든 쌍 비교에서 가장 높다.
- RAGAS는 답변에서 질문을 역으로 생성하고, 원래 질문과의 임베딩 유사도로 점수를 낸다. 그래서 **답변에 무관한 문장이 섞여도 거의 감지하지 못한다** (원본 > 무관 문장 삽입: 한국어 0.605, 영어 0.490, 우연 수준). 답이 없는 답변(nonanswer)은 잘 잡는다(0.90 / 0.96).
- RAGAS는 **틀린 답변을 낮게 매긴다** (한국어 원본 − 틀린 답 평균 0.136). 틀린 사실로 생성한 질문이 원래 질문과 덜 비슷해지기 때문으로 보인다. 관련성은 정답 여부와 무관해야 하므로 바람직하지 않다. JEV와 LLM Judge의 statement 점수는 차이가 0.01 안팎이다.
- WikiEval에서 RAGAS 0.72는 RAGAS 논문 보고치(0.78)와 비슷한 수준이다. 로컬 임베딩을 써서 공식 구성보다 불리할 수 있다.
- LLM Judge는 statement 확률을 거의 0/1로 줘서, 원본의 오탐 하나가 무관 문장 삽입만큼 점수를 떨어뜨린다 (Phase 4).

### 5.5 재현성

- 같은 80개 응답을 두 번 평가했을 때, JEV는 claim 판정이 0.4%만 뒤집혔고 응답 단위 판정(hallucination 여부)은 한 건도 바뀌지 않았다. LLM Judge는 claim 판정의 3.1%, 응답 판정의 1.2%가 바뀌었다.
- RAGAS는 응답 판정의 6.2%가 바뀌었고, 샘플 점수는 최대 0.333까지 달라졌다. 매번 문장 분해부터 다시 하므로 변동이 더 크다.
- Phase 1(Context Precision)에서는 JEV도 최대 0.83까지 변했다. 판단 유형에 따라 JEV의 안정성이 다르다. 이 프로젝트는 추출·판정 결과를 캐시하므로, 캐시를 쓰면 점수가 비트 단위로 재현된다.

### 5.6 시스템 간 일치

- JEV와 Hybrid의 샘플 점수 순위 일치(Kendall τ)는 0.71~0.91로 높다. Hybrid는 JEV의 판정을 대부분 유지하고 일부만 바꾼다.
- RAGAS와 나머지 시스템의 일치는 MIRACL Context Precision에서 특히 낮다 (0.42~0.48). 5.1절의 정의 차이 때문이다.

## 6. 한계

1. **보정은 분포에 의존한다.** 4.1절에서 기본 보정이 분포가 다른 데이터에서 손해가 되는 것을 확인했다. 서비스 도메인에서는 재보정하지 않은 Hybrid를 쓰면 안 된다.
2. **합성 라벨**: Faithfulness 한국어, Context Recall 전체, reference 기반 Context Precision의 reference, Answer Relevancy의 MIRACL 세트는 `gpt-6-astra`로 만든 합성 데이터다.
3. **정답 정의의 유리함**: Context Recall 정답은 RAGAS 방식에 유리하다 (5.3절). reference 기반 Context Precision에서 RAGAS는 reference를 보고, 이 프로젝트의 문항은 보지 않는다.
4. **RAGAS Answer Relevancy는 로컬 임베딩**으로 실행했다.
5. **LLM 판정은 결정적이지 않다.** 1~2%p 차이는 다시 돌리면 바뀔 수 있다.
6. **도메인**: 모두 Wikipedia·웹 기반 일반 도메인이다. 금융·의료 같은 전문 도메인에서는 다시 검증해야 한다.

## 7. 후속 작업 결과와 남은 제안

| 제안 (1차 리포트) | 결과 |
|---|---|
| 도메인 재보정 절차 | **완료.** `ragas-jev calibration sample / fit` ([가이드](../도메인_재보정_가이드.md)). 4.1절에서 효과 확인 |
| reference가 있는 데이터로 Context Precision 재비교 | **완료** (4.1절). Allganize 대신 Phase 3 recall 세트 사용 |
| RAGAS Answer Relevancy 추가 | **완료** (로컬 임베딩). 5.4절 |
| reference를 쓰는 Context Precision 문항 | **완료.** `chunk_relevance.v3`, 기본 문항으로 채택 (4.1, 5.1절) |

남은 제안:

1. **전문 도메인 검증**: 서비스 도메인 데이터로 재보정하고(v3 보정 포함), 이 벤치마크를 다시 돌린다. 서비스 데이터 표본과 보안·법무 검토(국외 API 전송)가 먼저 필요하다.

## 8. 재현 방법

```bash
# Phase 1~5 결과가 .cache/에 있어야 한다
uv run --group benchmark python benchmark/run_ragas_baseline.py --only context_relevance faithfulness context_recall
uv run --group benchmark python benchmark/run_ragas_baseline.py --only faithfulness --datasets ragtruth_qa_test --limit 80 --tag rep1
uv run python benchmark/run_phase2.py --datasets ragtruth_qa_test --limit 80 \
  --configs jev/claim_support.v1/joint llm:gpt-6-sol/claim_support.v1/joint --tag rep1 --summary .cache/phase2/summary_rep1.json
uv run python benchmark/run_phase6.py                                 # 1차 집계 (API 호출 없음)

# reference 기반 Context Precision
uv run python benchmark/prepare_cp_reference.py --lang ko
uv run python benchmark/prepare_cp_reference.py --lang en
uv run --group benchmark python benchmark/run_ragas_baseline.py --only context_precision
uv run python benchmark/run_phase1.py --datasets cp_ref_miracl_ko cp_ref_miracl_en --chunk-question chunk_relevance.v2 \
  --repeats 1 --llm-judge --summary .cache/phase1/summary_cp_ref.json
uv run python benchmark/run_phase1.py --datasets cp_ref_miracl_ko cp_ref_miracl_en --chunk-question chunk_relevance.v3 \
  --repeats 1 --llm-judge --summary .cache/phase1/summary_cp_ref_v3.json
uv run python benchmark/run_cp_reference.py

# RAGAS Answer Relevancy (로컬 임베딩)
uv run --group benchmark --group embeddings python benchmark/run_ragas_baseline.py --only answer_relevancy
```

`ragas`를 쓰려면 Python 3.12 가상환경이 필요하다 (`scikit-network`가 Windows + Python 3.14용 wheel을 제공하지 않는다). `benchmark` 그룹에서 `langchain-community<0.4`로 고정했다 (ragas 0.4.3이 0.4에서 제거된 모듈을 import한다). Answer Relevancy에는 `embeddings` 그룹(`sentence-transformers`, torch)이 필요하고, 임베딩 모델은 처음 실행할 때 Hugging Face에서 내려받는다.
