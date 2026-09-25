# RAGAS-JEV 평가 항목: 기존 RAGAS와의 비교

- 기준: 2026-09-24, `ragas-jev` main, JEV `jev-1.13.0`, 비교 대상 `ragas` 0.4.3
- 관련 문서: [설계방향](RAGAS-JEV_설계방향.md), [구현계획서](구현계획서.md), [Phase 6 벤치마크 리포트](reports/phase6_benchmark.md), [도메인 재보정 가이드](도메인_재보정_가이드.md)

이 문서는 RAGAS-JEV가 구현한 평가 항목을 RAGAS의 같은 항목과 나란히 놓고 정리한다. 각 항목에서 무엇을 재는지, 어떻게 계산하는지, 벤치마크에서 어느 쪽이 나았는지를 다룬다. 수치는 모두 [Phase 6 벤치마크](reports/phase6_benchmark.md)에서 가져왔다. 표기가 "a / b"이면 한국어 / 영어다.

## 1. 한눈에 보기

| 평가 항목 | RAGAS 대응 metric | 필요한 입력 | RAGAS-JEV 주 점수 | 벤치마크 결과 |
|---|---|---|---|---|
| **Context Precision** | `ContextPrecision` (reference 사용), reference가 없으면 `ContextRelevance` | question, contexts, (reference) | chunk별 관련 확률의 평균 | 질문만 볼 때 RAGAS-JEV 우세. reference를 볼 때는 v3 문항으로 RAGAS와 같거나 우세 |
| **Faithfulness** | `Faithfulness` | question, answer, contexts | claim별 지원 확률의 평균 | RAGAS-JEV 우세 (AUROC 0.830 vs 0.790) |
| **Context Recall** | `ContextRecall` | question, reference, contexts | reference claim별 지원 확률의 평균 | RAGAS 근소 우세 (Pearson 차이 0.006 이하) |
| **Answer Relevancy** | `AnswerRelevancy` (`ResponseRelevancy`) | question, answer | statement별 관련 확률의 평균 | RAGAS-JEV 크게 우세 (무관 문장 감지 0.98 / 0.95 vs 0.61 / 0.49) |
| **Evaluation Uncertainty** | 없음 | (위 항목의 판정 결과) | metric별 confidence·uncertainty, 저신뢰 판정 수 | RAGAS-JEV에만 있다 |

## 2. 공통 구조의 차이

네 metric 모두 "답변이나 문서를 평가 단위로 나눈다 → 단위마다 판정한다 → 점수로 합친다"는 흐름이다. RAGAS와 RAGAS-JEV는 이 흐름은 같고, 각 단계를 누가 어떻게 하는지가 다르다.

| 단계 | RAGAS | RAGAS-JEV |
|---|---|---|
| 분해 (평가 단위 만들기) | LLM이 ragas 자체 프롬프트로 분해한다. 평가할 때마다 다시 분해한다 | LLM(`gpt-6-luna`)이 버전 붙은 프롬프트로 분해한다 (`claim_extraction.v1`, `ref_claim_extraction.v1`, `statement_extraction.v1`). 결과를 캐시한다 |
| 판정 | LLM이 0/1(일부는 0/1/2)을 답한다 | **JEV**가 판정한다. Noul은 참일 확률 p(0~1)를, Score와 Choice는 확률분포를 준다. 한 샘플의 한 metric을 **요청 한 번**에 판정한다 |
| 점수 계산 | ragas 공식 | 결정적 코드([metrics.py](../src/ragas_jev/scoring/metrics.py)). 같은 판정이면 비트 단위로 같은 점수가 나온다 |
| 판정 신뢰도 | 없음 | 단위마다 confidence가 있다 (Noul은 `max(p, 1−p)`). metric마다 평균, 최솟값, 저신뢰 비율을 낸다 |
| 확률 보정 | 없음 | 선택 기능(기본 꺼짐, `--calibrate`). 질문 × 언어별 isotonic 보정 (`calibration_jev-1.13.0.json`). 도메인이 바뀌면 재보정해야 한다 |
| 재검증 (routing) | 없음 | confidence가 낮은 단위만 LLM이 다시 판정한다. metric마다 정책이 다르다 (5.2절) |
| 사람 검수 | 없음 | LLM 판정이 JEV와 반대로 나온 단위를 검수 큐(CSV)로 내보낸다. 사람 라벨을 받으면 점수를 다시 계산한다 (`rescore`) |
| 개인정보 | 없음 | 선택 기능(기본 꺼짐, `--pii-masking`). 켜면 외부 호출 전에 PII를 마스킹하고, 마스킹이 실패한 샘플은 보내지 않는다 (`status=blocked_pii`) |
| 재현성 (같은 입력 2회, Faithfulness 응답 판정이 뒤집힌 비율) | 6.2% | JEV 0.0%. 캐시를 쓰면 완전히 같다 |
| 샘플당 판정용 LLM 호출 | 1~5회 | 0~0.43회 (Hybrid). JEV만 쓰면 0회 |

설계방향 문서의 표현대로, RAGAS-JEV는 "LLM은 문제를 나누고, JEV는 판단하고, 프로그램은 점수를 계산한다".

## 3. metric별 비교

### 3.1 Context Precision

**측정 대상**: 검색된 context 중 쓸모 있는 것의 비율과, 쓸모 있는 context가 상위에 있는지.

| 구분 | RAGAS `ContextPrecision` | RAGAS-JEV |
|---|---|---|
| 평가 단위 | context 하나 | context(chunk) 하나 |
| 판정 | context마다 LLM 1회: "이 context가 reference 답변을 만드는 데 쓰였는가" (0/1) | 한 요청에 모든 chunk. Noul 문항 `chunk_relevance.v3` |
| 판정 문항 | ragas 내부 프롬프트 | v3: "Does `contexts[i]` contain information used in `reference`, the answer to `question`?" (reference가 없으면 v2: "Does `contexts[i]` contain information that answers `question`?") |
| 점수 | Average Precision: Σ(precision@k × v_k) / (관련 context 수) | 주 점수 `context_precision` = mean(p). 함께 내는 값: `_binary`, `_rank`, `_ap` |
| reference가 없을 때 | `ContextPrecision`은 쓸 수 없다. 대안인 `ContextRelevance`는 contexts **전체**를 0/0.5/1로 매긴다 (정의가 다르다) | v2 문항으로 자동 전환한다. 결과의 `question_id`에 실제로 쓴 문항이 남는다 |
| 샘플당 LLM 호출 | context 수만큼 (벤치마크에서 5회) | 0회 (JEV 1회). Hybrid는 0.28~0.42회 |

RAGAS-JEV의 추가 점수:

| 필드 | 계산 | 용도 |
|---|---|---|
| `context_precision` | mean(p) | 주 점수 (soft precision) |
| `context_precision_binary` | p ≥ 0.5인 chunk의 비율 | 판정 기준을 명확히 할 때 |
| `context_precision_rank` | Σ(w_i × p_i) / Σ w_i, w = (K−i+1)/K (K=5이면 1.0, 0.8, …, 0.2) | 순위를 선형 가중으로 반영 |
| `context_precision_ap` | p ≥ 0.5를 관련으로 보고 RAGAS와 같은 AP 공식 | RAGAS와 직접 비교할 때 |

**벤치마크**

| 조건 | RAGAS | JEV만 | Hybrid | 해석 |
|---|---|---|---|---|
| 질문만 봄 (MIRACL, 사람 라벨의 관련 비율과 Spearman) | 0.111 / 0.253 (`ContextRelevance`) | 0.621 / 0.544 | **0.681 / 0.567** | RAGAS는 관련 passage가 하나만 있어도 점수를 높게 준다. 전부 무관한 세트의 평균 점수도 0.586이다 (Hybrid 0.056, 정답 0) |
| reference 봄 (사람 라벨로 계산한 AP와 Kendall) | 0.459 / 0.518 | v3: 0.518 / 0.523 | v3 + 재보정: **0.537 / 0.545** | reference를 쓰는 v3 문항으로 격차가 사라졌다. AP Pearson은 영어에서만 RAGAS가 약간 높다 (0.696 vs 0.672) |

**주의**: 기본 보정 파일에는 v3용 보정이 없다. 그래서 v3는 JEV 원래 확률을 그대로 쓴다. 관련 비율이 다른 데이터에 v2 기본 보정을 쓰면 오히려 성능이 가장 나빠졌다 (AP Pearson 0.390 / 0.516). 서비스 도메인에서는 재보정해야 한다.

### 3.2 Faithfulness

**측정 대상**: 답변 내용이 context로 뒷받침되는 정도 (hallucination 탐지).

| 구분 | RAGAS `Faithfulness` | RAGAS-JEV |
|---|---|---|
| 평가 단위 | 답변 문장을 나눈 statement | 원자적 claim. 하나의 claim은 사실 하나만 담는다. 대명사와 생략된 주어는 풀어 쓰고, 숫자·날짜·이름은 그대로 옮긴다 |
| 분해에서 빼는 것 | 규칙이 없다 | 인사, "모르겠다" 같은 거절, 출처 언급("제공된 문서에 따르면"), 사실이 없는 헤지 |
| 판정 | LLM 1회가 모든 statement를 context와 비교해 0/1 NLI 판정 | Noul 문항 `claim_support.v1`: "Is `claim` fully supported by `contexts`?" 모든 context를 state 하나에 넣는다 (joint). 20,000자를 넘으면 나눠서 판정하고 claim별 최댓값을 쓴다 |
| 점수 | 지원된 statement 수 / 전체 statement 수 | 주 점수 `faithfulness` = mean(p). `faithfulness_binary` = p ≥ 0.5인 claim의 비율 |
| 수치 검사 | 없음 | claim의 숫자가 context에 있는지 코드로 대조한다. 만·억·조와 thousand·million 같은 단위도 정규화한다. 진단용으로만 쓴다: `faithfulness_numeric_mismatch_ratio`, `faithfulness_numeric_guarded` (불일치 claim의 p를 0.2로 제한한 점수) |
| 샘플당 LLM 호출 | 2회 (분해 1 + 판정 1) | 분해 1회 (캐시됨) + 판정 0회. Hybrid는 판정에 평균 0.34회를 더 쓴다 |

**벤치마크 (응답 단위 hallucination 탐지)**

| 데이터 | RAGAS | LLM Judge (`gpt-6-sol`) | JEV만 | Hybrid |
|---|---|---|---|---|
| RAGTruth QA AUROC | 0.790 | 0.821 | 0.809 | **0.830** |
| RAGTruth P / R / F1 | 0.623 / **0.950** / 0.752 | 0.648 / 0.930 / 0.764 | 0.706 / 0.867 / 0.778 | **0.750** / 0.835 / **0.790** |
| 한국어 합성 AUROC | 0.944 | 0.969 | 0.962 | **0.970** |

- RAGAS는 recall이 가장 높지만 precision이 가장 낮다. hallucination이 없는 응답도 많이 잡는다.
- LLM Judge는 RAGAS와 판정 모델이 같다. 그런데도 LLM Judge가 RAGAS보다 높다. 판정자가 같을 때는 RAGAS-JEV의 claim 분해가 RAGAS의 문장 분해보다 판정에 유리하다는 뜻이다.

### 3.3 Context Recall

**측정 대상**: reference 답변에 필요한 정보가 검색된 context에 들어 있는 정도.

| 구분 | RAGAS `ContextRecall` | RAGAS-JEV |
|---|---|---|
| 평가 단위 | reference의 **문장** | reference를 나눈 원자적 claim |
| 판정 | LLM 1회가 reference 문장마다 "context에서 나온 것인가"를 0/1로 판정 | Noul 문항 `ref_claim_coverage.v1`: "Is `claim` fully supported by `contexts`?" Faithfulness와 판정 방식이 같다 |
| 점수 | context로 뒷받침되는 문장 수 / 전체 문장 수 | 주 점수 `context_recall` = mean(p). `_binary` 점수와 수치 검사 진단값도 함께 낸다 |
| routing | 해당 없음 | 하지 않는다. 보정만으로 충분했다 |
| 샘플당 LLM 호출 | 1회 | 분해 1회 (캐시됨) + 판정 0회 |

**벤치마크 (기대 recall과의 Pearson)**: RAGAS **0.909 / 0.944**, Hybrid 0.903 / 0.938, LLM Judge 0.893 / 0.944, JEV만 0.878 / 0.920. context를 모두 뺀 경우와 모두 둔 경우는 네 시스템 모두 완벽히 구분한다 (AUROC 1.000).

RAGAS가 근소하게 앞선다. 이 벤치마크의 정답은 "reference 문장 중 출처 passage가 남은 비율"이라, 문장 단위로 판정하는 RAGAS 방식과 정확히 맞는다. RAGAS-JEV는 문장을 더 잘게 나누므로, 한 문장에서 나온 claim 중 일부만 뒷받침되면 정답과 어긋날 수 있다.

### 3.4 Answer Relevancy

**측정 대상**: 답변이 질문에 맞게 답하는 정도. 사실 여부와는 따로 본다.

| 구분 | RAGAS `AnswerRelevancy` | RAGAS-JEV |
|---|---|---|
| 방식 | LLM이 답변에서 질문을 여러 개 역으로 생성한다. 원래 질문과의 임베딩 코사인 유사도를 평균한다. 답변이 답을 회피하면(noncommittal) 0점이다 | 답변을 statement로 나눈 뒤 statement마다 관련성을 판정한다. 별도로 답변 전체에 4단계 척도를 한 번 매긴다 |
| 판정 문항 | 질문 생성 프롬프트 (판정 문항 없음) | `statement_relevance.v2` (Noul): "Is `statement` relevant to answering `question`? Judge relevance only, not whether it is true." / `answer_relevance_scale.v2` (Score, 0~3단계): Unrelated, Partially relevant, Mostly relevant, Direct and complete |
| 점수 | mean(cos sim) × (회피 여부) | 주 점수 `answer_relevancy` = statement p의 평균. 보조 점수 `answer_relevancy_scale` = 척도 기댓값 / 3 |
| 필요한 외부 자원 | LLM + 임베딩 모델 | JEV만 (임베딩 불필요) |
| routing | 해당 없음 | 하지 않는다. unit 단위로 검증하지 않았다 |

**벤치마크 (쌍 정확도, 높을수록 좋음)**

| 비교 | RAGAS | LLM Judge | JEV (= Hybrid) |
|---|---|---|---|
| WikiEval 좋은 답 > 부족한 답 | 0.720 | 0.780 | **0.920** |
| 원본 > 무관 문장 삽입 | 0.605 / 0.490 | 0.840 / 0.780 | **0.980 / 0.950** |
| 원본 > 답하지 않는 답변 | 0.900 / 0.960 | 0.950 / 0.930 | **0.970 / 0.980** |
| 원본과 틀린 답의 평균 점수 차 (0에 가까울수록 좋음) | 0.136 / 0.047 | −0.008 / −0.002 | 0.013 / 0.012 |

- RAGAS는 답변에 무관한 문장이 섞여도 거의 감지하지 못한다 (우연 수준).
- RAGAS는 틀린 답변에 낮은 점수를 준다. 관련성 점수가 사실 여부에 영향을 받는 것이다.
- RAGAS는 벤치마크에서 로컬 다국어 임베딩(`paraphrase-multilingual-MiniLM-L12-v2`)으로 실행했다. 프록시에 임베딩 엔드포인트가 없어서다. 공식 구성(OpenAI 임베딩)보다 불리할 수 있다.

### 3.5 Evaluation Uncertainty (RAGAS에 없음)

RAGAS 점수에는 "이 판정을 얼마나 믿을 수 있는가"가 없다. RAGAS-JEV는 단위마다 JEV 확률에서 신뢰도를 계산하고, 이를 metric과 샘플 단위로 요약한다.

| 필드 | 뜻 |
|---|---|
| `confidence[metric]` | 그 metric 단위들의 confidence 평균. Noul은 `max(p, 1−p)`, Score/Choice는 API가 준 값 |
| `uncertainty[metric]` | 1 − confidence |
| `escalation` | 전체 판정 수, 0.85 미만 / 0.60 미만 개수, Auditor·Strong Judge 재판정 수, 사람 검수 대상 수, routing 실패 수 |
| `units[]` | 단위마다 원래 JEV p(`jev_p`), 보정된 p, 판정 출처(`decision_source`: `jev`, `llm_auditor`, `strong_judge`, `human`), 문항 id |

점수가 같아도 불확실성이 높은 샘플은 따로 골라 검토할 수 있다. 판정 근거도 단위별로 추적할 수 있다.

## 4. 결과 필드 대응

| RAGAS 결과 | RAGAS-JEV 결과 (`SampleResult`) |
|---|---|
| `context_precision` | `retrieval.context_precision_ap` (같은 공식). 주 점수는 `retrieval.context_precision` (soft) |
| `context_recall` | `retrieval.context_recall_binary` (단위 비율). 주 점수는 `retrieval.context_recall` (soft) |
| `faithfulness` | `generation.faithfulness_binary` (단위 비율). 주 점수는 `generation.faithfulness` (soft) |
| `answer_relevancy` | 정의가 다르다. `generation.answer_relevancy` (statement 기반), `generation.answer_relevancy_scale` (척도 기반) |
| (없음) | `confidence`, `uncertainty`, `escalation`, `usage`(JEV·LLM 요청 수와 토큰), `units`, `status`, `reasons` |

RAGAS-JEV에서 계산할 수 없는 항목은 점수가 `null`이고, 이유가 `reasons`에 남는다. 이유 값은 `no_reference`, `no_contexts`, `no_claims` 등이다.

## 5. 운영 정책

### 5.1 기본 판정 문항

| metric | 기본 문항 | 이전 버전 |
|---|---|---|
| Context Precision | `chunk_relevance.v3`. reference가 없으면 v2 | v1: 너무 느슨했다. 같은 주제만 다뤄도 관련으로 판정했다 |
| Faithfulness | `claim_support.v1` (joint) | v2(엄격한 criteria)와 per_chunk 방식은 이득이 없었다 |
| Context Recall | `ref_claim_coverage.v1` | – |
| Answer Relevancy | `statement_relevance.v2`, `answer_relevance_scale.v2` | v1: 맞는 세부 설명도 무관으로 판정했고, 틀린 답에 불이익을 줬다 |

문항 문구를 바꿀 때는 기존 문항을 고치지 않고 새 버전 id를 추가한다. 예전 결과와 계속 비교할 수 있게 하기 위해서다.

### 5.2 metric별 routing 정책

| metric | accept / audit 임계값 | 재판정 대상 | 재판정 비율 (벤치마크) |
|---|---|---|---|
| Context Precision | 0.60 / 0.60 | confidence < 0.60이면 Strong Judge(`gpt-6-sol`) | unit 5.8%, 샘플당 0.40회 |
| Faithfulness | 0.65 / 0.0 | confidence < 0.65이면 Auditor(`gpt-6-luna`) | unit 8.8%, 샘플당 0.34회 |
| Context Recall | 0 / 0 | 없음 | 0% |
| Answer Relevancy | 0 / 0 | 없음 | 0% |

## 6. 한계

1. **보정은 데이터 분포에 의존한다.** 기본 보정은 공개 데이터(MIRACL 등)로 학습했다. 서비스 도메인에서는 [도메인 재보정](도메인_재보정_가이드.md) 없이 Hybrid를 쓰면 안 된다.
2. **분해는 여전히 LLM이 한다.** Faithfulness, Context Recall, Answer Relevancy는 LLM 분해 품질의 영향을 받는다. 판정 단계만 결정적이다.
3. **벤치마크 데이터의 일부는 합성이다.** 한국어 Faithfulness, Context Recall 전체, reference 기반 Context Precision의 reference, Answer Relevancy의 MIRACL 세트는 `gpt-6-astra`로 만들었다.
4. **도메인이 일반적이다.** 모두 Wikipedia와 웹 기반 데이터다. 금융·의료 같은 전문 도메인에서는 다시 검증해야 한다.
5. **외부 호출이 모두 국외로 간다.** JEV(TypeSafe)와 LLM 프록시(OpenAI)가 모두 국외 서버다. PII 마스킹은 기본으로 꺼져 있고 켜도 규칙 기반이다. 실제 고객 데이터를 쓰려면 마스킹을 켜고, 보안·법무 검토를 먼저 받아야 한다.
