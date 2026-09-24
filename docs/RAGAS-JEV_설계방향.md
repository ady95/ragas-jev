# RAGAS-JEV 설계 방향

## 1. 목적

본 문서는 RAGAS 기반의 RAG 평가 체계를 **JEV를 Primary Judge로 사용하도록 재설계**하는 방향을 정리한 문서이다.

핵심 목표는 다음과 같다.

- 기존 RAGAS의 주요 평가 축을 유지한다.
- JEV를 최종 판단 모델(Primary Judge)로 사용한다.
- 생성형 LLM은 평가 단위 생성, 분해, 정규화에 사용한다.
- 필요 시 LLM-as-a-Judge를 불확실 케이스에 대한 Secondary Judge / Auditor로 사용한다.
- 최종 점수 계산은 deterministic scoring으로 처리한다.
- 평가 결과에 score뿐 아니라 uncertainty / confidence를 포함한다.

---

## 2. 핵심 설계 원칙

RAGAS-JEV의 핵심은 다음의 역할 분리이다.

```text
Generative LLM
    ↓
평가 대상 구조화
    ↓
JEV
    ↓
Primary Decision
    ↓
Deterministic Scoring
```

이를 보다 구체적으로 표현하면 다음과 같다.

> LLM은 평가할 문제를 잘게 나누고,  
> JEV는 각 문제를 판단하며,  
> 프로그램은 최종 점수를 계산한다.

JEV만으로 모든 평가 과정을 처리하는 방식보다는,  
**생성형 LLM + JEV + deterministic scoring**의 Hybrid 구조가 더 적합하다.

---

## 3. 왜 JEV만으로는 충분하지 않은가

JEV는 다음과 같은 bounded decision에 매우 적합하다.

```text
이 claim은 context에서 지원되는가?

YES / NO

P(YES) = 0.94
P(NO)  = 0.06
```

또는:

```text
이 context chunk는 질문에 relevant한가?
```

하지만 다음과 같은 작업은 JEV보다 생성형 LLM이 더 적합하다.

- 긴 Answer를 atomic claim으로 분해
- Reference Answer를 평가 가능한 statement로 분해
- 문장 정규화
- 중복 claim 제거
- 평가 단위 생성
- 복합 답변을 독립된 factual unit으로 변환

예를 들어 다음 Answer가 있다고 가정한다.

```text
아인슈타인은 1879년 독일 울름에서 태어났고
이후 스위스 특허청에서 근무했다.
```

Faithfulness를 제대로 평가하려면 다음과 같이 분해하는 것이 좋다.

```text
claim 1:
Einstein was born in 1879.

claim 2:
Einstein was born in Ulm, Germany.

claim 3:
Einstein worked at the Swiss Patent Office.
```

이러한 decomposition은 Generative LLM이 담당하고,
각 claim의 support 여부는 JEV가 판단하는 구조가 적절하다.

---

## 4. 최종 권장 아키텍처

```text
                       RAG Output
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
       Question          Answer           Context[]
          │                 │                  │
          └─────────────────┼──────────────────┘
                            │
                            ▼
                   Generative LLM
                Evaluation Preprocessor
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Claims       Statements    Ref Claims
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                           JEV
                     Primary Judge
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           support       relevance    attribution
              │             │             │
              └─────────────┼─────────────┘
                            │
                    probabilities
                            │
            ┌───────────────┴────────────────┐
            │                                │
       confidence OK                  uncertainty high
            │                                │
            │                           LLM Judge
            │                            Auditor
            │                                │
            └───────────────┬────────────────┘
                            ▼
                  Deterministic Scorer
                            │
         ┌──────────────────┼──────────────────┐
         ↓                  ↓                  ↓
   Faithfulness      Context Precision   Context Recall
                                              +
                                      Answer Relevancy
```

---

## 5. 구성 요소별 역할

| 구성요소 | 역할 | 최종 평가 영향 |
|---|---|---|
| Generative LLM | claim/statement 생성, decomposition, normalization | 간접 |
| JEV | Primary Judge, bounded decision | 직접 |
| LLM-as-a-Judge | 불확실 케이스 보조 검증 | 선택적 |
| Deterministic Scorer | metric 공식 계산 | 직접 |

핵심 원칙은 다음과 같다.

- Generative LLM은 Judge가 아니라 Preprocessor로 사용한다.
- JEV가 가능한 한 모든 최종 판단을 수행한다.
- LLM-as-a-Judge는 모든 샘플에 사용하지 않는다.
- uncertainty가 높은 샘플만 선택적으로 escalation한다.

---

# 6. RAGAS-JEV 핵심 Metric

RAGAS-JEV는 다음 4개 핵심 metric을 중심으로 구성한다.

```text
RAGAS-JEV
│
├── Retrieval Quality
│   ├── Context Precision
│   └── Context Recall
│
├── Generation Quality
│   ├── Faithfulness
│   └── Answer Relevancy
│
└── Judge Reliability
    └── Evaluation Uncertainty
```

---

# 7. Faithfulness

## 7.1 목적

Generated Answer의 각 factual claim이 Retrieval Context에 의해 지원되는지를 평가한다.

---

## 7.2 권장 처리 구조

```text
Answer
  ↓
Generative LLM
  ↓
Atomic Claim Extraction
  ↓
claim 1
claim 2
claim 3
...
  ↓
JEV
  ↓
P(Supported)
```

예:

```text
Answer:

A사는 2025년에 매출 3조원을 기록했고
영업이익은 5,000억원으로 전년 대비 20% 증가했다.
```

LLM decomposition:

```text
claim 1:
매출은 3조원이다.

claim 2:
영업이익은 5,000억원이다.

claim 3:
영업이익은 전년 대비 20% 증가했다.
```

JEV 결과:

```text
claim 1 → 0.98
claim 2 → 0.94
claim 3 → 0.08
```

---

## 7.3 계산

확률 기반 Soft Faithfulness를 사용할 수 있다.

```text
Faithfulness =
mean(P(claim supported))
```

즉:

```text
(0.98 + 0.94 + 0.08) / 3
= 0.667
```

Binary threshold 방식도 병행할 수 있다.

```text
P >= 0.5 → Supported
P < 0.5  → Unsupported
```

권장 저장 값:

```text
faithfulness_binary
faithfulness_probabilistic
```

---

# 8. Context Precision

## 8.1 목적

Retriever가 반환한 각 Context Chunk가 실제 질문에 유용한지를 평가하고,
관련 chunk가 상위 rank에 위치하는지도 평가한다.

---

## 8.2 처리 구조

Context Precision은 RAGAS-JEV metric 중 가장 JEV-only에 가까운 항목이다.

```text
Question
+
Context #1 → relevant?
Context #2 → relevant?
Context #3 → relevant?
...
```

JEV는 각 chunk에 대해 relevance probability를 반환한다.

예:

```text
chunk1 → 0.98
chunk2 → 0.76
chunk3 → 0.08
chunk4 → 0.43
chunk5 → 0.91
```

---

## 8.3 Soft Context Precision

```text
Context Precision Soft =
mean(P(relevant))
```

예:

```text
(0.98 + 0.76 + 0.08 + 0.43 + 0.91) / 5
= 0.632
```

---

## 8.4 Rank-aware Precision

Retrieval ranking까지 평가하려면 rank weight를 적용한다.

예:

```text
rank 1 → 1.00
rank 2 → 0.80
rank 3 → 0.60
rank 4 → 0.40
rank 5 → 0.20
```

계산:

```text
ContextPrecisionRank =
Σ(rank_weight_i × P(relevant_i))
/
Σ(rank_weight_i)
```

이 방식은 단순 relevance뿐 아니라
**관련 문서가 상위에 검색되었는지**까지 반영한다.

---

# 9. Context Recall

## 9.1 목적

Reference Answer를 생성하는 데 필요한 정보가 Retrieval Context에 얼마나 포함되어 있는지 평가한다.

---

## 9.2 처리 구조

```text
Reference Answer
      ↓
Generative LLM
      ↓
Reference Claim Extraction
      ↓
claim 1
claim 2
claim 3
...
      ↓
JEV
      ↓
P(claim supported by retrieval context)
```

예:

```text
Reference Answer:

A사의 제품은 A, B, C이고
가격은 각각 100, 200, 300이다.
```

LLM decomposition:

```text
claim 1: Product A exists
claim 2: Product B exists
claim 3: Product C exists
claim 4: A price = 100
claim 5: B price = 200
claim 6: C price = 300
```

JEV는 각 claim이 Retrieval Context에 의해 지원되는지 판단한다.

---

## 9.3 계산

```text
ContextRecall =
mean(P(reference claim supported))
```

또는 binary 방식:

```text
supported claims / total reference claims
```

두 값을 동시에 저장하는 것이 좋다.

---

# 10. Answer Relevancy

## 10.1 목적

Generated Answer가 Question에 얼마나 직접적이고 관련성 있게 응답했는지 평가한다.

---

## 10.2 권장 구조

전체 답변을 한 번에 평가할 수도 있지만,
diagnostic capability를 높이려면 statement 단위로 분해하는 것이 좋다.

```text
Answer
  ↓
Generative LLM
  ↓
statement 1
statement 2
statement 3
statement 4
  ↓
JEV
  ↓
P(relevant)
```

예:

```text
statement #1 → 0.98
statement #2 → 0.96
statement #3 → 0.21
statement #4 → 0.93
```

이 경우 statement #3이 Answer Relevancy를 떨어뜨리는 원인이라는 것을
정확히 추적할 수 있다.

---

## 10.3 계산

```text
AnswerRelevancy =
mean(P(statement relevant))
```

또는 JEV Score primitive를 사용해 ordered scale로 평가할 수 있다.

예:

```text
0 Unrelated
1 Partially relevant
2 Mostly relevant
3 Direct and complete
```

확률분포:

```text
0 : 0.01
1 : 0.03
2 : 0.20
3 : 0.76
```

Expected-value 방식:

```text
AnswerRelevancy =
(0×0.01 + 1×0.03 + 2×0.20 + 3×0.76) / 3
```

---

# 11. JEV Primitive 활용

JEV의 bounded decision primitive는 다음과 같이 활용한다.

## 11.1 Noul

Yes/No 문제에 사용한다.

예:

```text
This claim is fully supported by retrieval_context.
```

출력:

```text
True  = 0.94
False = 0.06
```

활용 예:

- Claim support
- Chunk relevance
- Reference claim coverage
- Answer statement relevance

---

## 11.2 Score

Ordered scale 평가에 사용한다.

예:

```text
How completely is this answer grounded in the context?

0 Unsupported
1 Partially supported
2 Mostly supported
3 Fully supported
```

출력:

```text
0 = 0.02
1 = 0.08
2 = 0.25
3 = 0.65
```

Expected value로 0~1 score로 변환한다.

---

## 11.3 Choice

서로 다른 유형의 행동 또는 오류 유형을 구분할 때 사용한다.

예:

```text
How does the answer handle information absent from context?

- adds_no_unsupported_information
- explicitly_marks_uncertainty
- hedges
- states_unsupported_as_fact
```

Choice는 오류 유형 분석이나 policy 기반 평가에 활용한다.

---

# 12. Evaluation Uncertainty

JEV 기반 평가의 핵심 차별점 중 하나이다.

예:

```text
Case A

True  = 0.98
False = 0.02
```

과

```text
Case B

True  = 0.51
False = 0.49
```

는 threshold 0.5를 적용하면 둘 다 True이지만,
평가 신뢰도는 크게 다르다.

따라서 다음 값을 별도로 저장하는 것이 좋다.

```text
score
confidence
uncertainty
```

---

## 12.1 기본 Confidence 정의

Binary decision의 경우:

```text
confidence = max(P(True), P(False))
```

예:

```text
True  = 0.98
False = 0.02

confidence = 0.98
```

---

## 12.2 Uncertainty 정의 예시

가장 단순하게:

```text
uncertainty = 1 - confidence
```

또는 entropy 기반 uncertainty를 사용할 수도 있다.

---

# 13. LLM-as-a-Judge의 역할

LLM-as-a-Judge를 완전히 제거할 필요는 없다.

다만 모든 평가에 항상 사용하는 것보다는
**JEV가 애매하게 판단한 경우에만 Secondary Judge로 사용하는 구조**를 권장한다.

```text
                  JEV
                   │
               confidence
                   │
        ┌──────────┴──────────┐
        │                     │
   confidence OK         uncertainty high
        │                     │
        ↓                     ↓
   JEV verdict           LLM-as-a-Judge
                              Auditor
```

---

## 13.1 예시 Threshold

초기 PoC에서는 다음과 같이 설정할 수 있다.

```text
confidence >= 0.85
→ JEV 결과 그대로 사용

0.60 <= confidence < 0.85
→ LLM Judge로 재검증

confidence < 0.60
→ Strong LLM Judge 또는 Human Review
```

실제 threshold는 validation dataset을 통해 calibration해야 한다.

---

# 14. 최종 3계층 평가 구조

```text
Layer 1
Generative LLM
────────────────────

Evaluation Preprocessor

- claim extraction
- statement extraction
- reference decomposition
- normalization
- deduplication


           ↓


Layer 2
JEV
────────────────────

Primary Judge

- entailment
- support
- relevance
- attribution
- completeness
- correctness

→ probability distribution


           ↓


Layer 3
LLM-as-a-Judge
────────────────────

Optional Secondary Judge / Auditor

사용 조건:

- JEV uncertainty high
- semantic ambiguity
- JEV / ground truth disagreement
- evaluation debugging
- difficult reasoning sample


           ↓


Deterministic Metric Engine
────────────────────

- Faithfulness
- Context Precision
- Context Recall
- Answer Relevancy
```

---

# 15. 권장 결과 Schema

```json
{
  "evaluation_model": {
    "primary_judge": "JEV",
    "preprocessor": "Generative LLM",
    "secondary_judge": "Optional LLM-as-a-Judge"
  },

  "retrieval": {
    "context_precision": 0.87,
    "context_recall": 0.81
  },

  "generation": {
    "faithfulness": 0.94,
    "answer_relevancy": 0.92
  },

  "confidence": {
    "faithfulness": 0.96,
    "context_precision": 0.91,
    "context_recall": 0.69,
    "answer_relevancy": 0.94
  },

  "uncertainty": {
    "faithfulness": 0.04,
    "context_precision": 0.09,
    "context_recall": 0.31,
    "answer_relevancy": 0.06
  }
}
```

---

# 16. 구현 우선순위

PoC 구현은 다음 순서를 권장한다.

## Phase 1 — Context Precision

가장 단순하고 JEV 특성에 가장 잘 맞는 metric이다.

```text
Question + Chunk
→ JEV relevance decision
→ Precision score
```

검증 항목:

- JEV vs LLM Judge agreement
- Human Label agreement
- latency
- cost
- decision stability

---

## Phase 2 — Faithfulness

Generative LLM을 추가한다.

```text
Answer
→ Claim Extractor
→ JEV Claim Support Judge
→ Faithfulness
```

검증 항목:

- claim extraction 품질
- claim granularity
- entailment 정확도
- hallucination detection 성능

---

## Phase 3 — Context Recall

```text
Reference Answer
→ Reference Claim Extraction
→ JEV Attribution Judge
→ Recall
```

---

## Phase 4 — Answer Relevancy

```text
Answer
→ Statement Extraction
→ JEV Relevance Judge
→ Answer Relevancy
```

---

## Phase 5 — Secondary LLM Judge

불확실 케이스만 escalation한다.

```text
JEV
 ↓
uncertainty threshold
 ↓
LLM-as-a-Judge
```

이 단계에서 전체 평가 비용과 정확도의 trade-off를 측정한다.

---

# 17. Benchmark 비교 대상

RAGAS-JEV를 평가할 때 최소 다음 3개를 비교해야 한다.

```text
1. 기존 RAGAS / LLM-based Judge

2. JEV-only Decision Evaluation

3. Hybrid RAGAS-JEV
   - Generative LLM Preprocessor
   - JEV Primary Judge
   - Optional LLM Auditor
```

비교 지표:

- Human agreement
- Accuracy
- Precision / Recall / F1
- Rank correlation
- Calibration Error
- Evaluation variance
- Latency
- Cost
- Repeatability
- Uncertainty quality

---

# 18. 권장 최종 방향

RAGAS-JEV의 핵심 차별점은
**LLM을 없애는 것이 아니라 LLM의 역할과 Judge의 역할을 분리하는 것**이다.

권장 정의:

> RAGAS-JEV는 Generative LLM을 이용해 평가 대상을 구조화하고,
> JEV를 Primary Judge로 사용해 bounded probabilistic decision을 수행하며,
> deterministic scoring engine이 최종 RAGAS metric을 계산하는 평가 구조이다.

필요한 경우에만 LLM-as-a-Judge를 Secondary Judge로 사용한다.

전체 구조를 한 문장으로 요약하면 다음과 같다.

```text
LLM은 문제를 나누고,
JEV는 판단하고,
프로그램은 점수를 계산한다.
```

---

# 19. 최종 아키텍처 요약

```text
                       RAG Output
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     Question             Answer            Context[]
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    Generative LLM
                     Preprocessor
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Claims       Statements    Ref Claims
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                           JEV
                      Primary Judge
                            │
             ┌──────────────┼──────────────┐
             │              │              │
          Support        Relevance     Attribution
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                  Probability Outputs
                            │
             ┌──────────────┴──────────────┐
             │                             │
       Low Uncertainty               High Uncertainty
             │                             │
             │                      LLM-as-a-Judge
             │                         Auditor
             │                             │
             └──────────────┬──────────────┘
                            ▼
                  Deterministic Scoring
                            │
       ┌────────────────────┼─────────────────────┐
       │                    │                     │
 Faithfulness        Context Precision      Context Recall
                                                  │
                                           Answer Relevancy
```
