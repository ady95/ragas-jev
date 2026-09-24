# Phase 3 검증 리포트: Context Recall

- 실행일: 2026-09-24
- JEV 모델: `jev-1.13.0`, 문항 `ref_claim_coverage.v1`, contexts `joint`
- reference claim 추출: `gpt-6-luna` + `ref_claim_extraction.v1`. 모든 구성이 **같은 claim**을 판정한다
- 비교 LLM Judge: 기준선 `gpt-6-sol`, Auditor `gpt-6-luna` (같은 v1 문항, `seed=7`, 결정적이지 않음)
- 관련 문서: [구현계획서](../구현계획서.md) Phase 3, [Phase 2 리포트](phase2_faithfulness.md)

## 1. 요약

| 항목 | 결과 |
|---|---|
| context 제거에 대한 반응 | **세 Judge 모두 정상.** 관련 passage를 모두 빼면 recall이 0.95 → 0.22~0.27(JEV)로 떨어지고, full과 none을 완벽히 구분한다 (AUROC 1.000). 하나만 빼도 90~97%의 질문에서 recall이 떨어진다 |
| claim 단위 판정 (0.5 기준) | JEV F1 한국어 0.891 / 영어 0.910, `gpt-6-sol` 0.907 / 0.936. JEV가 0.016 / 0.026 낮다 |
| claim AUROC | JEV 0.940 / 0.952, `gpt-6-sol` 0.949 / 0.966 |
| JEV의 경향 | 관련 passage가 없는 조건에서 **JEV가 "지원됨"을 더 많이 준다** (한국어 25% vs `gpt-6-sol` 20%). 차이의 대부분은 p 0.5~0.8의 애매한 판정이다 |
| threshold | JEV를 **0.8 기준으로 판정하면** 한국어 F1 0.911로 `gpt-6-sol`(0.907)을 넘는다. LLM은 확률이 거의 0/1이라 threshold에 둔감하다 |
| 데이터 한계 | MIRACL에서 "무관"으로 라벨된 passage가 실제로 답을 담고 있는 경우가 많다. none 조건의 "지원됨" 판정 중 상당수(한국어 130건 / 영어 101건)는 두 Judge가 모두 동의해 라벨 잡음으로 보인다 |
| 속도·비용 | JEV p50 0.29초, LLM 4.2~4.8초. JEV 비용 한국어 $0.022, 영어 $0.020 (300 샘플). reference claim 추출은 LLM p50 8~9초 |

**판단**: Phase 1·2와 같은 기준(기준선 `gpt-6-sol` F1 − 0.05 이내, AUROC ≥ 0.85)으로 **통과**다 (F1 −0.016 / −0.026, AUROC 0.940 / 0.952). 계획서의 검증 항목("reference 누락 context를 제거한 샘플에서 recall 하락 확인")도 충족한다.

## 2. 데이터셋

[prepare_recall.py](../../benchmark/prepare_recall.py)로 MIRACL dev에서 만들었다. 합성 reference이며 사람이 검수하지 않았다.

| 데이터셋 | 질문 수 | 샘플 | reference claim | 만든 방법 |
|---|---|---|---|---|
| `recall_miracl_ko` | 100 | 300 | 680 (6.8개/reference) | MIRACL-ko dev에서 관련 passage가 2개 이상인 질문 |
| `recall_miracl_en` | 100 | 300 | 856 (8.6개/reference) | MIRACL-en dev에서 같은 조건 |

만든 방법:

1. `gpt-6-astra`가 관련 passage(최대 3개)만 써서 reference를 쓴다. 모든 passage에서 최소 한 문장씩 가져오고, **문장마다 출처 passage 번호를 함께 받는다.**
2. 질문마다 contexts를 세 가지로 만든다. passage 수는 5개로 같다.

| 조건 | contexts |
|---|---|
| full | reference에 쓴 관련 passage 전부 + 무관 passage |
| partial | 관련 passage 하나를 무관 passage로 교체 |
| none | 관련 passage를 모두 무관 passage로 교체 |

3. claim 단위 기대값: reference claim의 원문 구간이 속한 문장의 출처 passage가 contexts에 남아 있으면 "지원돼야 함", 없으면 "지원되면 안 됨".

주의:

- 무관 passage는 MIRACL annotator가 무관으로 판정한 같은 주제의 passage다. Phase 1에서 봤듯이 MIRACL 라벨은 엄격해서, 무관 passage가 답을 담고 있는 경우가 있다 (예: 질문 "그리스의 수도는?"의 무관 passage "[그리스] 수도는 아테네이다."). 따라서 **none 조건의 기대 recall은 0이 아니다.**
- 여러 passage가 같은 사실을 담고 있으면 partial 조건의 claim 라벨이 틀릴 수 있다.

## 3. 결과

"full에서 미지원 판정"은 지원돼야 할 claim을 미지원으로 본 비율, "none에서 지원 판정"은 관련 passage가 없는데 지원됨으로 본 비율이다. "단조 감소 비율"은 recall(full) ≥ recall(partial) ≥ recall(none)을 만족한 질문 비율이다.

#### recall_miracl_ko (300 samples, reference 100개)

reference claim 6.80개/reference

| 구성 | recall full | recall partial | recall none | AUROC full vs none | AUROC full vs partial | 단조 감소 비율 | partial에서 하락 비율 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|---|---|---|---|---|---|---|
| jev.claim_support.v1.joint | 0.954 | 0.718 | 0.274 | 1.000 | 0.909 | 0.930 | 0.920 | 0.29 | 0.0219 | 0 |
| llm-gpt-6-sol.claim_support.v1.joint | 0.990 | 0.713 | 0.204 | 1.000 | 0.922 | 0.940 | 0.890 | 4.42 | – | 0 |
| llm-gpt-6-luna.claim_support.v1.joint | 0.987 | 0.711 | 0.195 | 0.999 | 0.908 | 0.930 | 0.860 | 4.21 | – | 0 |

| 구성 | claim AUROC | claim P / R / F1 (지원됨 예측) | ECE | full에서 미지원 판정 | none에서 지원 판정 | partial claim AUROC | 구간 불일치 claim |
|---|---|---|---|---|---|---|---|
| jev.claim_support.v1.joint | 0.940 | 0.808 / 0.993 / 0.891 | 0.116 | 0.7% | 24.9% | 0.906 | 36 / 2040 |
| llm-gpt-6-sol.claim_support.v1.joint | 0.949 | 0.836 / 0.992 / 0.907 | 0.102 | 0.9% | 20.2% | 0.899 | 36 / 2040 |
| llm-gpt-6-luna.claim_support.v1.joint | 0.932 | 0.837 / 0.990 / 0.907 | 0.099 | 0.9% | 19.5% | 0.887 | 36 / 2040 |

#### recall_miracl_en (300 samples, reference 100개)

reference claim 8.56개/reference

| 구성 | recall full | recall partial | recall none | AUROC full vs none | AUROC full vs partial | 단조 감소 비율 | partial에서 하락 비율 | p50 지연(s) | 비용(USD) | 오류 |
|---|---|---|---|---|---|---|---|---|---|---|
| jev.claim_support.v1.joint | 0.953 | 0.703 | 0.222 | 1.000 | 0.962 | 0.980 | 0.970 | 0.28 | 0.0198 | 0 |
| llm-gpt-6-sol.claim_support.v1.joint | 0.989 | 0.682 | 0.125 | 1.000 | 0.971 | 1.000 | 0.970 | 4.82 | – | 0 |
| llm-gpt-6-luna.claim_support.v1.joint | 0.992 | 0.684 | 0.134 | 1.000 | 0.969 | 0.980 | 0.960 | 4.42 | – | 0 |

| 구성 | claim AUROC | claim P / R / F1 (지원됨 예측) | ECE | full에서 미지원 판정 | none에서 지원 판정 | partial claim AUROC | 구간 불일치 claim |
|---|---|---|---|---|---|---|---|
| jev.claim_support.v1.joint | 0.952 | 0.840 / 0.993 / 0.910 | 0.091 | 0.8% | 17.6% | 0.915 | 39 / 2568 |
| llm-gpt-6-sol.claim_support.v1.joint | 0.966 | 0.885 / 0.993 / 0.936 | 0.067 | 0.6% | 11.6% | 0.914 | 39 / 2568 |
| llm-gpt-6-luna.claim_support.v1.joint | 0.957 | 0.880 / 0.993 / 0.933 | 0.071 | 0.6% | 12.3% | 0.901 | 39 / 2568 |

### 3.1 threshold별 claim 판정 (JEV vs `gpt-6-sol`)

| 데이터셋 | Judge | threshold | F1 (지원됨) | full 미지원 | none 지원 |
|---|---|---|---|---|---|
| recall_miracl_ko | JEV | 0.5 | 0.891 | 0.7% | 24.9% |
| recall_miracl_ko | JEV | 0.6 | 0.897 | 0.9% | 22.2% |
| recall_miracl_ko | JEV | 0.7 | 0.904 | 1.3% | 19.9% |
| recall_miracl_ko | JEV | 0.8 | **0.911** | 2.4% | 16.3% |
| recall_miracl_ko | gpt-6-sol | 0.5 | 0.907 | 0.9% | 20.2% |
| recall_miracl_ko | gpt-6-sol | 0.8 | 0.909 | 1.0% | 19.5% |
| recall_miracl_en | JEV | 0.5 | 0.910 | 0.8% | 17.6% |
| recall_miracl_en | JEV | 0.6 | 0.917 | 1.3% | 15.7% |
| recall_miracl_en | JEV | 0.7 | 0.920 | 2.1% | 13.3% |
| recall_miracl_en | JEV | 0.8 | 0.921 | 3.6% | 10.8% |
| recall_miracl_en | gpt-6-sol | 0.5 | 0.936 | 0.6% | 11.6% |
| recall_miracl_en | gpt-6-sol | 0.8 | **0.939** | 0.6% | 11.0% |

### 3.2 none 조건에서 두 Judge의 판정 비교

| 데이터셋 | 둘 다 지원됨 | JEV만 지원됨 | `gpt-6-sol`만 지원됨 | 둘 다 미지원 |
|---|---|---|---|---|
| recall_miracl_ko | 130 | 38 | 7 | 505 |
| recall_miracl_en | 101 | 52 | 1 | 702 |

## 4. 해석

### 4.1 context 제거를 잘 감지한다

- 관련 passage를 모두 빼면 세 Judge 모두 recall이 크게 떨어지고, full과 none을 완벽히 구분한다 (AUROC 1.000).
- 하나만 빼도 한국어 92%, 영어 97%의 질문에서 JEV recall이 떨어졌다. 떨어지지 않은 질문은 남은 passage가 같은 사실을 담고 있었을 가능성이 크다.
- full 조건에서 지원돼야 할 claim을 놓친 비율은 1% 미만이다. reference claim 추출이 원문 passage의 사실을 잘 보존한다는 뜻이기도 하다.

### 4.2 none 조건의 "지원됨" 판정

- 두 Judge가 **모두** 지원됨으로 본 claim(한국어 130 / 영어 101)은 대부분 무관 passage에 실제로 그 사실이 있는 경우로 보인다. 예: "Hassium has no naturally occurring isotopes"는 무관 passage의 "Hassium is a synthetic element"가 뒷받침한다.
- **JEV만** 지원됨으로 본 claim(38 / 52)은 p가 0.5~0.8인 애매한 판정이 많았다. 두 유형이 보인다.
  - 세상 지식: "Konami published Silent Hill" (JEV 0.66, `gpt-6-sol` 0.01). contexts에는 게임 이름만 있다.
  - 부분 근거에 관대함: "Sara Paxton plays Mari Collingwood in the 2009 remake" (JEV 0.82). contexts에는 리메이크에 출연했다는 내용만 있고 배역명은 없다.
- Phase 2(Faithfulness)에서 본 "세상 지식으로 판단" 약점과 같은 현상이다.

### 4.3 JEV p는 "지원됨" 쪽으로 치우쳐 있다

- JEV는 threshold를 올릴수록 F1이 오르고 0.8에서 가장 좋다. LLM은 threshold에 거의 변화가 없다.
- full 조건의 soft recall도 JEV 0.954, LLM 0.99로 JEV가 낮다. 지원되는 claim에도 JEV는 1보다 낮은 p(0.8~0.95)를 자주 준다. 반대로 애매한 claim에는 0.5를 넘는 p를 준다. 즉 **JEV의 p는 0.5 기준보다 높은 곳에서 갈린다.**
- Phase 1(Context Precision)에서도 최적 threshold가 0.85였다. Phase 5 calibration에서 metric별·언어별 판정 threshold를 정하는 근거가 된다.

## 5. 결정 사항과 제안

**적용한 결정**

1. Context Recall은 Faithfulness와 같은 방식을 쓴다: `ref_claim_coverage.v1`, contexts `joint`, 판정 threshold 0.5 (코드 기본값 그대로).

**제안 (확인 필요)**

1. **Phase 3 통과 판정**: Phase 1·2와 같은 기준으로 통과다.
2. **Phase 5 calibration 항목 추가**: JEV support 판정 threshold 후보 0.8 (이 데이터 기준 한국어 F1 +0.020, none 조건 오판 −8.6%p). Faithfulness에서는 threshold를 올리면 hallucination 탐지 recall이 오르고 precision이 떨어지므로, 두 metric에서 함께 봐야 한다.
3. **세상 지식 배제 문항 (`claim_support.v3`)**: Phase 2 제안과 같다. Recall에서도 같은 약점이 보였으므로 두 metric에 함께 A/B한다.

## 6. 재현 방법

```bash
uv run python benchmark/prepare_miracl.py --lang ko
uv run python benchmark/prepare_miracl.py --lang en --max-queries 213
uv run python benchmark/prepare_recall.py --lang ko --queries 100   # gpt-6-astra로 생성 (실행마다 결과가 다름)
uv run python benchmark/prepare_recall.py --lang en --queries 100
uv run python benchmark/run_phase3.py
uv run python benchmark/run_phase3.py --render-only
```

원시 결과와 추출 캐시는 `.cache/phase3/`에 저장된다 (git 제외). 영어 MIRACL 세트는 Phase 1에서 무작위로 뽑은 213문항 중 조건을 만족하는 질문에서 다시 뽑았다.
