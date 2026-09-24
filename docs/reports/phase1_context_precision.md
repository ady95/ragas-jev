# Phase 1 검증 리포트: Context Precision

- 실행일: 2026-09-23
- JEV 모델: `jev-1.13.0` (버전 고정)
- 비교 LLM Judge (openai-oauth 프록시, `seed=7`, `temperature` 미지정)
  - `gpt-6-luna` + v2 문항: 현재 기본 LLM Judge (4.1절)
  - `gpt-6-sol` + v2 문항: 기준선 후보 비교 (4.2절)
  - `gpt-5.6-sol` + v1 문항: 최초 비교 (5장). 이때는 `temperature=0`을 보냈고 프록시가 오류 없이 받았지만, GPT-5 이후 모델에는 유효하지 않은 파라미터라 효과는 없던 것으로 본다
- GPT-5 이후 모델은 `temperature`를 지원하지 않으므로 **LLM Judge 결과는 모두 결정적이지 않다**
- 관련 문서: [구현계획서](../구현계획서.md) Phase 1

## 1. 요약

| 항목 | 결과 |
|---|---|
| 사람 라벨과의 일치 | 중간 수준. 한국어 v2 문항 기준 F1 0.571, κ 0.435. 순위 판별력(AUROC)은 0.88로 높다 |
| LLM Judge와 비교 (같은 v2 문항) | `gpt-6-luna`와는 비슷하다 (한국어 F1 0.571 동률). **`gpt-6-sol`이 가장 낫다**: JEV 대비 F1 한국어 +0.035 (0.606), 영어 +0.028 (0.670). AUROC는 JEV와 `gpt-6-sol`이 비슷하고 (0.883 vs 0.876), 오판 예측력은 JEV가 약간 높다 |
| 판정 경향 | JEV와 LLM 모두 사람보다 관대하다. 사람이 무관으로 본 passage를 관련으로 판정하는 경우가 많다 (precision 0.39~0.62). JEV와 LLM끼리는 κ 0.63~0.75로 사람 대비보다 잘 일치한다 |
| 보정(calibration) | 한국어가 영어보다 나쁘다 (ECE 0.22 vs 0.13). v1 기준 JEV p가 0.9 이상이어도 실제 관련 비율은 한국어 62%, 영어 77% |
| Confidence 신호 | 쓸 만하다. confidence가 낮을수록 오판이 많다 (AUROC 0.67~0.81, LLM보다 약간 높음). 다만 confidence ≥ 0.85에서도 정확도가 84.5%다 |
| 반복 안정성 | **JEV 응답은 결정적이지 않다.** 같은 요청 5회 중 값이 모두 같은 unit은 12~16%다. 변동 중앙값은 0.02로 작지만 최대 0.83까지 바뀐다. 데이터셋 단위 F1은 ±0.004 안에서 안정적이다 |
| 속도와 비용 | 요청당 p50 0.27초. LLM은 `gpt-6-luna` 4~6초, `gpt-6-sol` 6~8초, `gpt-5.6-sol` 10~13초. 한국어 213개 질문 전체가 v1 $0.034, v2 $0.040 (criteria 추가로 입력 토큰 18% 증가) |
| 문항 개선 | `chunk_relevance.v2`(MIRACL 기준에 맞춘 문항)가 v1보다 한국어 전 지표에서 낫다. **기본값을 v2로 바꿨다** |

**판단**: JEV는 chunk relevance에서 가장 좋은 LLM Judge(`gpt-6-sol`)보다 **0/1 판정 일치도가 약간 낮지만(F1 −0.03), 순위 판별력은 같고 uncertainty 신호는 약간 낫고, 25배 이상 빠르다.** `gpt-6-luna`와는 판정 일치도가 같다. 다만 두 Judge 모두 **0.5 threshold로 사람 라벨을 그대로 재현하는 수준은 아니다.** Exit criteria의 F1 목표치가 아직 정해지지 않았으므로 통과 여부는 판단하지 않았다. 7장에 목표치 후보를 제안했고, 그 기준으로는 충족한다.

## 2. 데이터셋

[NoMIRACL](https://huggingface.co/datasets/miracl/nomiracl) (Apache-2.0)을 [benchmark/prepare_miracl.py](../../benchmark/prepare_miracl.py)로 변환했다. 라벨은 MIRACL 원어민 annotator가 질문–passage 쌍마다 판정한 것이다.

| 데이터셋 | 질문 수 | 판정된 passage | 관련 비율 | 설명 |
|---|---|---|---|---|
| `miracl_ko_dev_relevant` | 213 (전체) | 3,057 | 17.9% | MIRACL 한국어 dev 전체 |
| `miracl_en_dev_relevant` | 213 (무작위 추출) | 2,210 | 27.1% | 한/영 비교용 |
| `miracl_ko_dev_non_relevant` | 100 (무작위 추출) | 1,000 | 0% | 모든 passage가 무관한 질문 |

주의:

- MIRACL passage에는 검색 순위가 없어 seed를 고정해 무작위로 섞었다. **rank-aware precision은 이 데이터에서 의미가 없어 분석하지 않았다.**
- 모두 공개 Wikipedia 데이터이므로 국외 API 전송에 제약이 없다. PII 마스킹은 기본값(켜짐) 그대로 실행했다.

## 3. 방법

- 파이프라인: `Evaluator`의 `context_precision`만 실행했다. 질문 1개당 JEV 요청 1회로 보냈다 (state = question + contexts, passage마다 Noul 1개).
- JEV: 캐시 없이 5회 반복했다. 판정 지표는 1회차 기준이다.
- LLM Judge: 같은 state와 문항을 한 번의 chat completion으로 보내 passage마다 `p_yes`를 받았다. 1회 실행.
- 판정 threshold 0.5. confidence 구간은 계획서의 routing 기준(0.85 / 0.60)을 따랐다.
- 문항 A/B: v1 전체 실행 후 v2를 JEV로 1회 실행했다.

| 문항 | instructions | criteria |
|---|---|---|
| `chunk_relevance.v1` | Is `contexts[i]` useful for answering `question`? | 없음 |
| `chunk_relevance.v2` | Does `contexts[i]` contain information that answers `question`? | true: 질문에 답하는 사실을 (일부라도) 담고 있음 / false: 주제나 개체만 같고 답은 없음 |

## 4. 결과: 문항 v1 vs v2 (JEV)

| 데이터셋 | 문항 | F1 | κ | AUROC | ECE | conf ≥ 0.85 비율 / 정확도 | Pearson(soft) | MAE(soft) | 무관 chunk를 관련으로 판정 |
|---|---|---|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | v1 | 0.536 | 0.383 | 0.872 | 0.252 | 51.3% / 82.1% | 0.568 | 0.259 | 30.0% |
| miracl_ko_dev_relevant | v2 | 0.571 | 0.435 | 0.883 | 0.220 | 62.8% / 84.5% | 0.627 | 0.227 | 25.5% |
| miracl_en_dev_relevant | v1 | 0.638 | 0.464 | 0.855 | 0.167 | 54.1% / 85.7% | 0.513 | 0.200 | 26.1% |
| miracl_en_dev_relevant | v2 | 0.642 | 0.477 | 0.854 | 0.132 | 63.1% / 86.2% | 0.533 | 0.178 | 23.0% |
| miracl_ko_dev_non_relevant | v1 | – | – | – | – | 62.9% / 93.8% | – | 0.223 | 15.1% |
| miracl_ko_dev_non_relevant | v2 | – | – | – | – | 71.5% / 95.1% | – | 0.189 | 12.7% |

v2는 한국어에서 F1 +0.035, κ +0.052, ECE −0.032 개선됐다. 무관 passage를 관련으로 판정하는 비율도 줄었다. 영어는 개선 폭이 작다. v2는 criteria가 붙어 JEV 입력 토큰이 약 18% 늘었다.

### 4.1 JEV vs `gpt-6-luna` (둘 다 v2 문항)

JEV는 4장의 v2 1회 실행 결과를 그대로 재사용했다(`--reuse-jev`). 재요청하면 JEV 값이 바뀌므로 같은 기준으로 비교하기 위해서다.

| 데이터셋 | Judge | Accuracy | Precision | Recall | F1 | κ | AUROC | ECE | uncertainty AUROC | Pearson(soft) | MAE(soft) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | JEV | 0.767 | 0.425 | 0.867 | 0.571 | 0.435 | **0.883** | 0.220 | **0.667** | **0.627** | 0.227 |
| miracl_ko_dev_relevant | gpt-6-luna | **0.805** | **0.471** | 0.724 | 0.571 | **0.452** | 0.802 | **0.176** | 0.615 | 0.570 | **0.133** |
| miracl_en_dev_relevant | JEV | 0.768 | 0.553 | **0.765** | 0.642 | 0.477 | **0.854** | **0.132** | **0.722** | **0.533** | 0.178 |
| miracl_en_dev_relevant | gpt-6-luna | **0.802** | **0.621** | 0.698 | **0.657** | **0.519** | 0.804 | 0.171 | 0.596 | 0.507 | **0.148** |
| miracl_ko_dev_non_relevant | JEV | 0.873 | – | – | – | – | – | – | **0.814** | – | 0.189 |
| miracl_ko_dev_non_relevant | gpt-6-luna | **0.895** | – | – | – | – | – | – | 0.789 | – | **0.108** |

| 데이터셋 | JEV–LLM 판정 일치율 | κ | Pearson(p) | LLM p50 / p95 지연(s) |
|---|---|---|---|---|
| miracl_ko_dev_relevant | 83.7% | 0.628 | 0.720 | 6.29 / 14.63 |
| miracl_en_dev_relevant | 87.2% | 0.716 | 0.803 | 5.07 / 11.85 |
| miracl_ko_dev_non_relevant | 92.4% | 0.630 | 0.720 | 4.39 / 9.79 |

읽는 법:

- **0.5 기준 판정**은 거의 같다. `gpt-6-luna`는 더 보수적이어서(precision↑ recall↓) accuracy가 조금 높다. F1은 한국어 동률, 영어는 LLM이 +0.015.
- **순위 판별력(AUROC)과 uncertainty 신호**는 JEV가 분명히 낫다. confidence 기반 routing과 threshold 보정을 설계하는 데 JEV 쪽이 더 유리하다.
- **절대값 오차(MAE)**는 LLM이 작다. JEV p가 관련 쪽으로 치우쳐 있어서다(6.2절).
- 참고: `gpt-6-luna` + v2(한국어 F1 0.571)가 `gpt-5.6-sol` + v1(0.615)보다 낮다. 모델과 문항이 함께 바뀌어 원인을 나눌 수 없다. 기준선을 확정하려면 한 가지씩 바꿔 비교해야 한다.

`gpt-6-luna` 사용 시 주의 사항 (구현에 반영함):

- `temperature=0`을 보내면 500 오류가 난다. GPT-5 이후 모델은 temperature를 지원하지 않으므로 [llm_judge.py](../../src/ragas_jev/audit/llm_judge.py)에서 temperature를 아예 보내지 않도록 고쳤다.
- 요청한 `json_schema` 대신 `{"chunk_0": {"p_yes": ...}}` 형식으로 답한다(첫 실행에서 212/526 요청 실패). 파서가 이 형식과, JSON 뒤에 텍스트가 붙는 경우도 받도록 고쳤다.

### 4.2 JEV vs `gpt-6-luna` vs `gpt-6-sol` (모두 v2 문항)

조건은 4.1과 같다. JEV 결과를 재사용했고, LLM 모델만 바꿨다.

| 데이터셋 | Judge | Accuracy | Precision | Recall | F1 | κ | AUROC | ECE | uncertainty AUROC | Pearson(soft) | MAE(soft) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | JEV | 0.767 | 0.425 | 0.867 | 0.571 | 0.435 | **0.883** | 0.220 | **0.667** | **0.627** | 0.227 |
| miracl_ko_dev_relevant | gpt-6-luna | 0.805 | 0.471 | 0.724 | 0.571 | 0.452 | 0.802 | **0.176** | 0.615 | 0.570 | **0.133** |
| miracl_ko_dev_relevant | gpt-6-sol | **0.806** | **0.476** | 0.834 | **0.606** | **0.490** | 0.876 | 0.180 | 0.659 | 0.599 | 0.158 |
| miracl_en_dev_relevant | JEV | 0.768 | 0.553 | **0.765** | 0.642 | 0.477 | **0.854** | **0.132** | **0.722** | **0.533** | 0.178 |
| miracl_en_dev_relevant | gpt-6-luna | **0.802** | **0.621** | 0.698 | 0.657 | 0.519 | 0.804 | 0.171 | 0.596 | 0.507 | **0.148** |
| miracl_en_dev_relevant | gpt-6-sol | 0.798 | 0.601 | 0.757 | **0.670** | **0.527** | 0.852 | 0.177 | 0.691 | 0.512 | 0.157 |
| miracl_ko_dev_non_relevant | JEV | 0.873 | – | – | – | – | – | – | **0.814** | – | 0.189 |
| miracl_ko_dev_non_relevant | gpt-6-luna | **0.895** | – | – | – | – | – | – | 0.789 | – | **0.108** |
| miracl_ko_dev_non_relevant | gpt-6-sol | 0.862 | – | – | – | – | – | – | 0.776 | – | 0.141 |

| 데이터셋 | JEV–gpt-6-sol 판정 일치율 | κ | Pearson(p) | gpt-6-sol p50 / p95 지연(s) | gpt-6-sol 최적 threshold (F1) |
|---|---|---|---|---|---|
| miracl_ko_dev_relevant | 88.7% | 0.748 | 0.819 | 7.85 / 14.28 | 0.95 (F1 0.648) |
| miracl_en_dev_relevant | 89.3% | 0.767 | 0.844 | 6.38 / 11.38 | 0.90 (F1 0.678) |
| miracl_ko_dev_non_relevant | 91.9% | 0.648 | 0.750 | 6.09 / 10.37 | – |

읽는 법:

- **`gpt-6-sol`이 세 Judge 중 사람 라벨과 가장 잘 맞는다.** JEV 대비 F1은 한국어 +0.035, 영어 +0.028, κ는 +0.055 / +0.050이다.
- **AUROC는 JEV와 `gpt-6-sol`이 비슷하다** (한국어 0.883 vs 0.876, 영어 0.854 vs 0.852). `gpt-6-luna`는 두 모델보다 0.05~0.08 낮다.
- **오판 예측력(uncertainty AUROC)은 JEV가 조금씩 높다** (한국어 0.667 vs 0.659, 영어 0.722 vs 0.691, 무관 세트 0.814 vs 0.776).
- **JEV와 `gpt-6-sol`은 서로 잘 일치한다** (판정 일치율 89%, κ 0.75~0.77). 두 모델이 사람과 어긋나는 사례도 상당 부분 겹친다는 뜻이다.
- `gpt-6-sol`도 JEV처럼 관련 쪽으로 치우쳐 있어 최적 threshold가 0.90~0.95다.
- 지연 시간: `gpt-6-sol` p50 6~8초. JEV(0.27초)보다 약 25~30배 느리다.
- `gpt-6-sol`도 `temperature`를 거부했고(500), 응답을 id 키 객체 형식으로 반환했다. 파서가 처리해 오류는 0건이었다.

### 4.3 Auditor 역할 모의 실험: JEV가 불확실한 판정만 LLM으로 재검증

Phase 5 Auditor는 JEV confidence가 0.85 미만인 판정만 다시 본다. 그래서 전체 성능보다 **JEV가 불확실한 구간에서의 정확도**가 중요하다. 4.2의 저장 결과로 계산했다 ([analyze_routing.py](../../benchmark/analyze_routing.py), 추가 API 호출 없음).

| 데이터셋 | JEV conf 구간 | unit 수 | JEV | gpt-6-luna | gpt-6-sol |
|---|---|---|---|---|---|
| miracl_ko_dev_relevant | ≥ 0.85 | 1919 | 84.5% | 84.1% | 85.7% |
| miracl_ko_dev_relevant | 0.6–0.85 | 904 | 66.0% | 74.6% | 73.1% |
| miracl_ko_dev_relevant | < 0.6 | 234 | 53.4% | 74.4% | 67.9% |
| miracl_en_dev_relevant | ≥ 0.85 | 1394 | 86.2% | 85.9% | 86.5% |
| miracl_en_dev_relevant | 0.6–0.85 | 636 | 64.0% | 70.4% | 69.2% |
| miracl_en_dev_relevant | < 0.6 | 180 | 49.4% | 70.6% | 65.0% |
| miracl_ko_dev_non_relevant | ≥ 0.85 | 715 | 95.1% | 94.5% | 94.0% |
| miracl_ko_dev_non_relevant | 0.6–0.85 | 226 | 73.9% | 77.0% | 67.7% |
| miracl_ko_dev_non_relevant | < 0.6 | 59 | 44.1% | 76.3% | 62.7% |

| 데이터셋 | 구성 (JEV conf < 0.85만 LLM으로 대체) | Accuracy | F1 | κ |
|---|---|---|---|---|
| miracl_ko_dev_relevant | JEV only | 0.767 | 0.571 | 0.435 |
| miracl_ko_dev_relevant | JEV + gpt-6-luna | 0.808 | 0.600 | 0.485 |
| miracl_ko_dev_relevant | JEV + gpt-6-sol | 0.799 | 0.601 | 0.482 |
| miracl_ko_dev_relevant | gpt-6-luna only | 0.805 | 0.571 | 0.452 |
| miracl_ko_dev_relevant | gpt-6-sol only | 0.806 | 0.606 | 0.490 |
| miracl_ko_dev_relevant | (재검증 비율 37.2%) | | | |
| miracl_en_dev_relevant | JEV only | 0.768 | 0.642 | 0.477 |
| miracl_en_dev_relevant | JEV + gpt-6-luna | 0.804 | 0.668 | 0.530 |
| miracl_en_dev_relevant | JEV + gpt-6-sol | 0.796 | 0.669 | 0.525 |
| miracl_en_dev_relevant | gpt-6-luna only | 0.802 | 0.657 | 0.519 |
| miracl_en_dev_relevant | gpt-6-sol only | 0.798 | 0.670 | 0.527 |
| miracl_en_dev_relevant | (재검증 비율 36.9%) | | | |
| miracl_ko_dev_non_relevant | JEV only | 0.873 | – | – |
| miracl_ko_dev_non_relevant | JEV + gpt-6-luna | 0.899 | – | – |
| miracl_ko_dev_non_relevant | JEV + gpt-6-sol | 0.870 | – | – |
| miracl_ko_dev_non_relevant | gpt-6-luna only | 0.895 | – | – |
| miracl_ko_dev_non_relevant | gpt-6-sol only | 0.862 | – | – |
| miracl_ko_dev_non_relevant | (재검증 비율 28.5%) | | | |

읽는 법:

- JEV가 확신하는 구간(≥ 0.85)에서는 세 Judge의 정확도가 거의 같다. 이 구간을 재검증해도 얻을 것이 없다.
- JEV가 불확실한 구간에서는 **`gpt-6-luna`가 가장 정확하다** (한국어 0.60–0.85 구간 74.6%, 0.60 미만 74.4%). 보수적으로 "무관"이라 답하는 경향이 엄격한 사람 라벨과 맞는 것으로 보인다.
- 약 37%만 재검증하는 Hybrid 구성(JEV + LLM)은 JEV 단독보다 F1 +0.03, κ +0.05 높고, `gpt-6-sol` 단독과 거의 같다.
- 결론: **벤치마크 기준선과 Auditor는 다른 모델이 맞다.** 기준선은 가장 강한 `gpt-6-sol`로 두고, Auditor는 불확실 구간 정확도가 가장 높고 더 빠른 `gpt-6-luna`로 둔다.
- 한계: LLM은 각 1회 실행이고 결정적이지 않다. 1~2%p 차이는 재실행으로 뒤바뀔 수 있다. 모의 실험은 threshold 0.85 하나만 봤다. Phase 5에서 threshold별 비용–정확도 곡선으로 다시 확인한다.

## 5. 결과: 문항 v1 전체 (JEV 5회 반복 + `gpt-5.6-sol`)

#### Unit 단위 판정 (threshold = 0.5)

| 데이터셋 | Judge | n | Accuracy | Precision | Recall | F1 | κ | AUROC | ECE | 최적 threshold (F1) |
|---|---|---|---|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | JEV | 3057 | 0.730 | 0.387 | 0.868 | 0.536 | 0.383 | 0.872 | 0.252 | 0.80 (F1 0.622) |
| miracl_ko_dev_relevant | LLM (gpt-5.6-sol) | 3057 | 0.811 | 0.484 | 0.841 | 0.615 | 0.501 | 0.886 | 0.161 | 0.95 (F1 0.665) |
| miracl_en_dev_relevant | JEV | 2210 | 0.755 | 0.532 | 0.797 | 0.638 | 0.464 | 0.855 | 0.167 | 0.70 (F1 0.658) |
| miracl_en_dev_relevant | LLM (gpt-5.6-sol) | 2210 | 0.809 | 0.618 | 0.775 | 0.688 | 0.553 | 0.867 | 0.133 | 0.60 (F1 0.692) |
| miracl_ko_dev_non_relevant | JEV | 1000 | 0.849 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | – |
| miracl_ko_dev_non_relevant | LLM (gpt-5.6-sol) | 1000 | 0.855 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | – |

#### Confidence 구간별 JEV 정확도

| 데이터셋 | 구간 | 비율 | n | Accuracy |
|---|---|---|---|---|
| miracl_ko_dev_relevant | conf ≥ 0.85 | 51.3% | 1567 | 82.1% |
| miracl_ko_dev_relevant | 0.60 ≤ conf < 0.85 | 37.5% | 1146 | 68.8% |
| miracl_ko_dev_relevant | conf < 0.60 | 11.3% | 344 | 45.9% |
| miracl_en_dev_relevant | conf ≥ 0.85 | 54.1% | 1195 | 85.7% |
| miracl_en_dev_relevant | 0.60 ≤ conf < 0.85 | 36.1% | 798 | 67.2% |
| miracl_en_dev_relevant | conf < 0.60 | 9.8% | 217 | 49.8% |
| miracl_ko_dev_non_relevant | conf ≥ 0.85 | 62.9% | 629 | 93.8% |
| miracl_ko_dev_non_relevant | 0.60 ≤ conf < 0.85 | 29.1% | 291 | 74.2% |
| miracl_ko_dev_non_relevant | conf < 0.60 | 8.0% | 80 | 53.8% |

#### Uncertainty 품질 (uncertainty로 오판을 예측하는 AUROC)

| 데이터셋 | JEV | LLM |
|---|---|---|
| miracl_ko_dev_relevant | 0.666 | 0.652 |
| miracl_en_dev_relevant | 0.703 | 0.690 |
| miracl_ko_dev_non_relevant | 0.770 | 0.720 |

#### Sample 단위 Context Precision

| 데이터셋 | Judge | 사람 precision 평균 | soft 평균 | binary 평균 | Pearson(soft) | Spearman(soft) | MAE(soft) | 관련 chunk ≥1개 판정 비율 |
|---|---|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | JEV | 0.180 | 0.436 | 0.408 | 0.568 | 0.581 | 0.259 | 99.5% |
| miracl_ko_dev_relevant | LLM | 0.180 | 0.331 | 0.317 | 0.578 | 0.547 | 0.165 | 99.5% |
| miracl_en_dev_relevant | JEV | 0.271 | 0.439 | 0.407 | 0.513 | 0.510 | 0.200 | 95.8% |
| miracl_en_dev_relevant | LLM | 0.271 | 0.360 | 0.343 | 0.493 | 0.492 | 0.157 | 95.8% |
| miracl_ko_dev_non_relevant | JEV | 0.000 | 0.223 | 0.151 | – | – | 0.223 | 61.0% |
| miracl_ko_dev_non_relevant | LLM | 0.000 | 0.168 | 0.145 | – | – | 0.168 | 69.0% |

#### 반복 실행 안정성 (JEV, 캐시 없음)

| 데이터셋 | 실행 수 | unit 수 | 모든 실행에서 값 동일 | 0.5 기준 판정 뒤집힘 | 평균 Δp | 최대 Δp |
|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | 5 | 3057 | 12.0% | 4.9% | 0.040 | 0.680 |
| miracl_en_dev_relevant | 5 | 2210 | 14.5% | 3.7% | 0.034 | 0.830 |
| miracl_ko_dev_non_relevant | 5 | 1000 | 16.4% | 2.2% | 0.026 | 0.330 |

#### 지연 시간과 비용 (1회 실행 기준)

| 데이터셋 | Judge | 요청 수 | 입력 토큰 | p50 지연(s) | p95 지연(s) | JEV 비용(USD) | 오류 |
|---|---|---|---|---|---|---|---|
| miracl_ko_dev_relevant | JEV | 213 | 805,526 | 0.27 | 0.35 | 0.0338 | 0 |
| miracl_ko_dev_relevant | LLM | 213 | 689,288 | 13.42 | 18.64 | – | 0 |
| miracl_en_dev_relevant | JEV | 213 | 497,163 | 0.28 | 0.37 | 0.0209 | 0 |
| miracl_en_dev_relevant | LLM | 213 | 460,546 | 10.41 | 15.51 | – | 0 |
| miracl_ko_dev_non_relevant | JEV | 100 | 281,917 | 0.27 | 0.58 | 0.0118 | 0 |
| miracl_ko_dev_non_relevant | LLM | 100 | 237,077 | 10.77 | 15.43 | – | 0 |

#### JEV vs LLM Judge

| 데이터셋 | unit 수 | 판정 일치율 | κ | Pearson(p) |
|---|---|---|---|---|
| miracl_ko_dev_relevant | 3057 | 86.9% | 0.718 | 0.836 |
| miracl_en_dev_relevant | 2210 | 88.2% | 0.750 | 0.879 |
| miracl_ko_dev_non_relevant | 1000 | 93.2% | 0.730 | 0.827 |

#### Reliability (JEV p 구간별 실제 관련 비율)

**miracl_ko_dev_relevant**

| p 구간 | n | 평균 p | 실제 관련 비율 |
|---|---|---|---|
| 0.0-0.1 | 620 | 0.055 | 0.018 |
| 0.1-0.2 | 505 | 0.142 | 0.026 |
| 0.2-0.3 | 333 | 0.243 | 0.048 |
| 0.3-0.4 | 202 | 0.338 | 0.074 |
| 0.4-0.5 | 170 | 0.442 | 0.100 |
| 0.5-0.6 | 190 | 0.544 | 0.105 |
| 0.6-0.7 | 156 | 0.644 | 0.192 |
| 0.7-0.8 | 151 | 0.749 | 0.185 |
| 0.8-0.9 | 231 | 0.848 | 0.381 |
| 0.9-1.0 | 499 | 0.951 | 0.619 |

**miracl_en_dev_relevant**

| p 구간 | n | 평균 p | 실제 관련 비율 |
|---|---|---|---|
| 0.0-0.1 | 472 | 0.052 | 0.044 |
| 0.1-0.2 | 337 | 0.139 | 0.047 |
| 0.2-0.3 | 220 | 0.243 | 0.136 |
| 0.3-0.4 | 167 | 0.344 | 0.186 |
| 0.4-0.5 | 116 | 0.446 | 0.207 |
| 0.5-0.6 | 112 | 0.548 | 0.214 |
| 0.6-0.7 | 119 | 0.645 | 0.311 |
| 0.7-0.8 | 101 | 0.742 | 0.366 |
| 0.8-0.9 | 155 | 0.849 | 0.406 |
| 0.9-1.0 | 411 | 0.949 | 0.771 |

## 6. 해석

### 6.1 판정이 사람보다 관대한 이유

- JEV와 두 LLM Judge 모두 recall은 0.70~0.87로 높고 precision은 0.39~0.62로 낮다. 사람이 무관으로 본 passage를 관련으로 판정하는 경향이다.
- JEV와 LLM Judge끼리의 일치도(κ 0.63~0.75)가 각각의 사람 대비 일치도(κ 0.38~0.55)보다 높다. 따라서 차이의 상당 부분은 JEV 고유의 약점이 아니라 **"관련"의 정의 차이**로 보인다. MIRACL annotator는 질문에 실제로 답하는 passage만 관련으로 표시한다.
- MIRACL 기준에 맞춘 v2 문항으로 한국어 격차가 일부 줄었다. 다음 문항 개선 전에 불일치 사례를 직접 검수해 라벨 자체의 엄격함인지, JEV 오판인지 나눠 볼 필요가 있다.

### 6.2 순위 판별력은 좋고 threshold가 맞지 않는다

- AUROC 0.85~0.88은 관련 passage를 무관한 passage보다 높게 매기는 능력이 좋다는 뜻이다.
- F1이 최대가 되는 threshold는 0.75~0.85다(v2 한국어 0.85). JEV p가 관련 쪽으로 치우쳐 있어 **0.5 threshold가 너무 낮다.**
- reliability 표(v1)를 보면 한국어는 p 0.5~0.6 구간의 실제 관련 비율이 10.5%에 불과하다. 한국어 보정이 영어보다 나쁜 것은 TypeSafe 문서에 적힌 "영어 외 언어는 정확도가 같지 않다"는 설명과 맞다.
- 따라서 Context Precision의 절대값(soft 평균 0.40)을 사람 기준 precision(0.18)과 같은 척도로 읽으면 안 된다. 샘플 간 상대 비교(Pearson 0.63)로 쓰거나, Phase 5에서 언어별로 보정해야 한다.

### 6.3 Confidence routing에 주는 시사점

- confidence가 낮을수록 정확도가 떨어진다 (v2 한국어): 0.85 이상 84.5%, 0.60~0.85 66%, 0.60 미만 53%. 방향은 맞으므로 routing 신호로 쓸 수 있다.
- 하지만 현재 threshold(0.85 / 0.60)로는 한국어 unit의 37%가 재검증 대상이 되고, 자동 수용 구간의 정확도도 85%에 못 미친다. Phase 5 calibration에서 언어별 threshold와 p 보정을 함께 정해야 한다.
- 모든 passage가 무관한 질문에서도 v2 기준 53%가 "관련 chunk 1개 이상"으로 판정됐다. "관련 context가 없음"을 감지하는 용도로는 현재 threshold가 약하다.

### 6.4 반복 안정성

- 같은 모델 버전(`jev-1.13.0`)에 같은 요청을 보내도 응답값이 바뀐다. unit 단위 변동은 중앙값 0.02, 상위 1% 0.27~0.31, 최대 0.83이다. 0.5 기준 판정이 뒤집힌 unit은 2~5%다.
- 데이터셋 단위 지표는 안정적이다 (5회 F1 범위 한국어 0.536~0.544). 5회 평균을 내도 F1은 +0.01, AUROC는 +0.001로 거의 나아지지 않는다.
- 결론: **점수의 비트 단위 재현은 캐시로만 보장된다.** 계획서의 "동일 입력·동일 캐시 상태에서 재현" 기준은 캐시(`CachedJudge`)로 충족되지만, 캐시 없는 재실행은 unit 단위 판정이 달라질 수 있다. 이 현상은 TypeSafe에 문의할 가치가 있다. 문서에는 "extremely consistent"라고 적혀 있다.

### 6.5 속도와 비용

- JEV는 요청당 0.27초(p50), LLM Judge는 `gpt-6-luna` 4~6초, `gpt-5.6-sol` 10~13초다. 한국어 213개 질문에 JEV는 v1 $0.034, v2 $0.040가 들었다. LLM 비용은 OAuth 프록시라 측정하지 않았다.
- 한 요청에 passage 9~20개를 묶는 fan-out 설계가 잘 동작했다. 오류 0건.

## 7. 결정 사항과 제안

**적용한 결정**

1. `chunk_relevance.v2`를 기본 문항으로 바꿨다 ([pipeline.py](../../src/ragas_jev/pipeline.py)).
2. LLM 모델을 역할별로 나눴다 ([config.py](../../src/ragas_jev/config.py), 4.3절 근거).
   - 벤치마크 기준선 `RAGAS_JEV_BASELINE_JUDGE_MODEL=gpt-6-sol`: JEV 성능을 판정하는 비교 대상. `run_phase1.py --llm-judge`의 기본 모델
   - Auditor `RAGAS_JEV_AUDITOR_MODEL=gpt-6-luna`: Phase 5에서 JEV 불확실 판정을 재검증
3. LLM 호출에서 `temperature`를 보내지 않는다 (GPT-5 이후 모델은 지원하지 않음).

**제안 (확인 필요)**

1. **Exit criteria 목표치**: 사람 라벨을 그대로 기준으로 삼으면 LLM Judge도 F1 0.62~0.69에 그친다. "한국어 F1 ≥ LLM Judge − 0.05" 또는 "AUROC ≥ 0.85"처럼 **LLM Judge 대비 상대 기준**으로 정하는 것을 제안한다. 가장 강한 기준선 `gpt-6-sol`(v2) 대비 JEV F1은 한국어 −0.035, 영어 −0.028로 첫 기준(−0.05 이내)을 충족하고, AUROC(한국어 0.883, 영어 0.854)도 충족한다. `gpt-6-luna` 기준으로는 더 여유 있게 충족한다. 기준선은 `gpt-6-sol`로 두는 것을 제안한다.
2. **불일치 검수**: JEV와 사람 라벨이 어긋난 한국어 사례 30~50개를 직접 보고 라벨 엄격함과 JEV 오판을 구분한다. v3 문항 설계의 근거가 된다.
3. **Phase 5 입력**: 언어별 threshold와 p 보정(예: isotonic)을 calibration 항목에 추가한다. 이 리포트의 reliability 표와 4.3절 모의 실험이 기초 자료다.
4. **TypeSafe 문의**: 동일 요청의 응답 변동(최대 0.83)이 정상 범위인지 확인한다.

## 8. 재현 방법

```bash
uv run python benchmark/prepare_miracl.py --lang ko
uv run python benchmark/prepare_miracl.py --lang en --max-queries 213
uv run python benchmark/prepare_miracl.py --lang ko --subset non_relevant --max-queries 100
uv run python benchmark/run_phase1.py --repeats 5 --llm-judge --chunk-question chunk_relevance.v1
uv run python benchmark/run_phase1.py --repeats 1 --chunk-question chunk_relevance.v2 --summary .cache/phase1/summary_v2.json
# LLM Judge만 새로 실행 (저장된 JEV v2 결과 재사용, 모델은 RAGAS_JEV_AUDITOR_MODEL)
uv run python benchmark/run_phase1.py --repeats 1 --chunk-question chunk_relevance.v2 --reuse-jev --llm-judge --summary .cache/phase1/summary_v2_llm.json
uv run python benchmark/render_phase1.py .cache/phase1/summary.json
# 기준선 모델 비교 (RAGAS_JEV_BASELINE_JUDGE_MODEL 또는 --llm-model)
uv run python benchmark/run_phase1.py --repeats 1 --chunk-question chunk_relevance.v2 --reuse-jev --llm-judge --llm-model gpt-6-sol --summary .cache/phase1/summary_v2_gpt6sol.json
# Auditor 모의 실험
uv run python benchmark/analyze_routing.py --models gpt-6-luna gpt-6-sol
```

원시 결과는 `.cache/phase1/`에 저장된다 (git 제외).
