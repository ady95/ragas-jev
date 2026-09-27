# Changelog

## 0.3.0 (2026-09-27)

### 추가

- **`ragas-jev calibration adapt`**: 라벨 없이, 평가 결과의 JEV 확률만으로 보정 곡선을 그 데이터의 "예" 비율에 맞춥니다(prior shift, EM). 결과는 일반 보정 파일과 형식이 같아 `evaluate --calibration`에 그대로 씁니다. 검증: [자동 보정 검증 리포트](docs/reports/auto_calibration.md)
- **`calibration sample --prelabel`**: LLM 판정(`--prelabel-model`, 기본 `RAGAS_JEV_STRONG_JUDGE_MODEL`)을 라벨 시트의 `llm_p`, `llm_label` 열에 참고용으로 채웁니다. `label`은 비워 두며, `fit`은 `label`만 읽습니다
- **보정 파일의 `priors`**: 곡선을 학습한 데이터의 "예" 비율을 문항 × 언어별로 저장합니다. `calibration fit`이 기록하고, 패키지 기본 보정 파일에도 추가했습니다(곡선은 그대로)
- **비율 차이 안내**: `evaluate`에서 보정을 켜면, 추정 비율이 보정 데이터와 0.15 이상 다른 문항을 알려 줍니다

### 변경

- 라벨 시트에 `llm_p`, `llm_label` 열이 추가됐습니다. 기존 시트도 그대로 읽힙니다

## 0.2.0 (2026-09-26)

### 동작이 바뀐 것 (업그레이드 전 확인)

- **PII 마스킹이 기본으로 꺼진다.** 원문이 그대로 LLM 엔드포인트와 JEV API로 전송된다. 개인정보가 섞일 수 있는 데이터는 `--pii-masking`을 붙이거나 `.env`에 `RAGAS_JEV_PII_MASKING=true`를 넣는다. 마스킹이 꺼진 채 외부 API를 호출하면 평가를 시작할 때 안내 문구를 한 줄 출력한다.
- **확률 보정이 기본으로 꺼진다.** JEV 원래 확률을 그대로 쓴다. 보정하려면 `--calibrate`(패키지의 기본 보정 파일) 또는 `--calibration 파일.json`을 쓰거나, `.env`에 `RAGAS_JEV_CALIBRATION=true` 또는 `RAGAS_JEV_CALIBRATION_FILE`을 넣는다. 지표별 LLM 재판정 기준은 보정된 확률로 정한 값이라, 벤치마크 수준의 Hybrid 성능을 원하면 보정을 켠다.
- **보정 파일이 없으면 오류로 멈춘다.** 0.1.0은 지정한 보정 파일이 없으면 경고 없이 보정을 건너뛰었다.
- **`init` 템플릿의 `OPENAI_BASE_URL` 기본값이 `https://api.openai.com/v1`이다.**

0.1.0으로 `ragas-jev init`을 실행한 `.env`에는 `RAGAS_JEV_PII_MASKING=true`가 적혀 있어, 업그레이드해도 마스킹이 켜진 채로 동작한다. `init`은 기존 파일을 덮어쓰지 않는다. 새 기본값을 따르려면 그 줄을 지우거나 `false`로 바꾼다.

### 추가

- `evaluate --calibrate/--no-calibrate`, `evaluate --pii-masking/--no-pii-masking`
- `calibration sample --pii-masking/--no-pii-masking`
- 설정 `RAGAS_JEV_CALIBRATION`

### 수정

- `calibration sample --pii-masking`으로 라벨 시트를 만들 때, 마스킹 없이 평가한 결과의 평가 단위 텍스트가 원문으로 들어가던 문제. 이제 같은 샘플의 다른 텍스트와 같은 자리표시자로 가린다.
- `AVAILABLE_MODELS`를 설정하지 않으면 `healthcheck`가 제공되는 모든 모델을 "목록에 없음"으로 출력하던 문제.

### 호환성

- 0.1.0의 `--no-calibration`은 `--no-calibrate`와 같은 뜻으로 계속 동작한다.

## 0.1.0 (2026-09-24)

첫 배포. Context Precision, Context Recall, Faithfulness, Answer Relevancy 평가, JEV 판정과 확률 보정, 신뢰도 기반 LLM 재판정, 사람 검토, 도메인 재보정, PII 마스킹, `ragas-jev init`과 `--env-file`.
