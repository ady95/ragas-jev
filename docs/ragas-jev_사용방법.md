# ragas-jev 사용 방법

- 대상 버전: `ragas-jev` 0.2.0 ([PyPI](https://pypi.org/project/ragas-jev/))
- 관련 문서: [README](../README.md), [평가 항목 비교](ragas-jev_평가항목.md), [도메인 재보정 가이드](도메인_재보정_가이드.md)

`ragas-jev`는 RAG 시스템의 질문·답변·검색 문서를 JSONL로 받아 네 가지 지표로 평가하는 CLI 도구입니다. 지표는 Context Precision, Context Recall, Faithfulness, Answer Relevancy입니다. 아래 순서대로 쓰시면 됩니다.

## 1. 설치와 키 설정

```bash
pip install ragas-jev        # Python 3.10 이상
ragas-jev init               # 현재 폴더에 .env 템플릿 생성
```

`.env`에 두 가지 키를 넣습니다.

```
TYPESAFE_API_KEY=...        # JEV(TypeSafe) API 키
OPENAI_API_KEY=...          # claim 추출·재판정용 LLM 키
OPENAI_BASE_URL=https://api.openai.com/v1   # 기본값. 다른 OpenAI 호환 서버를 쓰면 그 주소로 변경
```

- **어느 폴더에서나 같은 설정을 쓰려면**: `ragas-jev init --user`로 사용자 설정 파일을 만듭니다. 위치는 Windows `%APPDATA%\ragas-jev\.env`, macOS·Linux `~/.config/ragas-jev/.env`입니다.
- **특정 설정 파일을 지정하려면**: `ragas-jev --env-file ./prod.env ...`처럼 씁니다.
- **환경변수로 넣어도 됩니다.** 설정 파일보다 환경변수가 우선합니다.

설정 파일은 아래 중 처음 발견한 하나를 읽습니다.

1. `--env-file` 옵션 또는 `RAGAS_JEV_ENV_FILE` 환경변수로 지정한 파일
2. 현재 폴더의 `.env`
3. 사용자 설정 파일

설정이 끝나면 연결을 확인합니다. 테스트용 합성 문장만 보냅니다.

```bash
ragas-jev healthcheck
```

## 2. 입력 파일 만들기

한 줄에 샘플 하나를 JSON으로 적은 JSONL 파일입니다.

```json
{"sample_id":"q-1","question":"대한민국의 수도는?","answer":"서울입니다.","contexts":["서울은 대한민국의 수도이다.","부산은 항구 도시다."],"reference":"서울"}
```

| 필드 | 필수 | 설명 |
|---|:---:|---|
| `sample_id` | ✓ | 고유 ID. 이어서 실행할 때 중복 확인에 씀 |
| `question` | ✓ | 사용자 질문 |
| `answer` | ✓ | RAG 시스템의 답변 |
| `contexts` | ✓ | 검색된 문서. **검색 순위 순서대로** |
| `reference` | | 정답 답변. 있으면 Context Recall을 계산하고 Context Precision도 더 정확해짐 |
| `metadata` | | 추가 정보를 담는 객체 |

## 3. 평가 실행

```bash
# 키 없이 동작만 확인 (샘플 데이터 + mock 판정)
ragas-jev init --sample
ragas-jev evaluate -i sample_ko.jsonl -o results/mock.jsonl --judge mock --extractor sentence --no-audit

# 실제 평가
ragas-jev evaluate -i data.jsonl -o results/out.jsonl
```

mock 판정의 점수는 동작 확인용입니다. 실제 품질 평가에는 쓰지 마세요.

자주 쓰는 옵션:

| 옵션 | 뜻 |
|---|---|
| `--metrics context_precision,faithfulness` | 원하는 지표만 계산. 기본값은 전체 |
| `--no-audit` | LLM 재판정 없이 JEV 판정만 사용. 더 빠르고 저렴함 |
| `--limit 20` | 앞의 20개 샘플만 평가. 시험 실행용 |
| `--concurrency 8` | 동시에 처리할 샘플 수 (기본 4) |
| `--calibrate` | 패키지의 기본 보정 파일로 JEV 확률을 보정. 기본값은 꺼짐 |
| `--calibration my.json` | 지정한 보정 파일(도메인 재보정 결과 등)로 보정. `--calibrate`를 함께 쓸 필요 없음 |
| `--no-resume` | 출력 파일에 이미 있는 샘플도 다시 평가 |
| `--pii-masking` | 외부로 보내기 전에 개인정보를 마스킹. 기본값은 꺼짐 |

전체 옵션은 `ragas-jev evaluate --help`로 확인합니다.

- **중단 후 재개**: 결과는 샘플마다 한 줄씩 바로 저장됩니다. 중간에 멈춰도 같은 명령을 다시 실행하면 남은 샘플만 평가합니다.
- **캐시**: 추출과 판정 결과를 `.cache/ragas_jev/`에 저장합니다. 같은 입력을 다시 평가하면 빠르고 점수도 똑같이 나옵니다.
- **설정별 비교**: 설정을 바꿔 결과를 비교할 때는 **새 출력 파일**을 지정하세요.

## 4. 결과 읽기

실행이 끝나면 지표별 평균이 화면에 출력됩니다. 결과 파일에는 샘플마다 다음 정보가 저장됩니다.

| 위치 | 내용 |
|---|---|
| `retrieval` | `context_precision`, `context_recall` (0~1, 높을수록 좋음) |
| `generation` | `faithfulness`, `answer_relevancy` |
| `confidence`, `uncertainty` | 지표별 판정 신뢰도 |
| `units` | claim·chunk 단위 판정, 확률, 판정 출처 (JEV, LLM, 사람) |
| `status`, `reasons` | `ok` 또는 `partial`. 계산하지 못한 지표는 그 이유 (예: `no_reference`) |

점수가 낮으면 `units`를 보세요. 어느 claim이 근거 없이 나왔는지, 어느 chunk가 관련 없었는지 바로 확인할 수 있습니다.

## 5. 사람 검토 (선택)

JEV와 LLM의 판정이 엇갈린 부분만 CSV로 뽑아 사람이 확인하고, 그 결과로 점수를 다시 계산합니다.

```bash
ragas-jev review export -i results/out.jsonl -o review.csv
# review.csv의 label 열에 1(예) / 0(아니오) 입력
ragas-jev review import -i results/out.jsonl -l review.csv -o results/reviewed.jsonl
```

## 6. 도메인 재보정 (실서비스 전에 권장)

기본 보정은 Wikipedia 계열 공개 데이터로 학습했습니다. 서비스 도메인에 쓰려면 그 도메인의 라벨로 다시 학습하는 것이 좋습니다.

```bash
ragas-jev calibration sample -r results/out.jsonl -s data.jsonl -o label_sheet.csv
# label 열 작성 (질문 유형 × 언어별로 최소 100개)
ragas-jev calibration fit -l label_sheet.csv -o domain_calibration.json
ragas-jev evaluate -i data.jsonl -o results/recal.jsonl --calibration domain_calibration.json
```

라벨링과 검증 절차는 [도메인 재보정 가이드](도메인_재보정_가이드.md)를 참고하세요.

## 7. Python 코드에서 쓰기

지금은 CLI가 주 사용 방법입니다. 코드에서 쓰려면 구성 요소를 직접 조립해야 합니다. 아래는 LLM 재판정 없이 JEV 판정만 쓰는 예시입니다.

```python
import asyncio
from ragas_jev.config import get_settings
from ragas_jev.judge.jev_client import JevJudge
from ragas_jev.preprocess.llm_extractor import LlmExtractor
from ragas_jev.pipeline import Evaluator
from ragas_jev.scoring.calibration import Calibrator
from ragas_jev.schemas import RagSample

async def main():
    s = get_settings()
    judge, extractor = JevJudge.from_settings(s), LlmExtractor.from_settings(s)
    evaluator = Evaluator.from_settings(s, judge, extractor)
    evaluator.calibrator = Calibrator.load(s.calibration_path)
    try:
        sample = RagSample(sample_id="q-1", question="대한민국의 수도는?", answer="서울입니다.",
                           contexts=["서울은 대한민국의 수도이다."], reference="서울")
        [result] = await evaluator.evaluate([sample])
        print(result.retrieval, result.generation)
    finally:
        await judge.aclose()
        await extractor.aclose()

asyncio.run(main())
```

## 8. 주의할 점

- **데이터가 외부로 전송됩니다.** 평가할 텍스트는 LLM 엔드포인트와 JEV API로 보내집니다. PII 마스킹은 **기본으로 꺼져 있어** 원문이 그대로 전송됩니다. 개인정보가 섞일 수 있으면 `--pii-masking`을 붙이거나 `.env`에 `RAGAS_JEV_PII_MASKING=true`를 넣으세요. 켜면 이메일, 전화번호, 카드·계좌번호, 주민·사업자등록번호를 가려 보냅니다. 규칙 기반이라 이름이나 주소는 찾지 못하므로, 민감한 데이터는 조직의 보안 검토를 먼저 받으세요.
- **JEV 모델 버전을 바꾸지 마세요.** 기본 보정 파일은 `jev-1.13.0`에 맞춰 학습했습니다. `jev-latest`로 바꾸면 보정이 맞지 않습니다.
