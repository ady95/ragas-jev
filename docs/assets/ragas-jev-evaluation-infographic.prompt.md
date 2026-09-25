# RAGAS-JEV 평가 항목 인포그래픽 생성 프롬프트

## 2026-09-26 수정: 벤치마크 제외

내장 image_gen 도구로 기존 이미지를 수정했습니다. 아래 최초 생성 프롬프트는 이력으로 보존합니다.

```text
Edit the provided RAGAS-JEV Korean infographic for README use. Remove the entire bottom benchmark section: heading "RAGAS와 비교한 대표 벤치마크", higher-is-better icon, all comparison table rows, all numeric results, and the two notes under the table. Remove the entire "해석 시 참고" footer too, including benchmark qualifications, source/version/date text. Preserve all content above that section: title, subtitle, the LLM → JEV → 프로그램 flow, probability-average strip, all four metric cards (01 Context Precision, 02 Context Recall, 03 Faithfulness, 04 Answer Relevancy), and full-width 05 Evaluation Uncertainty panel, with original exact text, icons and colors. Make the image end cleanly just below the Evaluation Uncertainty panel with a small balanced white margin. Reflow to a compact near-square canvas, no blank void where table was. Preserve original navy/teal/blue/lavender palette, white background, clean Korean sans-serif type and sharp legibility. No new content or benchmark text. Output exactly one polished high-resolution image.
```

## 최초 생성

- 생성 방식: built-in image_gen
- 원본 문서: `docs/ragas-jev_평가항목.md`
- 이미지: `ragas-jev-evaluation-infographic.png`

```text
Use case: infographic-diagram
Create ONE polished Korean infographic raster image for the README of RAGAS-JEV, about 1800 x 2200 pixels portrait, high resolution crisp Korean typography. White/off-white background, dark navy text, teal and indigo accents, subtle pale cards, precise thin-line icons, generous whitespace. Editorial technical infographic, no photography, no fake logos, no decorative clutter. Large legible text; thoughtfully condensed information. All exact Korean and English text below must be rendered correctly. Layout: strong header; slim 3-step flow; 2x2 equal metric cards; full-width uncertainty panel; compact benchmark comparison table; small readable qualification footer. Do not add claims or statistics. No huge empty margins. The four metric cards have small appropriate icons and clear 01–04 numbers. Distinguish retrieval row from generation row. Text hierarchy should allow comfortable reading at README width 1000px.

Exact content:
HEADER:
"RAGAS-JEV"
"평가 항목 한눈에 보기"
"검색 품질 · 답변 품질 · 판정 신뢰도"

FLOW, three connected boxes:
"LLM" / "평가 단위 분해"
"JEV" / "단위별 확률 판정"
"프로그램" / "점수 계산"
Below flow: "주 점수 = 단위별 관련·지원 확률의 평균"

CARD 01:
"검색 품질"
"Context Precision"
"쓸모 있는 문서를 검색했는가?"
"단위: 검색 chunk"
"관련 확률 평균 + 순위 가중·AP 점수"
"reference가 없으면 질문 기반으로 평가"

CARD 02:
"검색 품질"
"Context Recall"
"정답에 필요한 정보를 찾았는가?"
"단위: reference의 원자적 claim"
"검색 문서의 claim 지원 확률 평균"
"reference 필수"

CARD 03:
"답변 품질"
"Faithfulness"
"답변의 주장이 근거로 뒷받침되는가?"
"단위: 답변의 원자적 claim"
"검색 문서의 claim 지원 확률 평균"
"숫자·단위 불일치도 별도 진단"

CARD 04:
"답변 품질"
"Answer Relevancy"
"질문에 맞는 내용으로 답했는가?"
"단위: 답변 statement"
"관련 확률 평균 + 전체 답변 4단계 척도"
"사실 여부와 관련성을 분리 · 임베딩 불필요"

FULL-WIDTH PANEL:
"05  Evaluation Uncertainty"
"이 판정을 얼마나 믿을 수 있는가?"
"confidence · uncertainty = 1 − confidence"
"단위별 확률과 판정 출처 추적"
"저신뢰 판정 재검토: Precision → Strong Judge / Faithfulness → Auditor"
"JEV와 LLM의 불일치는 사람 검수로 연결"

BENCHMARK:
"RAGAS와 비교한 대표 벤치마크"
Columns: "평가 · 측정 기준" | "RAGAS" | "RAGAS-JEV"
Rows exactly:
"Precision · 질문 기반 Spearman" | "0.111 / 0.253" | "0.681 / 0.567"
"Faithfulness · RAGTruth AUROC" | "0.790" | "0.830"
"Recall · Pearson" | "0.909 / 0.944" | "0.903 / 0.938"
"Relevancy · 무관 문장 삽입 쌍 정확도" | "0.605 / 0.490" | "0.980 / 0.950"
Highlight winning values per row: JEV for rows 1,2,4; RAGAS for row 3. Show upward arrow near table heading indicating higher is better, not an extra column.
Table notes:
"수치는 Hybrid 기준 · a / b = 한국어 / 영어 · 높을수록 좋음"
"Precision은 RAGAS ContextRelevance와 비교 · Relevancy는 JEV = Hybrid"

FOOTER readable:
"해석 시 참고"
"일부 합성 데이터 포함 · RAGAS Relevancy는 로컬 다국어 임베딩 사용"
"서비스 도메인 적용 전 재보정·검증 필요 · 분해 품질은 LLM에 의존"
"출처: ragas-jev_평가항목.md · Phase 6 벤치마크"
"2026-09-24 기준 · JEV 1.13.0 · ragas 0.4.3"

Constraints: exactly one cohesive image, accurate numbers, Korean text no misspellings, no claims of universally superior performance, no computer frame or mockup.
```
