# 틱택토(AlphaZero) 프로젝트 — 작업 진행 상황

**마지막 업데이트**: 2026-09-14

## 전체 진행 단계
- [x] 1단계 파악 — pre-understand.md
- [~] 2단계 개선 — refine-offer.md 완료 / trouble-shooting.xlsx 아직 미작성
- [ ] 3단계 문서화 — README.md / tic-tac-toe_report.md / 분석 보고서
- [ ] 4단계 전시 — tic-tac-toe_portfolio.md

## 지금까지 만든 파일
| 파일 | 상태 | 비고 |
|---|---|---|
| `_portfolio-work/pre-understand.md` | 완료 | algorithm/robotics 코드 전체 요약 + "요약 — 하이라이트 후보" + "총평" 섹션까지 포함 |
| `_portfolio-work/refine-offer.md` | 완료 | 코드 재검토 기반 개선 제안 6건 (loss function 반전 의심, config 경로/파일명 불일치, mcts predict 비효율, evaluate 비교 미완성, State.py 사장 코드, 로봇 시리얼 스크립트 정리) |
| `_portfolio-work/trouble-shooting.xlsx` | 미작성 | 필요하면 요청 시 refine-offer.md 내용을 표로 추출해 생성 가능 |

## 마지막으로 하던 작업
`codes/`, `robotics/` 폴더의 실제 코드를 다시 훑어서 2단계 개선 제안(`refine-offer.md`)을 작성함. 1단계에서 이미 발견됐던 이슈(State.py 사장 코드, config 파일명 불일치)를 재확인하고, 코드를 더 자세히 읽으면서 loss function 설계가 뒤바뀐 것으로 보이는 점과 MCTS 예측 함수의 비효율 등 새로운 항목을 추가로 발견함.

## 다음에 할 일
- [ ] `trouble-shooting.xlsx` 작성 여부 결정 (원하면 refine-offer.md 내용을 표로 옮겨 생성)
- [ ] refine-offer.md의 제안 중 실제로 반영할 항목 결정 (전체를 다 고칠 필요는 없음 — 포트폴리오 관점에서 무엇을 고치고 무엇을 "알고 있었지만 시간상 미반영"으로 남길지 판단)
- [ ] 3단계 문서화: 프로젝트 루트 `README.md`, `docs/tic-tac-toe_report.md` 작성

## 확인/결정 필요한 사항 (열린 질문)
- `config.py`의 `F_PATH`가 `.../tic-tac-toe/algorithm/train_files`를 가리키는데, 실제 폴더명은 `algorithm`이 아니라 `codes`로 되어 있음 — 리포 재구성 중 폴더명이 바뀌면서 생긴 불일치로 추정, 수정 필요
- `config.py`의 `F_NAME = "250218-3"`이 실제 학습 로그 파일 접두사(노트북상 "250225" 계열)와 다름 — 1단계에서부터 이어진 미해결 항목
- `net.py`/`train_network.py`의 loss function에서 policy(softmax 출력)는 MSE로, value(tanh 출력)는 CrossEntropyLoss로 학습되고 있어 일반적인 AlphaZero 구성과 반대로 보임 — 실제 학습 로그의 `value_loss ≈ 0.0`과 연결되는 것으로 추정되지만, 직접 실행해서 재확인이 필요
- `State.py`를 그대로 삭제해도 되는지 — 현재 파이프라인 어디에서도 참조되지 않고, 파일 자체도 존재하지 않는 모듈(`Environment.py`)을 import하고 있어 실행도 되지 않는 상태

## 메모
(선택 — 특이사항 없음)
