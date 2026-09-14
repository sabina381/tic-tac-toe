# pre-understand: TicTacToe AlphaZero + CNC Robot (개인 리포 tic-tac-toe)

> 이 문서는 `tic-tac-toe` 리포(algorithm/, robotics/)에 있는 코드를 직접 읽고 정리한 것입니다. 팀 리포 `24-2-TicTacToe`의 맥락(전체 프로젝트 배경, 팀 구성 등)은 이전 대화에서 확인한 내용을 참고용으로만 인용했고, 팀 리포 자체는 이번 작업에서 열람·수정하지 않았습니다.

## 프로젝트 배경 *(참고용, 팀 리포 기준)*

이 리포는 6인 팀 캡스톤 프로젝트 `24-2-TicTacToe`(팀장: 이승연/AI·Robotics 담당)에서 파생된 개인 포트폴리오용 정리본입니다. 팀 프로젝트는 (1) 신경망 기반 틱택토 AI 에이전트(AlphaZero), (2) 카메라로 보드를 인식하는 CV 모듈, (3) 실제 보드에 말을 그리는 CNC 로봇팔, 세 부분으로 구성되어 있었습니다. 이 개인 리포에는 그중 본인이 담당한 (1) AI 알고리즘과 (3) CNC 로봇 제어 코드만 가져왔고, 팀원이 담당한 CV 모듈은 제외했습니다.

## 개발 스택

- 알고리즘: Python, PyTorch (ResNet 기반 policy-value network), numpy, pandas, matplotlib/seaborn
- 로봇 제어: Arduino(C++), pyserial

---

## 1. `algorithm/` — 탐색 알고리즘 & AlphaZero 에이전트

### 1-1. 초기 탐색 알고리즘 실험 (minimax → monte-carlo → MTCS)

세 노트북은 이후 AlphaZero로 발전하기 전, 탐색 알고리즘을 하나씩 직접 구현해보며 단계적으로 실험한 기록입니다.

| 파일 | 무엇을 | 왜 | 결과/비고 |
|---|---|---|---|
| `minimax.ipynb` | 틱택토 전체 게임 트리를 완전 탐색하는 minimax(다이내믹 프로그래밍 방식) 구현 | "틱택토 정도로 작은 상태공간에서는 depth 제한 없이 전체 탐색이 가능하다"는 판단 하에 베이스라인으로 구현 | 마크다운 메모에 "수동 코딩으로 오류 잡음", "논리 오류를 수동으로 확인해서 찾아냄"이라는 기록이 남아 있어, 직접 케이스를 짚어가며 디버깅한 과정이 보임 |
| `monte-carlo.ipynb` | 랜덤 플레이아웃으로 가치를 추정하는 순수 몬테카를로 방법 구현 (+ minimax 비교 포함) | 완전 탐색이 불가능한 상황을 가정하고 확률적 추정 방식을 실험 | - |
| `MTCS.ipynb` | UCT(UCB1 기반 트리 탐색) 방식의 Monte Carlo Tree Search 구현 | "MTCS 코드 레퍼런스" 링크를 남겨두고 직접 참고하며 구현 *(추정: 블로그 자료를 참고해 학습 후 직접 작성)* | Node 클래스와 UCT 계산을 분리해서 별도로 테스트했다는 메모("헷갈려서 밖에 빼서 따로 시험해봄") 있음 |

### 1-2. AlphaZero 파이프라인 (환경 → 신경망 → MCTS → 학습 → 평가)

| 파일 | 무엇을 | 구체적 기술 | 비고 |
|---|---|---|---|
| `environment.py` | 게임 환경(`Environment`)과 게임 상태(`State`) 클래스. n목(`win_condition`으로 일반화, 현재 3×3/3목) 승패 판정, 한 게임 진행(`play_one_game`), 두 에이전트 간 대전 평가(`evaluate_first_algorithm`), MCTS policy를 히트맵+보드로 시각화해 이미지로 저장(`save_game_image`, matplotlib/seaborn) | numpy, matplotlib, seaborn | **이 파일 안에 `State` 클래스가 이미 정의되어 있음** — 자신의 수/상대 수를 좌표별로 나누고(`my_state`, `enemy_state`), `history`로 과거 상태를 누적해 신경망 입력에 활용 |
| `net.py` | 정책(policy)·가치(value)를 함께 출력하는 신경망 두 종류: Residual Block 기반 `ResNet`, 더 단순한 2-conv 구조의 `Net` | PyTorch (`nn.Conv2d`, `BatchNorm2d`, residual connection) | policy head는 softmax, value head는 tanh 출력 |
| `mcts.py` | 신경망 predict 결과(policy, value)로 트리를 확장하는 `Node`, PUCT 공식으로 자식 노드를 선택하는 `Mcts` 클래스 | PUCT = `-w/n + C_PUCT * p * sqrt(N)/(1+n)`, 온도(temperature) 기반 볼츠만 분포로 행동 샘플링 | AlphaZero 논문의 MCTS 구조를 직접 구현 *(추정: policy-value 결합 PUCT 방식은 AlphaZero/AlphaGo Zero 논문 구조를 참고)* |
| `enemy_agents.py` | 학습된 모델과 비교할 베이스라인 에이전트 4종: `RandomAgent`, `AlphaBetaAgent`(minimax+alpha-beta pruning), `McsAgent`(순수 몬테카를로 서치), `MctsAgent`(신경망 없는 순수 MCTS, UCB1 사용) | numpy, alpha-beta pruning, UCB1 | 1-1의 세 노트북에서 실험했던 탐색 알고리즘들을 재사용 가능한 baseline 에이전트로 재구현한 것으로 보임 |
| `file_save_load.py` | 학습 history/모델 파라미터를 pickle로 저장·로드 | pickle, `collections.deque`(history 최대 길이 `MEM_SIZE` 제한) | 파일이 없으면 새로 생성하는 방어 코드 포함 |
| `train_network.py` | `TrainNetwork` 클래스 — self-play로 데이터 생성(`_self_play_one_game`), 8방향 대칭 변환으로 데이터 증강(`save_self_play_history`), MSE(policy)+CrossEntropy(value) loss로 학습(`train_network`), 이 전체를 에피소드 단위로 반복하며 주기적으로 평가까지 돌리는 `train_cycle` | PyTorch, Adam optimizer, 8종 대칭(전치/좌우반전/상하반전 조합) 데이터 증강 | 예전에는 self-play·학습 루프가 `self_play.py`/`train_cycle.py`로 분리돼 있었는데, 이 버전에서는 `TrainNetwork` 클래스 메서드로 통합됨 |
| `evaluate.py` | `Evaluate` 클래스 — 최신 모델 vs 이전 best 모델 대전(`evaluate_network`, 기준 넘으면 best 교체) + best 모델과 4종 베이스라인 에이전트(random/mcs/mcts) 간 대전(`evaluate_best_player`), 대전 결과를 이미지로 저장 | pandas(결과 누적), pickle | 예전 `evaluate_best_player.py` + `evaluate_network.py` 두 파일의 기능을 한 클래스로 통합 |
| `config.py` | 보드 크기, 승리 조건, 학습률, MCTS 파라미터(C_PUCT, EVAL_CNT), 배치 크기, 평가 주기 등 하이퍼파라미터 전부를 한 곳에서 관리 | `torch.device("mps")`로 Apple Silicon GPU 사용 | `F_PATH`는 이번에 개인 리포 경로(`tic-tac-toe/algorithm/train_files`)로 수정함(원래는 팀 리포 경로였음) |
| `visualizing.py` | 저장된 대전 결과(pickle)를 불러와 라벨별로 win/draw/lose 등 지표를 5×4 subplot으로 시각화 | matplotlib | - |
| `main_tictactoe.ipynb` | 위 모듈들을 모두 import해서 실제로 50 에피소드 학습을 돌린 실행 노트북. 학습 후 랜덤/MCS/MCTS 에이전트와 대전 평가, 마지막엔 사람이 직접 커맨드라인 입력으로 학습된 에이전트와 대국하는 셀도 포함 | - | **실제 실행 결과**: 50 episode 학습 후 policy loss 0.04087 / value loss ≈0. 평가 에피소드(40회차 기준)에서 vs random 승률 0.95(9승 1무 0패), vs MCS 0.5(2승 6무 2패), vs 순수 MCTS 0.5(0승 10무 0패 — 한 번도 지지 않고 전부 비김) |

### 1-3. 발견한 사항 (확인 필요)

- **`State.py` 파일은 죽은 코드로 보입니다.** 이 파일은 `from Environment import Environment`를 참조하는 구버전 `State` 클래스를 담고 있는데, 실제 파이프라인(`train_network.py`, `evaluate.py`, `mcts.py`, `main_tictactoe.ipynb`)은 전부 `environment.py` 안에 정의된 `State` 클래스를 `from environment import *`로 가져다 씁니다. `State.py`를 import하는 코드는 어디에도 없었습니다. `net.py`가 예전 `ResNet.py`를 대체한 것과 같은 패턴으로, `State.py`도 정리 대상일 가능성이 있습니다 — 필요하시면 다음 정리 때 제거해드릴 수 있어요.
- `config.py`의 `F_NAME` 값은 `"250218-3"`으로 되어 있지만, 실제 학습 실행 결과 파일명은 `250225_*`로 남아있어 노트북을 실행할 때 `F_NAME`을 직접 바꿔서 썼던 것으로 보입니다 *(추정)*.

---

## 2. `robotics/` — CNC 로봇 제어

### 2-1. `controlCNC/` (Arduino, C++)

| 파일 | 무엇을 | 구체적 기술 |
|---|---|---|
| `StepMotor.h` / `Stepper.cpp` | 28BYJ-48(5V) 스텝모터 1개를 제어하는 `Stepper` 클래스. 하프스텝 8단계 시퀀스로 핀 신호를 직접 제어 | Arduino 핀 제어, 하프스텝 시퀀싱 |
| `MultiStepper.cpp` | x/y 두 스텝모터를 동시에 구동해 대각선으로 움직이는 `MultiStepper` 클래스 | 두 모터 동기 제어(대각선 이동 로직 직접 구현) |
| `TicTacToeArtist.cpp` | x/y(평면 이동) + z(펜 업다운) 3개의 스텝모터로 3×3 보드에 O/X를 그리고, 보드 격자선도 그리는 `TicTacToeArtist` 클래스. 그리드 좌표(`board_width_grid` 등)를 셀 단위로 환산해서 위치를 계산 | 좌표 변환, 펜 플로터 방식의 도형 그리기(원=8방향 이동 조합, X=대각선 2개) |
| `robot_example.ino` | 위 클래스들을 사용해 실제로 모터 핀을 연결하고 도형을 그려보는 테스트용 스케치. 대부분의 실행 라인이 주석 처리되어 있어 개별 기능을 하나씩 테스트했던 흔적 | - |

### 2-2. `serial_connection/` (PC ↔ Arduino 통신)

| 파일 | 무엇을 | 비고 |
|---|---|---|
| `CNC_main.ino` | 시리얼로 받은 명령 문자열(예: `"O4"`, `"X1"`, `"S"`)의 첫 글자로 동작(원/엑스/보드그리기)을, 나머지 숫자로 위치를 파싱해서 `TicTacToeArtist`를 호출하는 최종 펌웨어 | 9600 baud |
| `connect_CNC.py` | PC에서 이 펌웨어로 명령을 보내는 `Serial` 클래스(`draw_game_board`, `send_to_robot`, 응답 대기 `waiting_robot`) | pyserial, 9600 baud — 실제 최종 통합 데모(`final/main.ipynb`, 팀 리포)에서 이 버전을 사용 |
| `connect_CNC_input.py` | 사람이 터미널에서 직접 `S`/`O`/`X` + 위치를 입력하며 로봇에 명령을 하나씩 보내볼 수 있는 수동 테스트 스크립트 | 115200 baud (위 `CNC_main.ino`의 9600과 baud rate가 다름 — 실사용 시 주의 필요) |

---

## 요약 — 프로젝트 하이라이트 후보 (추정 포함)

* 틱택토에 AlphaZero(ResNet 기반 policy-value 네트워크 + PUCT-MCTS + self-play + 데이터 증강)를 처음부터 구현. `win_condition`/`state_size`를 파라미터로 분리해 다른 보드·규칙(예: 오목)으로도 확장 가능하도록 환경을 일반화해서 설계.
* Random / MCS(순수 몬테카를로) / 순수 MCTS(UCB1) / AlphaBeta(minimax+alpha-beta pruning) 4종의 비-신경망 에이전트를 직접 구현해, 신경망 유무에 따른 성능 차이를 정량적으로 비교할 수 있는 평가 프레임워크를 갖춤.
* 최신 모델이 기존 best 모델과의 대전(승점 기준 `CRITERIA=0.5` 초과)을 통과해야만 best로 교체되는 챔피언-도전자 구조를 구현해, self-play 학습 과정의 성능 퇴행을 방어.
* 8방향 대칭 변환(전치·좌우반전·상하반전 조합)으로 self-play 데이터를 증강해, 적은 게임 수로도 학습 데이터 효율을 높임.
* 실행 로그 기준 50 episode 학습만으로 vs random 승률 0.95(9승 1무 0패), vs 순수 MCTS는 20게임 전부 무승부(무패) 기록 — 틱택토는 최적 대 최적이면 항상 무승부로 끝나는 게임이라는 점을 고려하면, 적은 학습량 대비 최적 플레이에 근접한 결과로 해석 가능 *(표본이 10~20게임 수준으로 적어 통계적 신뢰도는 확인 필요)*.
* 소프트웨어 알고리즘에 그치지 않고, 스텝모터 제어 라이브러리(`Stepper`/`MultiStepper`)와 좌표→모터 스텝 변환, 도형 근사 알고리즘(원=8방향 이동 근사, X=대각선 2획)을 직접 설계해 신경망 에이전트가 계산한 수를 실제 로봇팔이 물리적으로 그리도록 구현 — AI 이론·RL 엔지니어링·임베디드 제어를 한 프로젝트 안에서 모두 다룸.
* minimax(완전탐색) → 몬테카를로 → MTCS(UCT) → AlphaZero로 이어지는 탐색 알고리즘을 단계적으로 직접 구현한 과정이 노트북에 그대로 남아있어, 이론 이해를 구현으로 증명하는 성장 서사로 활용 가능.
* 다만 학습 규모(50 episode, self-play 1게임/episode)가 작아 "확실히 강하다"기보다는 "초기 수렴 경향을 보여주는 수준"이며, `State.py`처럼 정리되지 않은 미사용 코드가 남아있는 등 정리가 필요한 부분도 있음 *(Stage 2 대상)*.

## 총평

이 프로젝트는 "이론 이해 → 직접 구현 → 정량적 검증 → 하드웨어 통합"까지 한 사이클을 완주했다는 점에서 개인/동아리 프로젝트치고는 폭이 넓다. minimax(완전탐색) → 몬테카를로(확률적 추정) → MTCS(UCT 트리탐색) → AlphaZero(신경망+PUCT)로 이어지는 구현 순서 자체가 기초 탐색 알고리즘부터 차근차근 쌓아 올려 논문 수준 알고리즘까지 도달한 과정을 보여준다. AlphaZero 구현은 PUCT 기반 MCTS, ResNet policy-value 네트워크, 8방향 대칭 데이터 증강, self-play와 챔피언-도전자(best model 교체) 로직까지 핵심 요소를 실제로 갖췄고, random/MCS/순수 MCTS라는 난이도가 다른 세 상대를 두고 정량적으로 평가하는 프레임워크도 직접 구축했다. 여기에 그치지 않고 스텝모터 구동 라이브러리를 C++로 직접 짜고 좌표 변환·도형 근사 로직까지 만들어 로봇이 실제로 O/X를 그리게 만든 부분은, AI 이론과 임베디드 제어를 모두 다뤘다는 점에서 확실한 차별점이 된다.

설계 세부에서도 논문을 그대로 베끼기보다 상황에 맞게 판단한 흔적이 보인다. `win_condition`/`state_size`를 파라미터로 분리해 다른 보드·규칙(예: 오목)으로 확장 가능하도록 환경을 일반화했고, 히스토리 프레임은 틱택토처럼 짧은 게임에서는 크게 중요하지 않다고 보고 `NUM_HISTORY=0`으로 단순화했다. 무승부 보상을 0이 아닌 `0.1`로 준 것은 패배보다 무승부를 명확히 선호하도록 가치함수를 유도하려는 튜닝으로 보이고, 온도 스케줄링(초반 탐험 → 후반 수렴)이나 챔피언-도전자 검증 구조는 AlphaGo Zero가 학습 불안정성을 막기 위해 쓴 장치를 이해하고 그대로 적용한 것으로 판단된다.

실제 실행 결과(50 episode 학습)도 다시 보면 의미가 있다. vs random 승률 0.95는 최소한의 패턴을 확실히 익혔다는 뜻이고, vs 순수 MCTS에서 무패(전부 무승부)를 기록한 것은 — 틱택토가 최적 플레이 기준 항상 무승부로 끝나는 게임이라는 점을 고려하면 — 제한된 학습량에도 불구하고 최적에 가까운 수준까지 도달했다고 해석할 수 있다. vs MCS 승률이 애매한 것은 MCS가 트리 구조 없이 랜덤 플레이아웃만 반복하는 약한 탐색임에도 도메인 지식 없이 우연히 좋은 수를 찾는 경우가 있어 발생하는, 표본이 적을 때 흔한 노이즈로 보인다.

로봇 통합은 알고리즘과는 다른 종류의 문제를 요구했다. 스텝모터는 연속적인 곡선을 그릴 수 없어 O를 8방향 이동으로 근사한 다각형으로, X는 대각선 두 획을 펜업/다운으로 조합해 그리는 방식을 택했고, `steps_per_grid`처럼 보드 한 칸에 해당하는 모터 스텝 수도 실측 캘리브레이션을 거쳤을 것으로 보인다. 파이썬(의사결정)과 아두이노(모터 제어) 사이를 시리얼 프로토콜로 엮어, 신경망이 계산한 수가 실제 로봇 팔의 움직임으로 이어지는 전체 파이프라인을 완성했다.

다만 학습 규모(50 episode, self-play 1게임/episode)가 작아 "확실히 강하다"기보다는 "초기 수렴 경향을 보여주는 수준"이고, 평가 게임 수(10~20게임)도 적어 승률의 통계적 신뢰도는 낮다. 여러 버전의 코드가 폴더별로 흩어져 있었던 점, `State.py`처럼 정리되지 않은 미사용 코드가 남아있는 점도 정리가 필요하다(Stage 2 대상). 종합하면, AI 이론(탐색 알고리즘 진화) · 강화학습 엔지니어링(AlphaZero 파이프라인 및 설계 판단) · 하드웨어 통합(로봇팔)이라는 세 축을 한 프로젝트로 보여줄 수 있는 소재이며, 결과를 과장하지 않고 "적은 학습량 대비 최적에 근접"이라는 정직한 프레이밍으로 문서화하면 설득력 있는 포트폴리오가 될 것으로 판단된다.
