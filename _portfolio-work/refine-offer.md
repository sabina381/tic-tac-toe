# 틱택토(AlphaZero) 프로젝트 — 개선 제안 (2단계: 개선)

> `codes/`, `robotics/` 폴더의 실제 코드를 다시 읽고 찾은 내용입니다. 제 판단으로 원인을 추정한 부분은 모두 *(추정)* / **확인 필요**로 표시했습니다. 전부 다 고칠 필요는 없고, 포트폴리오에 "발견했지만 시간상 다음 개선 과제로 남겼다"고 적어도 됩니다.

---

## 1. `net.py` / `train_network.py` — policy·value loss function이 뒤바뀐 것으로 보임 **(확인 필요, 중요도 높음)**

`net.py`의 두 head 출력 형태:

```python
# net.py
self.policy_head = nn.Sequential(
    ...,
    nn.Softmax(dim=1)   # 행동 개수만큼의 "확률분포"
)
self.value_head = nn.Sequential(
    ...,
    nn.Tanh()           # [-1, 1] 사이 "스칼라" 1개
)
```

그런데 `train_network.py`의 `_loss_function`은 이렇게 연결돼 있습니다.

```python
# train_network.py, line 134
def _loss_function(self, pred_policy, pred_value, y):
    y_policy = y[:, :-1]
    y_value = y[:, -1:]

    mse = F.mse_loss(pred_policy, y_policy)              # 확률분포에 MSE
    cross_entropy = self.cross_entropy(pred_value, y_value)  # 스칼라에 CrossEntropy
    return mse, cross_entropy, mse + cross_entropy
```

일반적인 AlphaZero 구성은 정반대입니다 — **policy(확률분포)는 cross-entropy 계열, value(스칼라)는 MSE**로 학습합니다. `CrossEntropyLoss`는 원래 "클래스가 여러 개인 분류"를 위한 손실 함수라서, 지금처럼 클래스가 사실상 1개(스칼라 1개)인 입력에 적용되면 내부적으로 `softmax(단일값) = 1`이 되어 값과 거의 무관하게 손실이 0에 가깝게 나올 수 있습니다.

*(추정)* 이게 바로 1단계 정리에서 이미 관찰됐던 **"value_loss가 거의 항상 0.0으로 찍히는 현상"**의 원인일 가능성이 높아 보입니다. 즉 value head가 실제로는 거의 학습되지 않고 있었을 수 있습니다.

**제안 (before/after)**:

```python
# before
mse = F.mse_loss(pred_policy, y_policy)
cross_entropy = self.cross_entropy(pred_value, y_value)

# after — policy에는 분포 간 차이(KL/Cross-Entropy 계열), value에는 회귀(MSE)
policy_loss = -(y_policy * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()  # cross-entropy
value_loss = F.mse_loss(pred_value, y_value)
```

`self.cross_entropy = CROSS_ENTROPY`(nn.CrossEntropyLoss 인스턴스)를 그대로 policy에 쓰려면 `pred_policy`가 softmax 이전의 logit이어야 하므로, `net.py`의 `policy_head` 마지막 `nn.Softmax`를 떼고 loss 쪽에서 CrossEntropyLoss가 내부적으로 softmax를 적용하게 하는 방법도 있습니다. 다만 그러면 추론(`mcts.py`의 `predict()`)에서 직접 softmax를 한 번 더 해줘야 하니, 구조를 얼마나 바꿀지는 선택 사항입니다.

같은 함수 안의 `loss.requires_grad_(True)` (line 178)도 같이 봐두면 좋습니다 — 모델 파라미터를 거쳐 나온 loss는 보통 이미 `requires_grad=True` 상태이므로, 이 호출이 필요했다는 것 자체가 연산 그래프 어딘가가 끊겼을 가능성을 시사합니다. 위 loss function을 고치면서 이 줄이 여전히 필요한지 같이 확인해보면 좋을 것 같습니다.

---

## 2. `config.py` — 경로/파일명이 실제 폴더 구조와 어긋남

```python
# config.py, line 8-9
F_PATH = "/Users/seungyeonlee/Documents/GitHub/tic-tac-toe/algorithm/train_files"
F_NAME = "250218-3"
```

- `F_PATH`: 실제 폴더명은 `algorithm`이 아니라 `codes`입니다. 지금 이 경로로는 `train_files`를 찾지 못합니다.
- `F_NAME`: 1단계에서부터 남아있던 항목으로, 실제 학습 로그 파일 접두사(노트북상 "250225" 계열)와 다릅니다.

**제안**:
```python
F_PATH = "/Users/seungyeonlee/Documents/GitHub/tic-tac-toe/codes/train_files"
F_NAME = "250225-1"  # 실제 사용한 학습 결과 파일 접두사로 교체 (정확한 값 확인 필요)
```

---

## 3. `mcts.py` — `predict()` / `Node.__init__`가 호출될 때마다 불필요한 객체를 새로 만듦

```python
# mcts.py, line 11-17
def predict(state, model):
    env = Environment(STATE_SIZE, WIN_CONDITION)   # 매번 새로 생성
    model = model.to(DEVICE)                        # 매번 device로 이동
    ...
```

```python
# mcts.py, line 43-46
class Node:
    def __init__(self, state, p, model):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)  # 노드마다 새로 생성
        self.model = model
```

`Environment(STATE_SIZE, WIN_CONDITION)`는 항상 같은 값으로 생성되는 상태 없는(stateless) 객체이고, `model.to(DEVICE)`도 이미 device에 있는 모델을 매번 다시 옮기는 동작입니다. `EVAL_CNT=200`이면 한 수를 둘 때마다 MCTS가 200번 시뮬레이션을 도는데, 그때마다 `predict()`가 불리고, 새 노드가 생성될 때마다 `Environment`도 새로 만들어집니다 — 게임 한 판(약 9수 이하)만 해도 수백~수천 번 반복되는 부분이라 누적되면 학습/평가 속도에 영향을 줄 수 있습니다.

**제안**: `env`를 모듈 레벨 상수나 `Mcts` 클래스가 한 번만 생성해서 하위 `Node`/`predict()`에 전달하는 방식으로 바꾸고, `model.to(DEVICE)`는 학습·평가 시작 시 한 번만 호출.

```python
# before
def predict(state, model):
    env = Environment(STATE_SIZE, WIN_CONDITION)
    model = model.to(DEVICE)
    ...

# after
_ENV = Environment(STATE_SIZE, WIN_CONDITION)  # 모듈 레벨에서 한 번만 생성

def predict(state, model):
    env = _ENV
    # model은 호출부(Mcts 등)에서 이미 .to(DEVICE) 된 상태로 전달
    ...
```

---

## 4. `evaluate.py` — 구현해둔 AlphaBeta 비교가 실제로는 꺼져 있음

```python
# evaluate.py, line 79-82
agents_dict = {'random': RandomAgent(False),
                # 'alpha-beta': AlphaBetaAgent(False),
                'mcs': McsAgent(False),
                'mcts': MctsAgent(False)}
```

`enemy_agents.py`에 `AlphaBetaAgent`(minimax+가지치기)까지 4종의 베이스라인 에이전트를 다 구현해 둔 게 이 프로젝트의 강점 중 하나인데, 실제 `evaluate_best_player()` 실행에서는 alpha-beta 비교 줄이 주석 처리되어 있어 random/MCS/pure MCTS 3종에 대해서만 결과가 남아 있습니다.

**제안**: 이 줄의 주석을 풀고 한 번 더 평가를 돌려서, "4종 베이스라인 전부와 비교"라는 원래 설계대로 결과를 남기면 평가 프레임워크의 완성도가 그대로 드러납니다. (주석 처리된 이유가 실행 시간 때문이었다면, 그 사실도 같이 적어두는 게 좋습니다 — *(추정)*.)

---

## 5. `State.py` — 사장 코드, 그대로 두면 실행도 안 됨

```python
# State.py, line 1-11
from Environment import Environment   # 존재하지 않는 모듈 (실제 파일명은 environment.py, 소문자)

class State(Environment):
    ...
```

`environment.py` 안에 실제로 쓰이는 `State` 클래스가 이미 내장되어 있고(`from environment import *`로 그쪽이 로드됨), 파이프라인의 어떤 파일도 `State.py`를 import하지 않습니다. 게다가 `State.py` 자체가 존재하지 않는 `Environment` 모듈(대문자, 실제로는 `environment.py`)을 import하고 있어서, 지금 이 파일만 따로 실행하면 바로 `ModuleNotFoundError`가 납니다.

**제안**: 리포지토리에 남겨두면 "정리 안 된 프로젝트"처럼 보일 수 있으니, 삭제하거나 `_archive/` 같은 폴더로 옮기고 README/주석에 "구버전, 미사용"이라고 명시하는 걸 추천합니다. (1단계 `pre-understand.md`에서도 이미 같은 항목이 "확인 필요"로 남아 있었는데, 이번에 사장 코드라는 점이 더 확실히 확인됐습니다.)

---

## 6. `robotics/serial_connection` — 사용하지 않는 테스트 스크립트의 baud rate 불일치

```python
# connect_CNC.py, line 14 (실제 사용된 버전)
ser = serial.Serial(serial_port, 9600, timeout=None)

# connect_CNC_input.py, line 5 (수동 테스트용 스크립트)
ser = serial.Serial('/dev/cu.usbserial-110', 115200)
```

`CNC_main.ino`(아두이노 쪽)는 9600 baud로 열려 있어서 `connect_CNC.py`(최종 데모에 쓰인 버전)와는 맞지만, `connect_CNC_input.py`는 115200으로 다르게 설정되어 있습니다. 이미 전에 `connect_CNC.py`를 최종본으로 정리했으므로 동작에는 문제가 없지만, 두 파일이 같은 폴더에 남아 있으면 나중에 보는 사람이 헷갈릴 수 있습니다.

**제안**: `connect_CNC_input.py` 상단에 "수동 테스트용, 실제 로봇 구동에는 미사용"이라는 주석을 추가하거나, `robotics/readme.md`에 두 파일의 역할 차이를 한 줄로 명시.

---

## 우선순위 메모

포트폴리오 관점에서 임팩트가 큰 순서로 보면:

1. **loss function (1번)** — 실제 결과(value_loss≈0.0)와 직접 연결되는, 프로젝트의 핵심 학습 로직에 관한 발견이라 우선순위가 가장 높습니다.
2. **evaluate.py alpha-beta 주석 해제 (4번)** — 이미 구현된 강점을 실제 결과로 증명하는 것이라 비용 대비 효과가 큽니다.
3. 나머지(2, 3, 5, 6번)는 완성도/가독성 관점의 정리로, 시간이 없다면 "발견했지만 남겨둔 개선 과제"로 문서화만 해도 충분합니다.
