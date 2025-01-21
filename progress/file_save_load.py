# import
import pickle

from ResNet import Net

# history 불러오는 함수
def load_history(file):
    try:
        with open(file, 'rb') as f:
            history = pickle.load(f)
    except FileNotFoundError:
        history = [] # 파일이 비어있는 경우 빈 리스트 생성

    return history

# history 저장 함수
def save_history(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

# model 파라미터를 가져오는 함수
def load_model(file):
    model = Net()
    with open(file, 'rb') as f:
        model.load_state_dict(pickle.load(f))
    return model

# 최근 모델 저장 함수
def save_model(file, model):
    with open(file, 'wb') as f:
        pickle.dump(model.state_dict(), f)


# 학습 지표 저장 함수
def save_visualizing_index(file, data):
    try:
        with open(file, 'rb') as f:
            df = pickle.load(f)
        df.append(data)
    except FileNotFoundError:
        df = data # 파일이 비어있는 경우 빈 리스트 생성

    with open(file, 'wb'):
        pickle.dump(df, f)
