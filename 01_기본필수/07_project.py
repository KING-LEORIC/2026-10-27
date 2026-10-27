"""실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
# # 데이터를 개발(develop) 세트와 최종 평가를 위한 테스트(test) 세트로 분할 (층화추출, 8:2 비율)
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
# 개발 세트를 다시 학습(train) 세트와 검증(valid) 세트로 분할 (75:25 비율, 전체 기준 6:2:2)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score

# 제공: 전날과 같은 입력 2열입니다. 실행시간과 failed는 입력에 넣지 않습니다.
features = ["input_chars", "task_type"]
X_train, X_valid = train[features], valid[features]
# 수치형 변수는 그대로 두고, 범주형 변수는 One-Hot Encoding을 수행하는 전처리 파이프라인 정의
pre = ColumnTransformer([
    ("num", "passthrough", ["input_chars"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
])

# 교차 검증 시 클래스 비율을 유지하면서 데이터를 n개로 나누기 위한 층화 K-Fold 객체 생성
# - n_splits=3 : 데이터를 3개의 폴드로 분할
# - shuffle=True : 데이터를 나누기 전에 무작위로 섞음
# - random_state=42 : 난수 시드 고정
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 비교할 2개의 머신러닝 분류 모델 후보 정의 - DecisionTree와 RandomForest
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]

rows, cv_rows = [], []
predictions = valid[["request_id", "failed"]].copy()

# 각 모델 후보별로 교차 검증 및 검증 세트 예측 수행
for name, estimator in candidates:
    # 제공: 전처리까지 Pipeline에 포함하여 fold 학습 부분에서만 fit하게 합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07a: 전처리까지 연결한 model을 train에서 교차검증하세요.

    # n_splits=3으로 설정된 folds 객체를 사용하여, X_train과 y_train을 3개의 폴드로 나눈 뒤 각 폴드마다 F1 스코어를 측정
    # - model : 평가할 머신러닝 모델 및 전처리가 포함된 파이프라인
    # - X_train, y_train : 교차 검증에 사용할 학습 특성과 타겟 데이터
    # - cv=folds : 위에서 정의한 3-fold 교차검증 객체 (n_splits=3)
    # - scoring='f1' : 평가 지표로 F1 Score 사용
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring='f1')
    # TODO 끝
    # TODO B07b: 전체 train으로 학습하고 valid를 예측하세요.
    # pass
    
    # 전체 훈련 데이터(X_train, y_train)를 사용하여 전처리 및 모델 학습 수행
    # - model : 학습을 진행할 파이프라인 객체
    # - X_train, y_train : 전체 학습 데이터의 특성과 타겟
    model.fit(X_train, y_train)
    # 학습된 모델을 통해 검증 데이터셋(X_valid)에 대한 예측값 생성
    # - model : 학습이 완료된 파이프라인 객체
    # - X_valid : 예측에 사용할 검증 특성 데이터
    prediction = model.predict(X_valid)
    # TODO 끝
    # 모델별 전체 요약 지표(CV 평균/표준편차, 검증 F1 스코어)를 목록에 추가
    rows.append({"model": name, "CV_F1": scores.mean(),
                 "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})
    
    # 폴드별 상세 F1 스코어 기록
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "F1": value})
    predictions[name] = prediction
    
    # 랜덤 포레스트 모델인 경우 특성 중요도(Feature Importance) 추출
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# 제공: 입력 파일을 덮어쓰지 않고 결과 4개를 같은 폴더에 만듭니다.
comparison = pd.DataFrame(rows)
comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")

# 실행 결과 터미널 출력
print("분할:", len(train), len(valid), len(test))
print(comparison.round(3).to_string(index=False))
print("중요도")
print(importance.round(3).to_string(index=False))
print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
