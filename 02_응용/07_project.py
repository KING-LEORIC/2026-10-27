"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# split 매니페스트 파일에서 train 및 valid에 해당하는 request_id를 position 순서대로 추출
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# 추출한 ID를 기반으로 원본 데이터에서 train과 valid 데이터프레임 구성
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
# 모델 학습에 사용할 4가지 특성(Features : "input_chars", "allowed_tools", "task_type", "priority") 정의
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# - 수치형 특성(FEATURES[:2]) : 중앙값(median)으로 결측치 대체
# - 범주형 특성(FEATURES[2:]) : 최빈값(most_frequent)으로 결측치 대체 후 원-핫 인코딩 수행
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix

# 5개 폴드로 나누고 클래스 비율을 유지하는 층화 K-Fold 인덱스 리스트 생성 (재현성을 위해 난수 시드 고정)
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
# 비교할 4개의 머신러닝 분류 모델 후보 정의 (Dummy, Tree, Forest, XGBoost)
candidates = [("Dummy", DummyClassifier(strategy="most_frequent")),
              ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1)), ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
rows, cv_rows = [], []
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})
for name, estimator in candidates:
    # 전처리 단계와 머신러닝 모델을 하나로 묶은 파이프라인 생성
    # clone을 통해 매번 깨끗한 전처리기 사용
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07a: train 내부 교차검증 뒤 전체 train으로 학습하세요.
    
    # 5-fold 교차 검증을 수행하여 각 폴드별 F1 스코어 측정
    # - model : 전처리와 모델이 포함된 파이프라인
    # - X_train, y_train : 교차 검증에 사용할 학습 특성과 타겟 데이터
    # - cv=folds : 미리 정의된 5-fold 교차 검증 인덱스 리스트
    # - scoring='f1' : 평가 지표로 F1 Score 사용
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring='f1')
    # - model : 학습을 진행할 파이프라인 객체
    # - X_train, y_train : 전체 학습 데이터의 특성과 타겟
    model.fit(X_train, y_train)
    # pass
    # TODO 끝
    # TODO A07b: valid의 예측을 prediction에 넣으세요.
    
    # 학습이 완료된 모델을 활용하여 검증 데이터셋(X_valid)에 대한 예측값 생성
    # - model : 전체 훈련 데이터로 학습이 완료된 파이프라인 객체
    # - X_valid : 예측에 사용할 검증 특성 데이터
    prediction = model.predict(X_valid)
    # TODO 끝
    # 검증 세트 예측 결과에 대한 혼동 행렬(Confusion Matrix) 계산 (True Negative, False Positive, False Negative, True Positive)
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    # 모델별 교차 검증 요약 통계(평균, 표준편차) 및 훈련/검증 F1 스코어, 오분류 건수(FP, FN) 기록
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
    # 각 폴드별 개별 F1 스코어 값을 상세 목록에 추가 (fit_n=48, evaluate_n=12는 학습 및 평가 샘플 수)
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
    predictions[name] = prediction
    # 모델이 'Forest'(랜덤 포레스트)일 경우, 각 특성의 중요도(Feature Importance) 데이터프레임 생성
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
print(comparison.round(4).to_string(index=False))
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
