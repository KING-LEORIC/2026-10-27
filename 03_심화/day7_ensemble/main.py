"""실습 07 심화: 전날 모델과 트리·앙상블의 분류·회귀 실험표를 생성합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.

import json
import platform
import importlib.metadata as metadata
import joblib
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.inspection import permutation_importance
from preprocessing import build_model
from evaluate import classification_metrics, regression_metrics

# 제공: 후보 순서는 CV 동률일 때도 사용합니다. test·valid를 보고 후보를 더 늘리지 않습니다.
# 분류 모델 후보군 정의 (형식: [("모델이름", 모델객체, 스케일링적용여부)])
# - Dummy : 가장 빈도가 높은 클래스로만 무조건 예측하는 가장 단순한 기준 모델 (스케일링 불필요)
# - Logistic : 규제 강도 C=1, 최대 반복 1000회인 로지스틱 회귀 모델 (선형 모델이므로 스케일링 필요: True)
# - KNN3 : 이웃의 수 k=3을 사용하는 K-최근접 이웃 분류 모델 (거리를 계산하므로 스케일링 필요: True)
# - Tree : 최대 깊이 3, 잎 노드 최소 샘플 수 3인 의사결정 나무 (트리 계열은 스케일링 불필요)
# - Forest : 트리 100개, 최대 깊이 4인 앙상블 랜덤 포레스트 모델 (트리 계열은 스케일링 불필요)
# - XGB : 트리 80개, 학습률 0.05인 XGBoost 부스팅 모델 (트리 계열은 스케일링 불필요)
classifiers = [("Dummy", DummyClassifier(strategy="most_frequent"), False),
               ("Logistic", LogisticRegression(C=1, max_iter=1000), True),
               ("KNN3", KNeighborsClassifier(n_neighbors=3), True),
               ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42), False),
               ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1), False),
               ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1), False)]
# 회귀 모델 후보군 정의 (형식: [("모델이름", 회귀모델객체, 스케일링적용여부)])
# - Dummy : 학습 데이터의 평균값(mean)으로만 무조건 예측하는 회귀 기준 모델 (스케일링 불필요)
# - Linear : 일반 선형 회귀(Linear Regression) 모델 (선형 모델이므로 스케일링 필요: True)
# - KNN15 : 이웃의 수 k=15를 사용하는 K-최근접 이웃 회귀 모델 (거리를 계산하므로 스케일링 필요: True)
# - Tree : 회귀용 의사결정 나무 모델 (트리 계열은 스케일링 불필요)
# - Forest : 회귀용 랜덤 포레스트 모델 (트리 계열은 스케일링 불필요)
# - XGB : 회귀용 XGBoost 부스팅 모델 (트리 계열은 스케일링 불필요)
regressors = [("Dummy", DummyRegressor(strategy="mean"), False),
              ("Linear", LinearRegression(), True),
              ("KNN15", KNeighborsRegressor(n_neighbors=15), True),
              ("Tree", DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42), False),
              ("Forest", RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                  max_features=1.0, random_state=42, n_jobs=1), False),
              ("XGB", XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="reg:squarederror", eval_metric="rmse",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1), False)]
class_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
reg_folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
class_rows, reg_rows, cv_rows, prediction_rows = [], [], [], []
fitted_class, fitted_reg = {}, {}
for name, estimator, scale in classifiers:
    model = build_model(estimator, scale=scale)
    # 제공: CV에 전처리 전 X와 Pipeline을 전달합니다. fold 내부에서 전처리를 학습합니다.
    # 5-fold 교차 검증을 수행하여 각 폴드별 F1 스코어 측정
    # - model : 전처리와 모델이 포함된 파이프라인 객체
    # - X_train, y_train : 교차 검증에 사용할 학습 특성과 타겟 데이터
    # - cv=class_folds : 계층별로 분할된 5-fold 교차 검증 인덱스 리스트
    # - scoring="f1" : 성능 평가 지표로 F1 Score 사용
    scores = cross_val_score(model, X_train, y_train, cv=class_folds, scoring="f1", n_jobs=1)
    # TODO C07a: 분류 후보를 train 전체로 학습하고 valid를 예측하세요.
    # 전체 학습 데이터(X_train, y_train)를 사용하여 파이프라인(전처리 + 모델) 학습
    model.fit(X_train, y_train)
    # pass
    # 학습된 모델을 이용해 검증 세트(X_valid)의 실패 여부 예측
    prediction = model.predict(X_valid)
    # TODO 끝
    # 모델이 예측한 클래스 목록 중 '요청 실패(1)'에 해당하는 인덱스를 찾아, 검증 데이터셋에 대해 요청이 실패할 확률(예측 확률 값) 추출
    probability = model.predict_proba(X_valid)[:, list(model.classes_).index(1)]
    class_rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                       "scale": "standard" if scale else "none",
                       "train_F1": classification_metrics(y_train, model.predict(X_train))["F1"],
                       **classification_metrics(y_valid, prediction)})
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"task": "classification", "model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "metric": "F1", "score": value})
    for request_id, actual, predicted, prob in zip(X_valid.index, y_valid, prediction, probability):
        prediction_rows.append({"request_id": request_id, "task": "classification", "model": name,
                                "actual": float(actual), "prediction": float(predicted), "probability_failed": float(prob)})
    fitted_class[name] = model
for name, estimator, scale in regressors:
    model = build_model(estimator, scale=scale)
    # 제공: 회귀는 fold별 RMSE를 평균합니다. 분류 F1과 합쳐 순위를 만들지 않습니다.
    # 회귀 모델에 대한 5-fold 교차 검증 수행 (음수 RMSE를 반환받은 뒤 양수로 변환)
    scores = -cross_val_score(model, X_train, train["elapsed_ms"], cv=reg_folds,
                              scoring="neg_root_mean_squared_error", n_jobs=1)
    # TODO C07b: 회귀 후보를 실행시간 정답으로 학습하고 valid를 예측하세요.
    # 전체 학습 데이터와 실행시간(elapsed_ms)을 이용해 회귀 파이프라인 학습
    model.fit(X_train, train["elapsed_ms"])
    # pass
    # 학습된 모델을 통해 검증 데이터(X_valid)의 실행시간 예측값 산출
    prediction = model.predict(X_valid)
    # TODO 끝
    reg_rows.append({"model": name, "CV_RMSE_ms": scores.mean(), "CV_std_ms": scores.std(),
                     "scale": "standard" if scale else "none", **regression_metrics(valid["elapsed_ms"], prediction)})
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"task": "regression", "model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "metric": "RMSE_ms", "score": value})
    for request_id, actual, predicted in zip(X_valid.index, valid["elapsed_ms"], prediction):
        prediction_rows.append({"request_id": request_id, "task": "regression", "model": name,
                                "actual": float(actual), "prediction": float(predicted), "probability_failed": None})
    fitted_reg[name] = model

# 제공: 모델 선택은 train 내부 CV로만 합니다. valid 결과는 선택된 조건의 별도 진단으로 보고합니다.
classification = pd.DataFrame(class_rows)
regression = pd.DataFrame(reg_rows)
chosen_class = classification.loc[classification["CV_F1"].idxmax(), "model"]
chosen_reg = regression.loc[regression["CV_RMSE_ms"].idxmin(), "model"]
classification.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
regression.to_csv(OUT / "regression_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(prediction_rows).to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 제공: 중요도는 선택 모델과 별개로 Forest를 진단합니다. 이름에 모델과 계산 방법을 남깁니다.
forest = fitted_class["Forest"]
mdi = pd.DataFrame({"feature": forest["pre"].get_feature_names_out(),
                    "importance": forest["model"].feature_importances_})
# 랜덤 포레스트 모델에 대해 검증 데이터셋 기준 순열 중요도(Permutation Importance)를 10회 반복 측정
permutation = permutation_importance(forest, X_valid, y_valid, scoring="f1", n_repeats=10, random_state=42, n_jobs=1)
permutation_table = pd.DataFrame({"feature": FEATURES, "importance": permutation.importances_mean,
                                  "repeat_std": permutation.importances_std})
# MDI의 학습8열과 순열의 원본4열은 의미·단위가 다릅니다. method와 data를 함께 저장합니다.
importance = pd.concat([mdi.assign(model="Forest", method="MDI", data="train60", repeat_std=float("nan")),
                        permutation_table.assign(model="Forest", method="permutation_F1", data="valid20")], ignore_index=True)
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

# 제공: 각 fold의 학습/평가 ID 기록. 보관한 test ID는 여기에 들어가지 않습니다.
fold_rows = []
for task, folds in [("classification", class_folds), ("regression", reg_folds)]:
    for number, (fit_positions, eval_positions) in enumerate(folds, start=1):
        for role, positions in [("fit", fit_positions), ("evaluate", eval_positions)]:
            for position in positions:
                fold_rows.append({"task": task, "fold": number, "role": role, "request_id": X_train.index[position]})
pd.DataFrame(fold_rows).to_csv(OUT / "fold_manifest.csv", index=False, encoding="utf-8-sig")
selection = {"classification": chosen_class, "regression": chosen_reg,
             "rule": "classification: max train-CV mean F1; regression: min train-CV mean RMSE; ties: listed order",
             "classification_candidates": [x[0] for x in classifiers], "regression_candidates": [x[0] for x in regressors],
             "features": FEATURES, "seed": 42, "folds": 5, "test_evaluated": False,
             "model_parameters": {name: {key: value for key, value in estimator.get_params().items()
                                       if key in ["strategy", "C", "max_iter", "n_neighbors", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "objective", "eval_metric", "tree_method", "device", "random_state", "n_jobs"]}
                                  for name, estimator, _ in classifiers},
             "regression_model_parameters": {name: {key: value for key, value in estimator.get_params().items()
                                                  if key in ["strategy", "n_neighbors", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "objective", "eval_metric", "tree_method", "device", "random_state", "n_jobs"]}
                                             for name, estimator, _ in regressors}}
(OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
environment = {"python": platform.python_version(), **{name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]}}
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
# 직접 생성한 파일만 같은 환경에서 restore.py로 복원합니다. 외부 joblib/pickle 파일은 읽지 않습니다.
# 최적의 성능을 낸 분류 및 회귀 모델 객체를 joblib을 이용해 하나의 파일(.joblib)로 직렬화 저장
joblib.dump({"classification": fitted_class[chosen_class], "regression": fitted_reg[chosen_reg]}, OUT / "selected_models.joblib")
print("분류 비교")
print(classification.round(4).to_string(index=False))
print("회귀 비교")
print(regression.round(3).to_string(index=False))
print("CV로 선택한 분류:", chosen_class, "/ 회귀:", chosen_reg)
print("중요도: Forest 진단용 / test20은 학습·변환·예측·평가하지 않았습니다.")
print("저장 완료: outputs/의 비교표·CV·중요도·선택·예측·환경·모델")
