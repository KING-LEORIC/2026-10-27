"""실습 08 심화: main.py가 직접 만든 모델을 새 프로세스에서 복원합니다."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from preprocessing import FEATURES

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
X_valid = raw.loc[valid_ids, FEATURES]
selection = json.loads((HERE / "outputs/selection.json").read_text(encoding="utf-8"))
previous = pd.read_csv(HERE / "outputs/validation_predictions.csv")
# TODO C08: 직접 생성한 outputs/selected_models.joblib을 복원하세요.
# 앞서 main.py에서 저장한 최적 모델 바이너리 파일(.joblib)을 읽어와 딕셔너리로 복원
models = joblib.load(HERE / "outputs/selected_models.joblib")
# TODO 끝
# 제공: 예측 배열뿐 아니라 저장된 ID·순서까지 맞춰 대조합니다.
for task in ["classification", "regression"]:
    model = models[task]
    saved = previous[(previous["task"] == task) & (previous["model"] == selection[task])]
    # 검증 데이터의 요청 ID 순서가 저장된 예측 파일의 ID 순서와 일치하는지 비교
    same_ids = saved["request_id"].tolist() == X_valid.index.tolist()
    # 복원된 모델로 예측한 결과값이 이전에 저장된 예측값과 일치하는지 비교 검증
    same_prediction = np.allclose(model.predict(X_valid), saved["prediction"].to_numpy())
    print(task, "ID 순서 일치:", same_ids, "/ 예측 일치:", same_prediction)
    if task == "classification":
        positive = list(model.classes_).index(1)
        # 분류 모델의 경우, 요청 실패 확률(클래스 1)이 이전에 저장한 확률값과 일치하는지 대조
        same_probability = np.allclose(model.predict_proba(X_valid)[:, positive], saved["probability_failed"].to_numpy())
        print("실패 확률 일치:", same_probability)
print("모델을 다시 학습하지 않았습니다. test도 사용하지 않았습니다.")
