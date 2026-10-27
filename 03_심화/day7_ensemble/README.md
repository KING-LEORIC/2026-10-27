# 실습 07~08 · 앙상블 비교 프로젝트

**목표:** 같은 자료·분할에서 기존 모델과 Tree·Forest·XGB를 비교하고, 선택 근거를 설명합니다. 완성된 제공 모듈을 재사용하므로 직접 작성하는 부분은 main.py의 두 블록과 restore.py 한 곳입니다.

```text
day7_ensemble/
├─ README.md
├─ main.py                 # 제공 실행 흐름 + 학습·예측 TODO 2곳
├─ preprocessing.py        # 제공: 열별 전처리와 Pipeline 생성
├─ evaluate.py             # 제공: 분류·회귀 지표 계산
├─ restore.py              # 복원 TODO 1곳
├─ requirements-lock.txt
├─ data/                   # 원본100건 + 고정 분할표
├─ expected/               # 비교용 실제 정답 결과
├─ outputs/                # 내가 생성하는 결과, 배포 시 비어 있음
└─ reports/ensemble_report.md
```

이 폴더를 VS Code로 엽니다. 단독 복사하여 사용해도 동작합니다. 설치가 필요할 때만 여기서 `python -m pip install -r requirements-lock.txt`를 실행합니다. `data`와 `expected`를 수정하지 않습니다.

## 고정한 비교 조건

입력은 input_chars·allowed_tools·task_type·priority 4열입니다. ID와 정답·실행 후 정보는 입력에 넣지 않습니다. train60·valid20·test20의 ID와 순서를 유지합니다. 분류는 train의 StratifiedKFold5, 회귀는 같은 train의 KFold5를 사용합니다. 후보끼리는 같은 문제의 같은 fold를 공유합니다.

분류 후보는 Dummy·Logistic·KNN3·Tree·Forest·XGB, 회귀는 Dummy·Linear·KNN15·Tree·Forest·XGB입니다. 선형·KNN에는 전날과 같은 숫자 표준화, 트리 계열에는 숫자 스케일링 생략을 적용합니다. 이 차이는 실험표 scale 열에 남깁니다.

**선택 규칙을 결과를 보기 전에 고정:** 분류는 train CV 평균 F1 최대, 회귀는 CV 평균 RMSE 최소, 동점이면 코드의 후보 순서가 먼저인 모델입니다. 바깥 valid는 별도 진단에 사용합니다. 같은 valid를 이전 수업부터 반복 관찰했으므로 독립적인 최종 성능이라고 주장하지 않습니다. test는 최종 평가 수업까지 보관합니다.

## 제공 함수 읽기

| 함수 | 입력 | 반환·사용 시점 |
|---|---|---|
| build_model(estimator, scale=False) | 아직 학습 안 한 모델과 스케일링 여부 | 아직 학습 안 한 새 Pipeline |
| classification_metrics(actual, prediction) | 실제·예측 클래스 | 실패1의 지표·TN·FP·FN·TP Dictionary |
| regression_metrics(actual, prediction) | 실제·예측 실행시간 | MAE_ms·RMSE_ms·R2 Dictionary |

함수를 호출했다고 자동으로 모든 자료를 학습하는 것이 아닙니다. fit에 어떤 자료를 넣었는지가 중요합니다.

## 실습 07 · 모듈형 앙상블 비교 프로젝트

**목표:** 전날 모델과 앙상블의 두 비교표를 같은 입력·분할에서 생성합니다.

**열 파일:** `main.py`  
**완성할 위치:** `C07a`, `C07b`

- **C07a**: 분류 후보를 train 전체로 학습하고 valid를 예측하세요.
- **C07b**: 회귀 후보를 실행시간 정답으로 학습하고 valid를 예측하세요.

**힌트:** 제공 모듈은 수정하지 않고, main.py의 분류·회귀 fit·predict만 먼저 완성합니다.

```powershell
python main.py
```

**실제 정답 실행으로 확인한 출력**

```text
분류 비교
   model  CV_F1  CV_std    scale  train_F1  Accuracy  Precision  Recall     F1  TN  FP  FN  TP
   Dummy 0.0000  0.0000     none    0.0000      0.70     0.0000  0.0000 0.0000  14   0   6   0
Logistic 0.5219  0.2092 standard    0.6207      0.60     0.3333  0.3333 0.3333  10   4   4   2
    KNN3 0.3356  0.1764 standard    0.5714      0.75     0.6000  0.5000 0.5455  12   2   3   3
    Tree 0.5424  0.1236     none    0.7097      0.60     0.3333  0.3333 0.3333  10   4   4   2
  Forest 0.2800  0.2544     none    0.8000      0.65     0.3333  0.1667 0.2222  12   2   5   1
     XGB 0.5000  0.1229     none    0.7742      0.55     0.2000  0.1667 0.1818  10   4   5   1
회귀 비교
 model  CV_RMSE_ms  CV_std_ms    scale  MAE_ms  RMSE_ms     R2
 Dummy     606.184    100.844     none 541.260  658.745 -0.162
Linear     639.832     57.721 standard 551.797  639.996 -0.097
 KNN15     598.850     91.213 standard 530.063  641.788 -0.103
  Tree     599.635     96.116     none 632.028  719.001 -0.384
Forest     624.011     96.974     none 600.337  690.368 -0.276
   XGB     655.793     92.170     none 609.341  700.020 -0.312
CV로 선택한 분류: Tree / 회귀: KNN15
중요도: Forest 진단용 / test20은 학습·변환·예측·평가하지 않았습니다.
저장 완료: outputs/의 비교표·CV·중요도·선택·예측·환경·모델
```

**생성 파일:** `outputs/classification_comparison.csv`, `outputs/regression_comparison.csv`, `outputs/cv_scores.csv`, `outputs/validation_predictions.csv`, `outputs/fold_manifest.csv`, `outputs/selection.json`, `outputs/environment.json`, `outputs/feature_importance.csv`

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- CV와 valid가 다른 모델을 높게 평가하면 어떻게 기록하나요?
- 순열 중요도가 낮은 특성을 여기서 제거하고 같은 valid를 다시 사용하면 어떤 한계가 있나요?
- 선택 모델과 중요도 계산 모델은 같은가요?

## 실습 08 · 복원과 최종 설명

**목표:** main.py의 결과를 새 프로세스에서 그대로 재현합니다.

**열 파일:** `restore.py`  
**완성할 위치:** `C08`

- **C08**: 직접 생성한 outputs/selected_models.joblib을 복원하세요.

**힌트:** 먼저 main.py로 파일을 생성한 뒤 python restore.py를 실행합니다.

```powershell
python restore.py
```

먼저 같은 폴더에서 `python main.py`를 실행하여 모델을 직접 생성합니다.

**실제 정답 실행으로 확인한 출력**

```text
classification ID 순서 일치: True / 예측 일치: True
실패 확률 일치: True
regression ID 순서 일치: True / 예측 일치: True
모델을 다시 학습하지 않았습니다. test도 사용하지 않았습니다.
```

**생성 파일:** 없음. 콘솔 출력과 설명을 확인합니다.

**완료 기준:** TODO가 실제 계산·모델 호출에 사용되고, 위 출력 및 단위·행 수가 일치하며 다음 질문에 답할 수 있어야 합니다. 반올림 끝자리 차이만 있다면 검증 버전부터 확인합니다.

- 새 모델을 fit한 것과 복원한 것은 어떤 차이가 있나요?
- 복원 결과가 일치하면 새로운 데이터에서도 성능이 보장될까요?

## 결과 파일 읽기

| 파일 | 내용·확인점 |
|---|---|
| classification_comparison.csv | 6후보, train F1, CV 평균·편차, valid 지표. F1·FP·FN 등의 일반 지표 열은 valid20 기준 |
| regression_comparison.csv | 6후보, CV RMSE 평균·편차, valid MAE·RMSE(ms)·R² |
| cv_scores.csv | 12후보×5fold=60행, 학습48·평가12, 분류 F1 또는 회귀 RMSE |
| fold_manifest.csv | task별 각 fold의 학습·평가 request_id 기록. 서로 겹치지 않는지 확인 |
| validation_predictions.csv | 6분류+6회귀×20건=240행. task·model·request_id로 구분 |
| feature_importance.csv | Forest MDI8행 + 원본열 순열4행. method·data가 다르므로 값을 같은 척도로 순위 비교하지 않음 |
| selection.json | 선택 모델·후보·주 지표·설정·동점 규칙·test 미평가 기록 |
| environment.json | 실제 실행 환경 버전 |
| selected_models.joblib | 직접 생성한 분류·회귀 선택 Pipeline. 배포물에는 없고 main.py 실행 시 생성 |

`CV_std`는 fold 점수들의 모집단 표준편차(ddof=0)이며 신뢰구간이 아닙니다. 순열 중요도의 repeat_std도 순열10회 반복의 편차일 뿐 신뢰구간이 아닙니다. 회귀 CV_RMSE는 fold별 RMSE의 평균으로, 합친 OOF 예측에서 계산한 RMSE와 다를 수 있습니다.

## 8교시 제출과 설명

main.py를 재실행하고, restore.py로 같은 ID·예측·확률이 나오는지 확인합니다. **[reports/ensemble_report.md](reports/ensemble_report.md)**에 근거3개, 오류 사례, 중요도 해석 한계, CV/valid/test의 역할을 작성합니다. 제공된 expected 수치를 복사하는 대신 직접 생성한 outputs에서 근거를 가져옵니다.

짝이 README만 보고 새 폴더에서 실행하도록 합니다. 코드를 변경했다면 그 변경과 재현 결과를 보고서에 남깁니다. 모델 성능이 높다는 이유만으로 더 높은 점수를 받지는 않습니다. 별도 실패 입력 테스트를 작성할 필요는 없습니다.

Git을 이미 배웠다면 이 프로젝트 폴더에서 `git init`, `git add README.md main.py preprocessing.py evaluate.py restore.py data reports .gitignore requirements-lock.txt`, `git commit -m "Add ensemble comparison"`으로 기록합니다. 이미 저장소가 있으면 init은 생략합니다. 생성한 모델 바이너리나 개인정보·비밀키는 커밋하지 않습니다. 커밋을 위해 시스템의 다른 저장소나 설정을 수정하지 않습니다.
