import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

# ── 1. Feature 컬럼 정의 ─────────────────────────────────────
# 모델이 학습에 사용할 컬럼 목록
FEATURE_COLS = [
    "RSI",
    "MACD",
    "MACD_signal",
    "BB_width",
    "Volume_ratio",
    "Return_1d",
    "Return_5d"
]

TARGET_COL = "Target"


# ── 2. 학습/테스트 데이터 분리 함수 ─────────────────────────
def split_data(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    시계열 데이터를 학습/테스트로 분리한다.

    ※ 시계열은 반드시 앞부분을 학습, 뒷부분을 테스트로 써야 해요.
       일반 train_test_split은 날짜를 섞어버려서 미래 데이터로
       과거를 예측하는 데이터 누수(Data Leakage)가 생겨요.

    Parameters
    ----------
    test_ratio : float
        테스트 데이터 비율 (기본값 0.2 = 마지막 20%)
    """
    split_idx = int(len(df) * (1 - test_ratio))

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    print(f"학습 데이터: {len(train)}행 ({train.index[0].date()} ~ {train.index[-1].date()})")
    print(f"테스트 데이터: {len(test)}행  ({test.index[0].date()} ~ {test.index[-1].date()})")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    return X_train, X_test, y_train, y_test


# ── 3. 모델 학습 함수 ────────────────────────────────────────
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Random Forest 모델을 학습한다.

    주요 파라미터 설명
    ------------------
    n_estimators  : 결정 트리 개수 (많을수록 안정적, 느려짐)
    max_depth     : 트리 최대 깊이 (너무 깊으면 과적합)
    min_samples_leaf : 리프 노드 최소 샘플 수 (과적합 방지)
    class_weight  : 클래스 불균형 자동 보정
    random_state  : 재현성 보장
    """
    model = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 10,
        min_samples_leaf = 20,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1   # 모든 CPU 코어 사용
    )

    model.fit(X_train, y_train)
    print("모델 학습 완료")
    return model


# ── 4. 모델 평가 함수 ────────────────────────────────────────
def evaluate_model(model: RandomForestClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    학습된 모델의 성능을 평가한다.

    평가 지표
    ---------
    Accuracy : 전체 정확도
    AUC      : 상승/하락 구분 능력 (0.5 = 랜덤, 1.0 = 완벽)
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]  # 상승 확률

    accuracy    = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"정확도 (Accuracy) : {accuracy:.4f}")
    print(f"AUC Score         : {auc:.4f}")
    print(f"\n분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=["하락(0)", "상승(1)"]))

    return {
        "accuracy" : accuracy,
        "auc"      : auc,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob
    }


# ── 5. Feature Importance 추출 함수 ─────────────────────────
def get_feature_importance(model: RandomForestClassifier) -> pd.DataFrame:
    """
    각 Feature가 모델 예측에 얼마나 기여했는지 반환한다.
    가설 검증의 핵심 → RSI, Volume_ratio가 상위권인지 확인
    """
    importance_df = pd.DataFrame({
        "Feature"   : FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    return importance_df