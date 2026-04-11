# src/predictor.py

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features    import add_technical_indicators
from src.model       import FEATURE_COLS, train_model, split_data
from src.data_loader import download_all
from src.features    import process_all, build_combined_dataset

# 모델 저장 경로
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

TICKERS = ["NVDA", "TSM", "AMD", "ASML", "QCOM"]
THRESHOLD = 0.6  # 매수 기준 확률


# ── 1. 모델 저장 함수 ────────────────────────────────────────
def save_model(model) -> None:
    """학습된 모델을 pkl 파일로 저장한다."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장 완료 → {MODEL_PATH}")


# ── 2. 모델 로드 함수 ────────────────────────────────────────
def load_model():
    """저장된 모델을 불러온다. 없으면 새로 학습한다."""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("저장된 모델 로드 완료")
        return model
    else:
        print("저장된 모델 없음 → 새로 학습합니다...")
        stock_data = download_all(save_csv=False)
        processed  = process_all(stock_data, save_csv=False)
        combined   = build_combined_dataset(processed)
        X_train, X_test, y_train, y_test = split_data(combined)
        model = train_model(X_train, y_train)
        save_model(model)
        return model


# ── 3. 오늘 데이터 수집 함수 ─────────────────────────────────
def get_today_data(ticker: str) -> pd.DataFrame:
    """
    오늘 기준 최근 60일 데이터를 수집하고
    기술적 지표를 계산한다.
    (지표 계산에 최소 20일 이상 필요하기 때문에 60일 수집)
    """
    end   = datetime.today()
    start = end - timedelta(days=60)

    df = yf.download(
        tickers     = ticker,
        start       = start.strftime("%Y-%m-%d"),
        end         = end.strftime("%Y-%m-%d"),
        auto_adjust = True,
        progress    = False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Ticker"] = ticker
    df = add_technical_indicators(df)
    df = df.dropna()

    return df


# ── 4. 단일 종목 예측 함수 ───────────────────────────────────
def predict_ticker(model, ticker: str) -> dict:
    """
    단일 종목의 오늘 데이터로 내일 상승 확률을 예측한다.
    """
    df = get_today_data(ticker)

    if len(df) == 0:
        return {"ticker": ticker, "prob": None, "signal": None, "error": "데이터 없음"}

    # 가장 최근 행 (오늘) 의 Feature 추출
    today_features = df[FEATURE_COLS].iloc[-1:]
    today_date     = df.index[-1].date()

    # 예측
    prob   = model.predict_proba(today_features)[0][1]
    signal = "매수 추천" if prob >= THRESHOLD else "관망"

    return {
        "ticker"    : ticker,
        "date"      : today_date,
        "prob"      : prob,
        "signal"    : signal,
        "close"     : round(df["Close"].iloc[-1], 2),
        "rsi"       : round(df["RSI"].iloc[-1], 2),
        "volume_ratio": round(df["Volume_ratio"].iloc[-1], 2)
    }


# ── 5. 전체 종목 예측 함수 ───────────────────────────────────
def predict_today(model=None) -> pd.DataFrame:
    """
    전체 반도체 종목의 오늘 매수 신호를 출력한다.
    """
    if model is None:
        model = load_model()

    print(f"\n{'='*55}")
    print(f"반도체 섹터 매수 신호 분석")
    print(f"{'='*55}")

    results = []

    for ticker in TICKERS:
        result = predict_ticker(model, ticker)
        results.append(result)

    df_result = pd.DataFrame(results)
    df_result["prob_pct"] = (df_result["prob"] * 100).round(1)

    # 출력
    print(f"\n{'종목':<6} {'현재가':>8} {'RSI':>6} {'거래량비율':>8} {'상승확률':>8} {'신호'}")
    print("-" * 55)

    for _, row in df_result.iterrows():
        print(f"{row['ticker']:<6} "
              f"${row['close']:>8.2f} "
              f"{row['rsi']:>6.1f} "
              f"{row['volume_ratio']:>8.2f} "
              f"{row['prob_pct']:>7.1f}% "
              f"  {row['signal']}")

    print(f"\n{'='*55}")
    print(f"기준: 상승 확률 ≥ {THRESHOLD*100:.0f}% → 매수 추천")
    print(f"분석 날짜: {df_result['date'].iloc[0]}")
    print(f"{'='*55}\n")

    return df_result