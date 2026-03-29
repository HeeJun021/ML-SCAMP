import pandas as pd
import numpy as np
import ta
import os

# 저장 경로 설정
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ── 1. 기술적 지표 추가 함수 ─────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터에 기술적 지표를 추가한다.

    추가되는 지표
    -------------
    - RSI        : 과매수/과매도 판단 (0~100)
    - MACD       : 추세 방향 및 강도
    - MACD Signal: MACD의 이동평균 (교차 신호)
    - BB_upper   : 볼린저밴드 상단
    - BB_lower   : 볼린저밴드 하단
    - BB_width   : 밴드 폭 (변동성 측정)
    - Volume_ratio: 거래량 변화율 (오늘/20일 평균)
    - Return_1d  : 전일 대비 수익률
    - Return_5d  : 5일 수익률
    """
    df = df.copy()

    # RSI (14일 기준)
    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=14
    ).rsi()

    # MACD
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # 볼린저밴드 (20일 기준)
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["Close"]

    # 거래량 변화율 (오늘 거래량 / 20일 평균 거래량)
    df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # 수익률
    df["Return_1d"] = df["Close"].pct_change(1)   # 전일 대비
    df["Return_5d"] = df["Close"].pct_change(5)   # 5일 대비

    return df


# ── 2. 타겟 컬럼 추가 함수 ───────────────────────────────────
def add_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    다음날 종가가 오늘보다 높으면 1, 낮으면 0을 타겟으로 추가한다.

    Parameters
    ----------
    threshold : float
        상승 판단 기준 수익률 (기본값 0.0 → 1원이라도 오르면 1)
        예: 0.005 → 0.5% 이상 올라야 1
    """
    df = df.copy()

    # 다음날 종가 수익률 계산
    next_return = df["Close"].pct_change(1).shift(-1)

    # threshold 이상 오르면 1, 아니면 0
    df["Target"] = (next_return > threshold).astype(int)

    return df


# ── 3. 결측값 제거 함수 ──────────────────────────────────────
def remove_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    지표 계산 초반부에 생기는 NaN 행을 제거한다.
    (RSI는 14일, 볼린저밴드는 20일 이후부터 값이 생김)
    """
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"  결측값 제거: {before}행 → {after}행 ({before - after}행 제거)")
    return df


# ── 4. 단일 종목 전처리 함수 ─────────────────────────────────
def process_stock(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    단일 종목 데이터에 지표, 타겟을 추가하고 저장한다.
    """
    print(f"[{ticker}] 전처리 중...")

    df = add_technical_indicators(df)
    df = add_target(df)
    df = remove_nan(df)

    print(f"[{ticker}] 완료 → {len(df)}행, {df.shape[1]}개 컬럼\n")
    return df


# ── 5. 전체 종목 전처리 및 저장 함수 ────────────────────────
def process_all(stock_data: dict, save_csv: bool = True) -> dict:
    """
    모든 종목에 전처리를 적용하고 processed/ 에 저장한다.

    Parameters
    ----------
    stock_data : dict
        {ticker: DataFrame} 형태 (data_loader.download_all() 결과)

    Returns
    -------
    dict
        {ticker: 전처리된 DataFrame}
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    processed = {}

    for ticker, df in stock_data.items():
        df_processed = process_stock(df, ticker)
        processed[ticker] = df_processed

        if save_csv:
            path = os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv")
            df_processed.to_csv(path)
            print(f"[{ticker}] 저장 완료 → {path}\n")

    print("=" * 40)
    print(f"전체 {len(processed)}개 종목 전처리 완료")
    return processed


# ── 6. 통합 데이터셋 생성 함수 ───────────────────────────────
def build_combined_dataset(processed: dict) -> pd.DataFrame:
    """
    전처리된 모든 종목을 하나의 DataFrame으로 합친다.
    Phase 1 공통 모델 학습에 사용된다.
    """
    dfs = list(processed.values())
    combined = pd.concat(dfs, axis=0).sort_index()

    print(f"통합 데이터셋 완료 → 총 {len(combined)}행, {combined.shape[1]}개 컬럼")
    return combined