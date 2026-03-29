import yfinance as yf
import pandas as pd
import os

# ── 1. 수집할 종목과 기간 설정 ──────────────────────────────
TICKERS = {
    "NVDA": "NVIDIA",
    "TSM": "TSMC",
    "AMD": "AMD",
    "ASML": "ASML",
    "QCOM": "Qualcomm"
}

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

# ── 2. 저장 경로 설정 ────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ── 3. 단일 종목 다운로드 함수 ───────────────────────────────
def download_stock(ticker: str) -> pd.DataFrame:
    """
    yfinance로 단일 종목 주가 데이터를 다운로드한다.

    Parameters
    ----------
    ticker : str
        종목 티커 (예: "NVDA")

    Returns
    -------
    pd.DataFrame
        OHLCV 데이터 (Open, High, Low, Close, Volume)
    """
    print(f"[{ticker}] 다운로드 중...")

    df = yf.download(
        tickers    = ticker,
        start      = START_DATE,
        end        = END_DATE,
        auto_adjust= True,   # 배당/액면분할 자동 보정
        progress   = False
    )

    # 멀티컬럼 구조 제거 (yfinance 최신 버전 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 티커 컬럼 추가 (나중에 5개 종목 합칠 때 필요)
    df["Ticker"] = ticker

    print(f"[{ticker}] 완료 → {len(df)}행, {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


# ── 4. 전체 종목 다운로드 및 저장 함수 ──────────────────────
def download_all(save_csv: bool = True) -> dict:
    """
    TICKERS에 정의된 모든 종목을 다운로드하고 CSV로 저장한다.

    Parameters
    ----------
    save_csv : bool
        True면 data/raw/ 에 CSV 저장

    Returns
    -------
    dict
        {ticker: DataFrame} 형태의 딕셔너리
    """
    os.makedirs(RAW_DIR, exist_ok=True)  # 폴더 없으면 자동 생성

    stock_data = {}

    for ticker in TICKERS:
        df = download_stock(ticker)
        stock_data[ticker] = df

        if save_csv:
            path = os.path.join(RAW_DIR, f"{ticker}.csv")
            df.to_csv(path)
            print(f"[{ticker}] 저장 완료 → {path}\n")

    print("=" * 40)
    print(f"전체 {len(stock_data)}개 종목 다운로드 완료")
    return stock_data


# ── 5. 저장된 CSV 불러오기 함수 ──────────────────────────────
def load_stock(ticker: str) -> pd.DataFrame:
    """
    data/raw/ 에 저장된 CSV를 불러온다.

    Parameters
    ----------
    ticker : str
        종목 티커 (예: "NVDA")

    Returns
    -------
    pd.DataFrame
    """
    path = os.path.join(RAW_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{ticker}.csv 가 없어요. download_all() 먼저 실행하세요.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


# ── 6. 전체 종목 합치기 함수 ─────────────────────────────────
def load_all_combined() -> pd.DataFrame:
    """
    저장된 모든 종목 CSV를 하나의 DataFrame으로 합친다.
    Phase 1 공통 모델 학습에 사용된다.

    Returns
    -------
    pd.DataFrame
        모든 종목이 합쳐진 통합 데이터셋
    """
    dfs = []

    for ticker in TICKERS:
        df = load_stock(ticker)
        dfs.append(df)

    combined = pd.concat(dfs, axis=0)
    combined = combined.sort_index()  # 날짜 순 정렬

    print(f"통합 데이터셋 완료 → 총 {len(combined)}행")
    return combined
