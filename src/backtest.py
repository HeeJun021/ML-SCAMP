import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "reports")


# ── 1. 매수 신호 생성 함수 ───────────────────────────────────
def generate_signals(test_df: pd.DataFrame,
                     y_prob: np.ndarray,
                     threshold: float = 0.6) -> pd.DataFrame:
    """
    모델의 상승 확률이 threshold 이상인 날만 매수 신호를 생성한다.

    Parameters
    ----------
    test_df   : 테스트 데이터 DataFrame
    y_prob    : 모델이 예측한 상승 확률 배열
    threshold : 매수 기준 확률 (기본값 0.6 = 60% 이상일 때만 매수)
    """
    df = test_df.copy()
    df["Prob"]   = y_prob           # 상승 확률
    df["Signal"] = (df["Prob"] >= threshold).astype(int)  # 1=매수, 0=관망

    print(f"전체 {len(df)}일 중 매수 신호: {df['Signal'].sum()}일 ({df['Signal'].mean()*100:.1f}%)")
    return df


# ── 2. 수익률 계산 함수 ──────────────────────────────────────
def calculate_returns(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    매수 신호가 있는 날의 다음날 수익률을 계산한다.

    전략 설명
    ---------
    - Signal = 1 인 날 종가에 매수
    - 다음날 종가에 매도
    - 수수료는 미포함 (단순 전략 검증 목적)
    """
    df = signal_df.copy()

    # 다음날 수익률
    df["Next_return"] = df["Close"].pct_change(1).shift(-1)

    # 전략 수익률 (신호 있는 날만 수익 반영)
    df["Strategy_return"] = df["Signal"] * df["Next_return"]

    # 매수 후 보유(Buy & Hold) 수익률 (비교 기준)
    df["BuyHold_return"] = df["Next_return"]

    # 누적 수익률
    df["Strategy_cum"] = (1 + df["Strategy_return"]).cumprod()
    df["BuyHold_cum"]  = (1 + df["BuyHold_return"]).cumprod()

    return df


# ── 3. MDD 계산 함수 ─────────────────────────────────────────
def calculate_mdd(cumulative_returns: pd.Series) -> float:
    """
    MDD(Maximum DrawDown, 최대 낙폭)를 계산한다.

    MDD 설명
    --------
    고점 대비 최대 하락폭.
    예: 고점 100 → 최저 70 → MDD = -30%
    낮을수록 하락장에서 방어를 잘 한 거예요.
    """
    peak = cumulative_returns.cummax()        # 지금까지의 최고점
    drawdown = (cumulative_returns - peak) / peak  # 고점 대비 하락률
    mdd = drawdown.min()                      # 최대 낙폭
    return mdd


# ── 4. 성과 요약 함수 ────────────────────────────────────────
def summarize_performance(result_df: pd.DataFrame) -> dict:
    """
    백테스팅 핵심 성과 지표를 계산하고 출력한다.

    지표 설명
    ---------
    총 수익률  : 전략 시작~끝 누적 수익률
    연간 수익률: 총 수익률을 연 단위로 환산
    MDD       : 최대 낙폭 (리스크 지표)
    샤프 지수  : 수익 대비 리스크 비율 (높을수록 좋음)
    승률      : 매수 신호 중 실제로 오른 비율
    """
    df = result_df.dropna()

    # 총 수익률
    total_return_strategy = df["Strategy_cum"].iloc[-1] - 1
    total_return_buyhold  = df["BuyHold_cum"].iloc[-1] - 1

    # 연간 수익률 (거래일 기준 252일)
    n_years = len(df) / 252
    annual_return = (1 + total_return_strategy) ** (1 / n_years) - 1

    # MDD
    mdd_strategy = calculate_mdd(df["Strategy_cum"])
    mdd_buyhold  = calculate_mdd(df["BuyHold_cum"])

    # 샤프 지수 (무위험 수익률 0% 가정)
    sharpe = (df["Strategy_return"].mean() /
              df["Strategy_return"].std()) * np.sqrt(252)

    # 승률 (매수 신호 중 실제 상승 비율)
    signal_days = df[df["Signal"] == 1]
    win_rate = (signal_days["Next_return"] > 0).mean()

    print(f"\n{'='*40}")
    print(f"백테스팅 성과 요약")
    print(f"{'='*40}")
    print(f"전략 총 수익률  : {total_return_strategy*100:.2f}%")
    print(f"Buy&Hold 수익률 : {total_return_buyhold*100:.2f}%")
    print(f"연간 수익률     : {annual_return*100:.2f}%")
    print(f"{'─'*40}")
    print(f"전략 MDD        : {mdd_strategy*100:.2f}%")
    print(f"Buy&Hold MDD    : {mdd_buyhold*100:.2f}%")
    print(f"{'─'*40}")
    print(f"샤프 지수       : {sharpe:.4f}")
    print(f"승률            : {win_rate*100:.2f}%")
    print(f"매수 신호 횟수  : {len(signal_days)}회")
    print(f"{'='*40}")
    
    return {
        "total_return"  : total_return_strategy,
        "annual_return" : annual_return,
        "mdd"           : mdd_strategy,
        "sharpe"        : sharpe,
        "win_rate"      : win_rate
    }


# ── 5. 시각화 함수 ───────────────────────────────────────────
def plot_cumulative_returns(result_df: pd.DataFrame,
                            save: bool = True) -> None:
    """
    전략 vs Buy&Hold 누적 수익률 그래프를 그린다.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = result_df.dropna()

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Strategy_cum"],
             label="ML Strategy", color="blue", linewidth=1.5)
    plt.plot(df.index, df["BuyHold_cum"],
             label="Buy & Hold",  color="gray", linewidth=1.5, linestyle="--")

    plt.title("누적 수익률 비교: ML Strategy vs Buy & Hold")
    plt.xlabel("날짜")
    plt.ylabel("누적 수익률")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "cumulative_returns.png")
        plt.savefig(path, dpi=150)
        print(f"그래프 저장 완료 → {path}")

    plt.show()