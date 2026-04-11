# 🔬 ML-SCAMP
### Semiconductor Common Analysis & ML-based Prediction

> 머신러닝 기반 반도체 주식 상승 확률 예측 및 투자 전략 백테스팅 분석

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 주제 | 머신러닝 기반 반도체 주식 상승 확률 예측 |
| 핵심 가치 | 단순 가격 예측(Regression)이 아닌 통계적 상승 확률(Classification) 기반 의사결정 시스템 |
| 타겟 섹터 | AI 반도체 (NVDA, TSM, AMD, ASML, QCOM) |
| 분석 기간 | 2018.01 ~ 2025.12 (약 8년, 총 9,885행) |
| 개발 환경 | Python 3.11, PyTorch, scikit-learn, yfinance |

---

## 연구 가설

> "반도체 산업은 특유의 수요 사이클과 기술 이벤트에 민감하며,
> RSI·거래량 변화율 등 기술적 지표가 주가 반등의 유의미한 신호로 작용할 것이다."

---

## 프로젝트 구조
```
ML-SCAMP/
│
├── data/
│   ├── raw/               # yfinance 원본 데이터
│   └── processed/         # 기술적 지표 추가된 가공 데이터
│
├── notebooks/
│   ├── 01_data_eda.ipynb          # 데이터 탐색 및 환경 확인
│   ├── 02_phase1_baseline.ipynb   # RF 공통 모델 + 백테스팅 + 실시간 예측
│   ├── 03_phase2a_pca.ipynb       # PCA 노이즈 제거 실험
│   ├── 04_phase2b_lstm.ipynb      # LSTM 딥러닝 실험
│   └── 05_generalization.ipynb    # 빅테크 섹터 일반화 테스트
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # yfinance 데이터 수집
│   ├── features.py        # 기술적 지표 계산 + 타겟 생성
│   ├── model.py           # Random Forest 학습 + 평가
│   ├── backtest.py        # 백테스팅 + MDD 계산
│   └── predictor.py       # 실시간 매수 신호 예측
│
├── results/
│   ├── figures/           # 시각화 그래프
│   ├── reports/           # quantstats HTML 리포트
│   └── rf_model.pkl       # 학습된 모델
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 단계별 수행 전략

### Phase 1 — Baseline: 공통 모델 구축 및 가설 검증
5개 종목 통합 데이터셋 구축
↓
Random Forest 상승 확률 학습
↓
Feature Importance로 가설 검증
↓
임계치(≥0.6) 기반 백테스팅

### Phase 2A — PCA 기반 노이즈 제거
Feature 간 상관관계 분석
(MACD ↔ MACD_signal 상관계수 0.95 발견)
↓
7개 Feature → 4개 주성분 압축
↓
Phase 1과 성능 비교

### Phase 2B — LSTM 딥러닝 시계열 학습
20일 시퀀스 데이터 구성
↓
PyTorch + RTX 4070 GPU 학습
↓
Random Forest vs LSTM 성능 비교

### Phase 3 — 일반화 테스트
반도체 모델을 빅테크 섹터에 적용
↓
섹터간 성능 차이 분석
↓
반도체 특화 모델 유효성 검증

---

## 핵심 결과

### 모델 성능 비교
| 모델 | Accuracy | AUC | 순위 |
|------|---------|-----|------|
| **Phase 1 (RF, 7 Features)** | **0.5225** | **0.5369** |
| Phase 2B (LSTM, 20일 시퀀스) | 0.52 | 0.5223 |
| Phase 2A (RF + PCA, 4 주성분) | 0.5114 | 0.5186 |

### 백테스팅 성과 (Phase 1 기준)
| 종목 | 총수익률 | MDD | 샤프지수 | 승률 |
|------|---------|-----|---------|------|
| ASML | +58.2% | -9.75% | 1.78 | 75% |
| QCOM | +43.5% | -6.55% | 1.44 | 74% |
| TSM  | +5.5%  | -3.42% | 0.56 | 75% |
| NVDA | +2.2%  | -6.51% | 0.20 | 55% |
| AMD  | -21.7% | -28.57% | -1.16 | 29% |

### 가설 검증 결과
| 가설 | Feature | 중요도 | 결과 |
|------|---------|--------|------|
| RSI가 유의미한 신호 | RSI | 0.137 (4위) |
| 거래량이 유의미한 신호 | Volume_ratio | 0.154 (2위) |
| RSI 30 이하 → 상승 확률 높음 | RSI 0~30 구간 | 실제 상승 55.6% |

### 일반화 테스트
| 시나리오 | AUC | 결과 |
|---------|-----|------|
| A. 반도체 → 반도체 | 0.5368 | 기준 성능 |
| B. 반도체 → 빅테크 | 0.5215 | 소폭 하락 |
| C. 빅테크 → 빅테크 | 0.5148 | B보다 낮음 |

---

## 실시간 예측 사용법

```python
# 오늘의 반도체 종목 매수 신호 확인
from src.predictor import predict_today, load_model

model  = load_model()
result = predict_today(model)
```

---

## 개발 환경 설정

```bash
# 1. 가상환경 생성
conda create -n semiconductor python=3.11
conda activate semiconductor

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. PyTorch GPU 버전 설치 (RTX 4070 기준)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 한계점 및 향후 개선 방향

| 한계 | 원인 | 개선 방향 |
|------|------|---------|
| AUC 0.54 수준 | 기술적 지표만 사용 | 펀더멘털 데이터 추가 |
| 외부 이벤트 반영 불가 | 실적/금리 등 미포함 | NLP 뉴스 감성 분석 결합 |
| AMD 패턴 학습 실패 | 높은 변동성 | 종목 특화 모델 검토 |
| 수수료 미반영 | 단순화 목적 | 실전 적용 시 반영 필요 |
| LSTM 과적합 | 데이터 부족 | 데이터 확장 + Early Stopping |

---

## 참고 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.11 |
| 데이터 수집 | yfinance |
| 데이터 처리 | pandas, numpy |
| 기술적 지표 | ta (Technical Analysis Library) |
| 머신러닝 | scikit-learn (Random Forest, PCA) |
| 딥러닝 | PyTorch (LSTM) |
| 시각화 | matplotlib, seaborn, plotly |
| 환경 관리 | Miniconda |
| GPU | NVIDIA RTX 4070 + CUDA |

---

> **"단순 가격 예측이 아닌 통계적 상승 확률 기반의 의사결정 시스템을 구축하였고,
> 연구 가설을 데이터로 직접 검증 완료하였음."**