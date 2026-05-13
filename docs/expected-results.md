# CausalStock 재현 검증 기준 — 논문 보고 수치

> 본 문서는 NeurIPS 2024 CausalStock 논문의 모든 정량 결과를 표 형태로 정리한 것이다. 재현 구현(Paper α)의 sanity check 기준으로 사용한다.
>
> **수치 출처**: 논문 본문 Table 1, Table 2, Table 4 (Hyper-param sensitivity), Table 5 (Spearman corr.), Figure 4 (Investment simulation).
> **표기**: `ACC ± std` (소수점 4자리, 본문 그대로).
> **재현 허용 오차 기준 제안**: ACC ±0.5%p, MCC ±0.015 (논문 standard deviation 범위와 비슷). 그 이상 차이 나면 구현 오류 의심.

---

## Table 1 — Main Results

### 1a. News-driven Multi-stock Movement Prediction Task

데이터셋: ACL18 (US), CMIN-US (US), CMIN-CN (CN). std는 10 runs 기준.

| Model | ACL18 ACC | ACL18 MCC | CMIN-US ACC | CMIN-US MCC | CMIN-CN ACC | CMIN-CN MCC |
|---|---|---|---|---|---|---|
| HAN | 57.64 ± 0.0040 | 0.0518 ± 0.0050 | 53.72 ± 0.0020 | 0.0103 ± 0.0015 | 53.59 ± 0.0037 | 0.0159 ± 0.0026 |
| StockNet | 58.23 ± 0.0030 | 0.0808 ± 0.0071 | 52.46 ± 0.0041 | 0.0220 ± 0.0025 | 54.53 ± 0.0062 | 0.0450 ± 0.0043 |
| PEN | 59.89 ± 0.0090 | 0.1556 ± 0.0018 | 53.20 ± 0.0051 | 0.0267 ± 0.0023 | 54.83 ± 0.0086 | 0.0857 ± 0.0065 |
| CMIN | 62.69 ± 0.0029 | 0.2090 ± 0.0016 | 53.43 ± 0.0085 | 0.0460 ± 0.0055 | 55.28 ± 0.0094 | 0.1110 ± 0.0990 |
| **CausalStock** | **63.42 ± 0.0039** | **0.2172 ± 0.0017** | **54.64 ± 0.0083** | **0.0481 ± 0.0057** | **56.19 ± 0.0084** | **0.1417 ± 0.0813** |

**우위 폭 vs SOTA (CMIN)**:
- ACL18: +0.73%p ACC, +0.0082 MCC
- CMIN-US: +1.21%p ACC, +0.0021 MCC
- CMIN-CN: +0.91%p ACC, +0.0307 MCC

### 1b. Multi-stock Movement Prediction Task (price-only, 뉴스 없음)

데이터셋: KDD17 (US), NI225 (JP), FTSE100 (UK). std는 5 runs 기준.

| Model | KDD17 ACC | KDD17 MCC | NI225 ACC | NI225 MCC | FTSE100 ACC | FTSE100 MCC |
|---|---|---|---|---|---|---|
| LSTM | 51.18 ± 0.0066 | 0.0187 ± 0.0110 | 50.79 ± 0.0079 | 0.0148 ± 0.0162 | 50.96 ± 0.0065 | 0.0187 ± 0.0129 |
| ALSTM | 51.66 ± 0.0041 | 0.0316 ± 0.0119 | 50.60 ± 0.0066 | 0.0125 ± 0.0139 | 51.06 ± 0.0038 | 0.0231 ± 0.0077 |
| StockNet | 51.93 ± 0.0001 | 0.0335 ± 0.0050 | 50.15 ± 0.0054 | 0.0050 ± 0.0118 | 50.36 ± 0.0095 | 0.0134 ± 0.0135 |
| Adv-ALSTM | 51.69 ± 0.0058 | 0.0333 ± 0.0137 | 51.60 ± 0.0103 | 0.0340 ± 0.0201 | 50.66 ± 0.0067 | 0.0155 ± 0.0140 |
| DTML | 53.53 ± 0.0075 | 0.0733 ± 0.0195 | 52.76 ± 0.0103 | 0.0626 ± 0.0230 | 52.08 ± 0.0121 | 0.0502 ± 0.0214 |
| **CausalStock** | **56.09 ± 0.0069** | **0.1235 ± 0.0189** | **53.01 ± 0.0150** | **0.0640 ± 0.0310** | **52.88 ± 0.0009** | **0.0534 ± 0.0210** |

**우위 폭 vs SOTA (DTML)**:
- KDD17: +2.56%p ACC, +0.0502 MCC (가장 큰 폭)
- NI225: +0.25%p ACC, +0.0014 MCC (가장 작은 폭 — JP 시장 도전적)
- FTSE100: +0.80%p ACC, +0.0032 MCC

> **재현 시 주의**: NI225 / FTSE100은 종목 수가 적고 (51, 24) 데이터 기간이 짧아 std 자체가 큰 편. KDD17이 가장 안정적인 재현 검증 대상.

---

## Table 2 — Ablation Study

데이터셋: ACL18, CMIN-US, CMIN-CN. (KDD17/NI225/FTSE100 ablation은 미보고)

### 2a. Main Framework Variants

| Ablation Variant | ACL18 ACC | ACL18 MCC | CMIN-US ACC | CMIN-US MCC | CMIN-CN ACC | CMIN-CN MCC |
|---|---|---|---|---|---|---|
| CausalStock w/o TCD | 51.08 | 0.0102 | 51.48 | 0.0106 | 51.37 | 0.0102 |
| CausalStock w/o news | 58.10 | 0.1421 | 53.16 | 0.0375 | 54.16 | 0.1264 |
| CausalStock w/o link non-existence | 58.21 | 0.1652 | 52.32 | 0.0241 | 53.96 | 0.0670 |
| CausalStock w/o Lag-dependent TCD | 59.19 | 0.1757 | 52.93 | 0.0312 | 54.97 | 0.1298 |
| CausalStock w/ Variable-dependent TCD | 63.50 | 0.2175 | 54.60 | 0.0479 | 56.25 | 0.1419 |
| **CausalStock (full)** | **63.42** | **0.2172** | **54.64** | **0.0481** | **56.19** | **0.1417** |

**기여도 해석**:
- **TCD가 핵심**: w/o TCD에서 12%p 급락 → causal discovery가 본 모델의 근간.
- **News도 중요**: w/o news에서 5–8%p 하락. 단, news 없이도 53–58%로 baseline 이상 (price-only TCD가 잘 작동).
- **Link non-existence modeling**: 5%p 하락 → 단순 Sigmoid보다 explicit non-existence likelihood가 효과.
- **Lag-dependency**: 1.3–4.2%p 향상 → 본 논문의 핵심 novelty 입증.
- **Variable-dependent TCD**: 미세 개선 가능하나 복잡도 O(L·D²) → O(L·D⁴). D=300인 CMIN-CN에서는 비현실적.

### 2b. Traditional News Encoder Variants (DNE 대체)

| News Encoder | ACL18 ACC | ACL18 MCC | CMIN-US ACC | CMIN-US MCC | CMIN-CN ACC | CMIN-CN MCC |
|---|---|---|---|---|---|---|
| Glove + Bi-GRU | 60.78 | 0.1952 | 53.87 | 0.0467 | 55.13 | 0.1326 |
| BERT (base-multilingual-cased) | 61.74 | 0.2067 | 53.92 | 0.0472 | 55.43 | 0.1352 |
| RoBERTa (base) | 61.81 | 0.2071 | 54.06 | 0.0477 | 55.58 | 0.1364 |
| FinBERT [1] | 61.72 | 0.2062 | 54.01 | 0.0471 | 55.61 | 0.1362 |
| FinGPT (v3.3) | 61.69 | 0.2060 | 54.00 | 0.0470 | 55.60 | 0.1360 |
| Llama (7b-chat-hf) | 62.20 | 0.2130 | 54.40 | 0.0480 | 55.85 | 0.1390 |

### 2c. Denoised News Encoder LLM Variants

| DNE LLM | ACL18 ACC | ACL18 MCC | CMIN-US ACC | CMIN-US MCC | CMIN-CN ACC | CMIN-CN MCC |
|---|---|---|---|---|---|---|
| FinGPT (v3.3) | 61.92 | 0.2105 | 54.30 | 0.0475 | 55.67 | 0.1386 |
| Llama (7b-chat-hf) | 62.82 | 0.2164 | 54.52 | 0.0483 | 55.97 | 0.1406 |
| **GPT-3.5 (default)** | **63.42** | **0.2172** | **54.64** | **0.0481** | **56.19** | **0.1417** |

**핵심 발견**:
- 같은 LLM(FinGPT, Llama)을 두 가지 방식(traditional embedding vs DNE scoring)으로 비교했을 때, **DNE가 일관되게 상회** → discrete-score scheme이 효과적.
- GPT-3.5 > Llama > FinGPT (DNE) → LLM 일반 능력 > finance-specific tuning. 흥미로운 관찰.
- Llama-DNE (62.82) > Llama-traditional (62.20) → 동일 LLM에서 +0.62%p.

---

## Table 4 — Hyper-parameter Sensitivity (ACC만, Appendix C.4)

기준점: `lr = 1e-5, L = 5, λ = 0.01`. 한 번에 한 파라미터만 변경.

### 4a. Learning rate

| lr | 1e-3 | 1e-4 | **1e-5** | 1e-6 |
|---|---|---|---|---|
| ACL18 (with news) | 62.56 | 62.34 | **63.42** | 61.58 |
| KDD17 (w/o news) | 55.45 | 55.69 | **56.09** | 55.13 |

→ 1e-5가 둘 다 최적. 너무 크거나 작으면 1%p 정도 손해.

### 4b. Time lag L

| L | 3 | **5** | 7 | 9 |
|---|---|---|---|---|
| ACL18 (with news) | 61.04 | **63.42** | 63.29 | 63.15 |
| KDD17 (w/o news) | 54.94 | **56.09** | 55.95 | 55.94 |

→ L=5 최적. L=3은 정보 부족, L=7 이상은 plateau (소폭 하락). **L=5는 robust한 default**.

### 4c. Loss weight λ

| λ | 0 | 0.1 | **0.01** | 0.001 |
|---|---|---|---|---|
| ACL18 (with news) | 58.26 | 62.35 | **63.42** | 63.45 |
| KDD17 (w/o news) | 53.19 | 55.57 | **56.09** | 55.45 |

→ **λ=0 (BCE 없음) 시 5%p 급락**: BCE auxiliary loss가 학습 안정화에 결정적.
→ λ=0.001과 0.01 결과 거의 동일 (ACL18에서 63.45 > 63.42로 약간 역전). 실제로 0.001~0.01 사이 사용 가능.

---

## Table 5 — Causal Strength vs Market Cap (Spearman correlation)

각 시장에서 시가총액과 stock causal strength 간 Spearman rank correlation. **모든 데이터셋에서 강한 양의 상관** 입증.

| 통계 | ACL18 (US) | NI225 (JP) | CMIN-CN (CN) | FTSE100 (UK) |
|---|---|---|---|---|
| Spearman corr. | 0.7939 | 0.7212 | 0.6491 | 0.8909 |
| P-value | 0.006 | 0.0185 | 0.0036 | 0.0005 |

**해석**: 시가총액 큰 종목 = causal influence 큰 종목. 경제적 직관과 일치. CausalStock이 발견한 causal graph가 임의적이지 않다는 외적 타당성 증거.

> **재현 시 sanity check**: 한국 시장(KOSPI 200) 적용 시 동일한 trend 확인 필요. KOSPI는 삼성전자 시총 비중이 절대적이라 더 극단적인 양의 상관 예상.

---

## Figure 4 — Investment Simulation (Top-3 portfolio strategy)

매일 예측 확률 top-3 종목을 equal weight로 매수.

| Model | ACL18 SR | ACL18 APV | KDD17 SR | KDD17 APV | NI225 SR | NI225 APV |
|---|---|---|---|---|---|---|
| Market Index | 0.107 | 1.07 | 0.056 | 1.10 | 0.080 | 1.18 |
| PEN | 0.293 | 1.12 | 0.132 | 1.39 | 0.171 | 1.43 |
| DTML | 0.304 | 1.11 | 0.157 | 1.39 | 0.184 | 1.42 |
| CMIN | 0.357 | 1.24 | 0.169 | 1.46 | 0.201 | 1.51 |
| **CausalStock** | **0.369** | **1.32** | **0.192** | **1.49** | **0.259** | **1.52** |

**해석**:
- 모든 데이터셋에서 SR, APV 모두 SOTA.
- NI225에서 SR 개선폭 가장 큼 (+0.058 vs CMIN, +29% 상대 향상). 일본 시장에서 baseline들이 약했던 만큼 CausalStock의 효용 큼.
- ACL18 APV 1.32 (3개월 ~7%p 우위 over market). 연환산 시 매우 큰 차이.

> **재현 시 주의**: SR/APV는 분류 정확도보다 표본 변동성에 매우 민감하다. 같은 모델로도 random seed에 따라 SR ±0.05 정도 흔들리는 것이 일반적. CMIN-US/CMIN-CN의 investment simulation 결과는 본 논문에서 미보고.

---

## 재현 우선순위 — 어떤 수치를 먼저 맞춰야 하는가

| 우선순위 | 항목 | 이유 | 허용 오차 |
|---|---|---|---|
| **1** | **CausalStock w/o TCD on ACL18** (ACC ≈ 51%) | 모듈 분리 sanity check. TCD 없는 baseline이 random에 가까운지 확인 | ACC ±1%p |
| **2** | **CausalStock w/o news on ACL18** (ACC ≈ 58%) | TCD 모듈 단독 동작 확인 (price-only) | ACC ±1%p |
| **3** | **CausalStock full on ACL18** (ACC = 63.42) | 전체 pipeline | ACC ±0.5%p, MCC ±0.015 |
| **4** | **DNE-Llama on ACL18** (ACC = 62.82) | GPT-3.5 API 비용 없이 검증 가능. 가장 현실적인 1차 재현 목표 | ACC ±0.5%p |
| **5** | **w/o Lag-dependent TCD on ACL18** (ACC = 59.19) | Novelty 모듈의 기여도 검증 | ACC ±1%p |
| **6** | Hyper-param sensitivity 재현 | L=5, lr=1e-5, λ=0.01 의 robustness 확인 | trend만 일치하면 OK |
| **7** | Spearman corr. on ACL18 (0.79) | Causal graph의 외적 타당성 | ±0.1 |
| **8** | CMIN-US, CMIN-CN, KDD17, NI225, FTSE100 재현 | 일반화 입증 | ACC ±1%p |
| **9** | Investment simulation (SR, APV) | 표본 변동성 큼 | ±0.05 SR |

**1차 목표**: ACL18에서 ACC 62.9–63.9 + 주요 ablation 4개 (w/o TCD, w/o news, w/o Lag-dependent TCD, full)의 상대 순서 유지.
**2차 목표**: DNE-Llama 변형으로 GPT-3.5 비용 없이 62.3–63.3 달성 (Paper α의 cost-efficient baseline).
**3차 목표**: 6개 데이터셋 전체에서 일관성 검증, Spearman corr.까지 매칭.
