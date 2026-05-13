# CausalStock 논문 전체 요약

> **Paper**: Li, Sun, Lin, Gao, Shang, Yan. *CausalStock: Deep End-to-end Causal Discovery for News-driven Multi-stock Movement Prediction*. NeurIPS 2024 (arXiv:2411.06391v1, 10 Nov 2024).
> **Affiliation**: Gaoling School of AI, Renmin University of China (RUC) — Rui Yan 연구실의 PEN(AAAI'23) → CMIN(ACL'23) 라인 후속작.
> **Code/Data**: 논문 본문에는 GitHub URL 없음. NeurIPS Paper Checklist의 "Open access to data and code" 항목은 *"We release code and data in GitHub"* 라고만 답변되어 있어 실제 repo 위치는 별도 확인 필요 (Reproduction Questions 참조).

본 문서는 재현 구현(Paper α — 한국형 CausalStock)을 위한 reference 요약이다. 수식 번호는 원논문의 (1)–(18)을 그대로 따른다.

---

## 1. 문제 정의와 입출력 spec

**Task**: News-driven multi-stock movement prediction (이진 분류, rise=1 / fall=0).

**입력**: D개 종목, target trading day T, 과거 L 시점의 정보
```
X_{<T} = {X_t^i}_{i=1..D, t=T-L..T-1} = [C_{<T}, P_{<T}]
       = {[C_t^i, P_t^i]}
```
- `P_t^i`: i번째 종목 t일의 가격 feature 임베딩 (ACL18/CMIN은 7-dim raw price, KDD17/NI225/FTSE100은 11-dim 변환 feature)
- `C_t^i`: i번째 종목 t일의 뉴스 corpora 표현 (DNE 통과 후 dense embedding)

**출력**: `y_T = {y_T^i}_{i=1..D} ∈ {0,1}^D` — T일의 D개 종목 movement 동시 예측.

**확률 모델링 목표**: log-likelihood `log p(y_T | X_{<T})` 최대화. Causal graph G를 latent로 도입하여 다음과 같이 분해 (Eq. 2):

```
p(y_T | X_{<T}) = ∫_G p(y_T | X_{<T}, G) · p(G | X_{<T}) dG
```

좌변은 prediction process (parameters θ), 우변 두 번째 항은 causal discovery process (variational params φ).

---

## 2. 전체 아키텍처 개요

세 가지 핵심 모듈로 구성 (Figure 2):

1. **Market Information Encoder (MIE)** — `[C_t^i, P_t^i]` 생성. 내부에 **LLM-based Denoised News Encoder (DNE)** 와 Price Encoder.
2. **Lag-dependent Temporal Causal Discovery (Lag-dependent TCD)** — `p(G | X_{<T})` 학습. Bayesian variational inference + Gumbel-Softmax.
3. **Functional Causal Model (FCM)** — 발견한 G와 입력으로 `p(y_T | X_{<T}, G)` 계산. Additive noise SCM 기반.

학습은 **ELBO + BCE의 가중합**을 최소화 (Eq. 14).

---

## 3. 모듈 1: Market Information Encoder (MIE)

### 3.1 Price Encoder

i번째 종목 t일의 raw price vector
```
P̂_t^i = [P̂_t^{i,a}, P̂_t^{i,h}, P̂_t^{i,l}, P̂_t^{i,o}, P̂_t^{i,c}, V_t]
```
(adjusted close, high, low, open, close, volume — 단, ACL18/CMIN-US/CMIN-CN은 7-dim, KDD17/NI225/FTSE100은 11-dim 변환 feature 사용. Appendix C.1 Table 3 참조).

Embedding layer를 통과시켜 `P_t^i ∈ R^{d_p × 1}` 생성. 본 논문에서는 `d_p = 4` (Appendix C.4 grid search 결과).

### 3.2 LLM-based Denoised News Encoder (DNE) ⭐

**핵심 디자인 포인트**: BERT/RoBERTa/FinBERT처럼 dense vector를 뽑는 것이 아니라, **LLM(GPT-3.5)이 뉴스를 5개 차원으로 점수화**하고 그 점수를 그대로 representation으로 쓴다.

**5개 평가 차원** (Appendix A의 prompt 참조):

| 차원 | 척도 | 의미 |
|---|---|---|
| Correlation | 0–10 | 뉴스와 종목의 관련도 |
| Sentiment | –1 ~ +1 | 감성 극성 |
| Importance | 0–10 | 뉴스 이벤트의 중요도 |
| Impact | 0–10 | 주가에 미치는 영향력 |
| Duration | 0–10 | 영향력 지속 기간 |

**Prompt 구조** (Appendix A): [System] + [Default Prompt] + [Input]. Default Prompt에 *"please try to avoid assigning all-zero scores"* 와 출력 포맷 강제 ("Correlation: x\nSentiment: y\n...") 가 명시되어 있다.

**Pipeline**:
```
news text → LLM (GPT-3.5) → 5-dim score Ĉ_t^i ∈ R^{l × 5}
                          → Embedding layer → C_t^i ∈ R^{l × d_m}
```
여기서 `l = 10` (하루 최대 뉴스 수), `d_m = 64` (news embedding size).

**Ablation 결과 (Table 2)**: DNE-GPT-3.5(63.42) > DNE-Llama(62.82) > DNE-FinGPT(61.92) > 모든 traditional encoder(≤62.20). **LLM의 텍스트 평가 능력을 discrete-score denoising으로 환원하는 것이 핵심 novelty 중 하나**.

---

## 4. 모듈 2: Lag-dependent Temporal Causal Discovery (Lag-dependent TCD) ⭐

### 4.1 Temporal Causal Graph 정의 (Preliminary)

`G = [G_1, ..., G_L] ∈ R^{L × D × D}` — lag별 DAG의 시퀀스.
`G_{l,ji} = 1` ⟺ `X_{t-l}^j → X_t^i` causal link 존재.

**중요 관찰**: 시간 비가역성 덕분에 temporal causal graph는 자연스럽게 DAG → **NOTEARS류의 acyclicity 제약이 불필요**. (Section 4.3)

### 4.2 본 논문의 차별점: Lag-dependency

기존 Rhino[14], DECI[9]는 lag-independent factorization 사용:
```
p(G | X_{<T}) = ∏_l p(G_l | X_{T-l})
```

CausalStock은 **lag-dependent** factorization (Eq. 3):
```
p(G | X_{<T}) = p(G_1 | X_{T-1}) · ∏_{l=2..L} p(G_l | G_{l-1}, X_{T-l})
```

직관: 인접한 lag의 causal structure가 서로 관련 있다 (e.g., lag-1에서 활성화된 edge는 lag-2에서도 활성화될 가능성 ↑). Ablation에서 w/o Lag-dependent TCD 대비 +4%p (Table 2).

### 4.3 Graph Prior (Eq. 4)

```
p(G) ∝ exp(-λ_s · ||G_{1:L}||_F^2  -  λ_d · ||G_{1:L} - G^p_{1:L}||_F^2)
```
- 첫 항: graph sparseness (`λ_s = 1` per Appendix C.4)
- 둘째 항: domain-specific knowledge `G^p` 에 대한 soft constraint (`λ_d`는 본문에 명시값 없음 — 옵션). **이 부분이 한국형 확장에서 재벌 그룹/지분구조 prior로 활용 가능한 지점**.

### 4.4 Variational Posterior (Eq. 5–8)

각 edge `G_{l,ji}` 의 posterior를 Bernoulli로 둔다:
```
q_φ(G_{1,ji}) ~ B(1, σ_{1,ji})
q_φ(G_{l,ji} | G_{l-1,ji}) ~ B(1, σ_{l,ji})
```

**Existence/non-existence likelihood tensors** `U, V ∈ R^{L × D × D}` (learnable):
- `u_{l,ji}`: edge 존재 likelihood
- `v_{l,ji}`: edge 미존재 likelihood

**Lag-dependency 구현** (Eq. 6):
```
u'_{l,ji} = h_u(u_{l,ji}, u_{l-1,ji})
v'_{l,ji} = h_v(v_{l,ji}, v_{l-1,ji})
```
`h_u, h_v`는 trainable 3-layer MLPs.
*(Appendix C.4에서는 "1-layer MLPs"로 다른 기술 — Reproduction Questions에 기록함)*

**Edge probability** (Eq. 7, softmax 형태):
```
σ_{l,ji} = exp(u'_{l,ji}) / (exp(u'_{l,ji}) + exp(v'_{l,ji}))
```

학습 시 discrete sampling을 위해 **Gumbel-Softmax reparameterization** [24, 16] 사용.

### 4.5 Causal Weight Graph

Edge **존재 여부**만 모델링하면 인과의 **강도**를 표현 못한다. 그래서 별도 learnable tensor를 둔다:
```
Ĝ = {Ĝ_l}_{l=1..L} ∈ R^{L × D × D}
```

**Causal strength = G ⊙ Ĝ** (Hadamard product, dot product). 시각화·해석성 분석에 사용 (Figure 3b).

---

## 5. 모듈 3: Functional Causal Model (FCM)

### 5.1 Additive Noise SCM (Eq. 9)

```
y_T^i = F_i(Pa^i_G(<T), z_T^i) = f_i(Pa^i_G(<T)) + z_T^i
```
- `Pa^i_G(<T)`: G에 따른 i번째 노드의 lagged parents
- `z_T^i ~ N(0, (σ^i)^2)`, `σ^i` learnable
- `f_i : R^{D × L} → R^1` differentiable non-linear function, G가 규정하는 sparsity를 strict하게 만족 (`X_t^j ∉ Pa^i_G(<T) ⟹ ∂f_i/∂X_t^j = 0`)

### 5.2 본 논문의 FCM 구체 형태 (Eq. 10)

```
f_i(Pa^i_G(<T)) = Sigmoid( ζ_i( Σ_{l=1..L} Σ_{j=1..D} G_{l,ji} · Ĝ_{l,ji} · [ h_ℓ(P_{T-l}^j), ψ(C_{T-l}^j) ] ) )
```

- `ℓ, ψ`: shared-weight neural networks (모든 node·lag에 대해 동일 가중치 — 효율성 위함)
- `ζ_i`: per-node MLP
- `[·,·]`: concatenation
- Sigmoid가 movement 확률 출력

**구현 디테일** (Appendix C.4): `ζ_i, ℓ, ψ` 모두 **3-layer MLP, hidden size 332**.

### 5.3 Conditional Log-likelihood (Eq. 11–12)

Change-of-variables 공식:
```
p_θ(y_T^i | Pa^i_G(<t)) = p_{z^i}(z_T^i) · |∂F_i/∂z_T^i|^{-1} = p_{z^i}(z_T^i)
```
(`|∂F_i/∂z_T^i| = 1` because Eq. 9 is additive noise.)

따라서:
```
log p_θ(y_T | X_{<T}, G) = Σ_i log p_{z^i}(z_T^i)
```

---

## 6. 학습 목적함수 (Training Objective)

### 6.1 ELBO 유도 (Eq. 13)

```
log p_θ(y_T | X_{<T})
  ≥ E_{q_φ(G)}[ Σ_i log p_{z^i}(z_T^i)  +  log p(G) ]  +  H(q_φ(G))
```

세 항: (a) likelihood, (b) graph prior, (c) posterior entropy.

### 6.2 BCE Auxiliary Loss (Eq. 14)

ELBO만으로는 학습이 어려워, 다음의 보조 BCE를 추가:
```
BCE(g_T, y_T) = -Σ_i [ g_T^i · log(y_T^i)  +  (1-g_T^i) · log(1-y_T^i) ]
```
`g_T`는 ground-truth movement.

**최종 손실**:
```
L = (1/D) · ( -ELBO  +  λ · BCE(g_T, y_T) )
```
`λ = 0.01` (Appendix C.4 grid search 결과).

### 6.3 학습 시 주의사항 — Causal Stationary 가정 (Appendix B)

> *"Considering the instability of news data, we only leverage price data P_{<T} to discover causal graph G ... Technically, this could be realized by **detaching the gradient from C_{<T} to G**."*

즉, **causal discovery 모듈에는 price만 흐르고, news는 FCM aggregation 단계에서만 gradient를 받는다.** 구현 시 매우 중요한 디테일이다.

가정(Causal Markov Property, Minimality & Structural Identifiability, Correct Specification, Causal Sufficiency, Regularity of log-likelihood)은 DECI[9]의 가정을 따른다 (Appendix B).

---

## 7. 데이터셋 (Appendix C.1)

### 7.1 News-driven multi-stock task (3개 — 가격+텍스트)

| Dataset | Country | #Stocks | Train | Valid | Test | Price src | Text src | Price dim |
|---|---|---|---|---|---|---|---|---|
| **ACL18** [42] | US | 88 (9 industries) | 2014/01/02–2015/08/02 | 2015/08/03–2015/09/30 | 2015/10/01–2016/01/01 | Yahoo Finance | Twitter | 7 |
| **CMIN-US** [23] | US | 110 | 2018/01/01–2021/04/30 | 2021/05/01–2021/08/31 | 2021/09/01–2021/12/31 | Yahoo Finance | Yahoo | 7 |
| **CMIN-CN** [23] | CN | 300 (CSI300) | 2018/01/01–2021/04/30 | 2021/05/01–2021/08/31 | 2021/09/01–2021/12/31 | Yahoo Finance | Wind | 7 |

ACL18 price vector 7-dim: `[date, movement %, open, high, low, close, volume]`.

### 7.2 Multi-stock task without news (3개 — 가격만, 11-dim 변환 feature)

| Dataset | Country | #Stocks | Train | Valid | Test | Price dim |
|---|---|---|---|---|---|---|
| **KDD17** [45] | US | 50 | 2007/01/03–2015/01/01 | 2015/01/02–2016/01/03 | 2016/01/04–2017/01/01 | 11 |
| **NI225** [44] | JP | 51 | 2016/07/01–2018/03/01 | 2018/03/02–2019/01/06 | 2019/01/07–2019/12/31 | 11 |
| **FTSE100** [44] | UK | 24 | 2014/01/06–2017/01/03 | 2017/01/04–2017/07/03 | 2017/07/04–2018/06/30 | 11 |

11-dim 변환은 Adv-ALSTM (Feng et al. 2019) [8]의 정규화 + 가격 entry 간 상호작용 feature 방식을 따른다.

Train-test split은 **chronological** (시간 순서대로). Valid도 시간 순서대로 train 다음에 위치.

### 7.3 데이터셋 출처 URL (footnote)

- ACL18: https://github.com/yumoxu/stocknet-dataset
- CMIN-US/CN: https://github.com/BigRoddy/CMIN-Dataset
- KDD17: https://github.com/fulifeng/Adv-ALSTM
- NI225, FTSE100: https://datalab.snu.ac.kr/dtml

---

## 8. 평가 지표 (Appendix C.2)

### 8.1 Classification metrics

Confusion matrix `[[tp, fn], [fp, tn]]` 기준:

```
ACC = (tp + tn) / (tp + tn + fp + fn)

MCC = (tp·tn - fp·fn) / sqrt( (tp+fp)(fn+tp)(fn+tn)(fp+tn) )
```

*(원문 Eq. 15에 `fp + gn` 으로 적힌 부분은 오타. `fp + fn` 이 맞다.)*

### 8.2 Trading simulation metrics

**Accumulated Portfolio Value** (Eq. 17):
```
APV_t = ∏_{i=1..t} (1 + r^i)
```

**Sharpe Ratio** (Eq. 18):
```
SR = E[APV_t - R_f] / S[APV_t - R_f]
```
`R_f`: risk-free return.

**포트폴리오 전략**: 매일 예측 확률 top-3 종목을 동일 가중치로 매수.

---

## 9. Baselines

### 9.1 News-driven task
- **HAN** [15]: news-level + temporal attention, Bi-GRU
- **StockNet** [42]: MIE + VMD + ATA, recurrent + continuous latent variables
- **PEN** [21]: Bi-GRU + Shared Representation Learning + VOS (Vector of Salient)
- **CMIN** [23]: causality-enhanced correlation + 2 memory networks

### 9.2 Multi-stock (price-only) task
- **LSTM** [25]: 175 technical indicators + 5 normalized price features, 10-month rolling window
- **ALSTM** [28]: dual-stage attention LSTM
- **Adv-LSTM** [8]: ALSTM + adversarial training
- **DTML** [44]: attentive LSTM + multi-level context + transformer encoder

---

## 10. 학습 하이퍼파라미터 (Appendix C.4)

| Hyper-param | 값 | Search space |
|---|---|---|
| Optimizer | Adam [20] | — |
| Learning rate | **1e-5** | [1e-3, 1e-4, 1e-5, 1e-6] |
| Time lag L | **5** | [3, 5, 7, 9] |
| Loss weight λ | **0.01** | [0, 0.1, 0.01, 0.001] |
| Batch size | 32 | — |
| Price encoder hidden | 4 | [4, 8, 16] |
| Sparseness weight λ_s | 1 | — |
| News max words / day | w = 20 | — |
| News count / day | l = 10 | — |
| Word embedding | d_w = 50 | — |
| News embedding | d_m = 64 | — |
| h_u, h_v | 1-layer MLP | (본문 4.3에서는 3-layer라고 언급 — **모순 있음**) |
| ζ_i, ℓ, ψ | 3-layer MLP, hidden 332 | — |
| Param init | Xavier [11] | — |
| Hardware | 4 × NVIDIA Tesla V100 | — |
| #Runs (std 계산) | news-driven 10회 / price-only 5회 | — |

---

## 11. 핵심 결과 (요약, 상세는 expected-results.md 참조)

- **Table 1 (Main results)**: ACL18/CMIN-US/CMIN-CN/KDD17/NI225/FTSE100 6개 데이터셋에서 모두 SOTA. ACL18에서 ACC 63.42% (CMIN 62.69% 대비 +0.73%p, MCC 0.2172 vs 0.2090).
- **Table 2 (Ablation)**:
  - w/o TCD → ACC 51%대로 급락 (causal graph가 핵심)
  - w/o news → 5–8%p 하락
  - w/o Lag-dependent TCD → 3–4%p 하락 (lag-dependency 효과 입증)
  - Variable-dependent TCD → 미세 개선 가능하나 복잡도 O(L·D^2) → O(L·D^4)
  - DNE-GPT-3.5 > DNE-Llama > DNE-FinGPT > 모든 traditional encoder
- **Table 5 (Spearman corr.)**: 시가총액과 causal strength 간 강한 양의 상관 (ACL18: 0.79, FTSE100: 0.89). Causal discovery가 경제적 직관과 align.
- **Investment simulation (Figure 4)**: ACL18에서 SR 0.369 / APV 1.32 (CMIN 0.357 / 1.24 대비 우위).

---

## 12. 한국형 확장 (Paper α)을 위한 재현 시 주목 포인트

1. **DNE의 LLM 선택**: 원논문은 GPT-3.5. 한국 시장에서는 한국어 LLM(HyperCLOVA-X, EXAONE, GPT-4o-Korean) 비교 필요. 5-aspect prompt를 한국어 번역해야 함.
2. **Lag-dependent TCD**: 가장 중요한 novelty 모듈. NOTEARS 류 acyclicity 제약 없이 Gumbel-Softmax + Bernoulli 조합으로 구현 가능.
3. **Causal stationary 가정**: news C에서 G로 가는 gradient를 detach해야 함. 한국 시장 적용 시에도 동일.
4. **Domain-specific prior G^p**: 본 논문에서는 옵션으로만 언급 (Eq. 4). **한국형 확장의 차별점**: 재벌 그룹 구조 / DART 지분 관계 / 공급망 관계를 G^p로 인코딩.
5. **데이터 누락/sparsity 처리**: 원논문은 명시하지 않으나, 한국 시장 데이터는 영문보다 텍스트 sparsity가 클 가능성이 큼. all-zero score case의 처리 (Appendix A의 prompt에서 *"avoid all-zero, but allow if truly impossible"*) 가 한국어에서도 동일하게 작동하는지 검증 필요.
6. **FCM 교체 옵션**: 지도교수 연구 line(VAR + GRU hybrid)으로 `f_i`를 교체하는 것이 Paper α의 핵심 한 축. 원논문 FCM은 단순 sigmoid·concat·MLP이므로 교체 난이도 낮음.

---

## 13. Limitations (Appendix E, 원저자 자체 평가)

1. **시간 불변 causal graph**: 학습 후 G는 고정. 향후 meta-learning / incremental learning 으로 time-varied G 학습 필요.
2. **Bernoulli 가정**: edge 존재 여부만 modeling. multi-level causal relation을 보려면 더 복잡한 분포 필요.
3. **Safety issue (Appendix F)**: LLM 평가 결과가 human value를 위배할 위험.

→ 한국형 확장은 (1) regime-switching으로 시간 불변성 문제를, (5) 재벌 prior로 식별성을 동시에 공략 가능.
