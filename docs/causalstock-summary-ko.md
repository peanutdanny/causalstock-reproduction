# CausalStock 논문 정리와 재현 현황 (한국어)

> **작성일**: 2026-05-21
> **목적**: 논문의 핵심을 단계별 스텝으로 풀어 한국어로 정리하고, 현재 재현 작업이 어디까지 와 있는지 한눈에 파악
> **관련 문서**:
> - 깊은 수식·디테일 → [paper-summary.md](paper-summary.md) (한·영 혼용)
> - 구현·테스트 상세 → [project-status.md](project-status.md)
> - 미해결 질문 → [reproduction-questions.md](reproduction-questions.md)
> - 재현 목표 수치 → [expected-results.md](expected-results.md)

---

## 한 줄 요약

> **종목 간 *상관*이 아닌 *인과* 관계를 학습하고, LLM으로 뉴스 노이즈를 제거하여, 6개국 시장에서 SOTA를 달성한 multi-stock 방향예측 모델.**

---

## 1부. 논문 핵심 정리

### 1.1 풀고자 한 문제 (2가지)

**문제 ①. 종목 간 관계를 어떻게 모델링하나**
- 기존 연구는 **상관관계**(correlation)로 표현 → 양방향, 방향성 없음
- 그러나 실제 영향은 방향성이 있음 (예: 삼성전자 실적 발표 → SK하이닉스 주가)
- **저자 주장**: 종목 간 영향은 **인과관계**(causal)로 모델링해야 한다

**문제 ②. 뉴스 데이터의 노이즈**
- 뉴스에는 쓸모없는 정보가 너무 많음 → 유효 신호 추출 어려움
- 기존 BERT/FinBERT 류 dense embedding으로는 부족

### 1.2 어떻게 풀었나 (3가지 핵심 모듈)

```
[가격 OHLCV]                    [뉴스 corpora]
      │                              │
      ▼                              ▼
[Price Encoder]      [LLM-based Denoised News Encoder (DNE)]
                            · GPT-3.5가 5-aspect로 점수화
                              (Correlation, Sentiment, Importance,
                               Impact, Duration)
      │                              │
      └───────── concat ─────────────┘
                       │
                       ▼
        [Lag-dependent Temporal Causal Discovery (TCD)] ⭐
          · 종목 간 인과 그래프 G를 latent로 학습
          · Variational Bayes + Gumbel-Softmax
                       │
                       ▼
            [Functional Causal Model (FCM)]
              · 발견한 G로 다음날 rise/fall 예측
              · Additive noise SCM
                       │
                       ▼
              y_T = {0,1}^D (D개 종목 동시 예측)
```

### 1.3 최종 결과

**저자가 입증한 4가지**:

1. **6개국 SOTA**: ACL18(미)·CMIN-US·CMIN-CN·KDD17·NI225(일)·FTSE100(영) 전부에서 기존 baseline 초과
2. **TCD가 성능의 원천**: ablation에서 TCD 제거 → ACC 51%로 폭락(거의 random)
3. **해석가능성**: 발견된 causal strength ↔ 시가총액 강한 양의 상관 (ACL18 0.79, FTSE100 0.89)
4. **투자 시뮬레이션**: top-3 매수 전략에서 Sharpe Ratio·누적수익 모두 baseline 능가

**저자가 인정한 한계** (Appendix E):
- 학습 후 인과 그래프 G가 **시간 불변** → regime change 대응 불가
- **Bernoulli edge**만 (있다/없다) → 다단계 인과 강도 표현 불가
- LLM 평가 결과의 **safety** 문제

---

## 2부. 논문의 단계별 스텝

### Step 1. 문제 정의

**입력**:
- D개 종목, target day T, 과거 L 시점의 가격+뉴스
- `X_{<T} = {[C_t^i, P_t^i]}` for `i=1..D`, `t=T-L..T-1`

**출력**:
- `y_T ∈ {0,1}^D` — D개 종목 모두의 다음날 rise(1)/fall(0)

**확률 모델**:
- log-likelihood `log p(y_T | X_{<T})` 최대화
- **핵심 아이디어**: causal graph G를 latent로 도입

```
p(y_T | X_{<T}) = ∫_G p(y_T | X_{<T}, G) · p(G | X_{<T}) dG
                    └─ prediction ─┘   └ causal discovery ┘
                       params θ           params φ
```

### Step 2. 데이터 선정

총 6개 데이터셋. 두 task로 분리:

**Task A — 뉴스 포함 (3개)**

| 데이터셋 | 국가 | 종목 수 | 뉴스 출처 | Price dim |
|---|---|---:|---|---:|
| **ACL18** | US | 88 | Twitter | 7 |
| **CMIN-US** | US | 110 | Yahoo | 7 |
| **CMIN-CN** | CN | 300 (CSI300) | Wind | 7 |

**Task B — 가격만 (3개)**

| 데이터셋 | 국가 | 종목 수 | Price dim |
|---|---|---:|---:|
| **KDD17** | US | 50 | 11 |
| **NI225** | JP | 51 | 11 |
| **FTSE100** | UK | 24 | 11 |

**Train/Valid/Test split**: 시간 순서대로(chronological). 무작위 셔플 ❌.

### Step 3. 데이터 전처리

**가격**
- 7-dim raw OHLCV: `[date, movement%, open, high, low, close, volume]`
- 11-dim 변환: Adv-ALSTM 방식 정규화 + 가격 entry 간 상호작용 feature

**뉴스 → DNE 점수화**
- 종목 + 뉴스 텍스트를 GPT-3.5에 프롬프트
- 5-aspect 정수 점수 반환 → 5차원 벡터로 사용
- (paper의 Appendix A에 prompt 템플릿 있음)

**Label 생성**
- rise=1 if `close_t > close_{t-1}` else 0
- 즉 단순 binary

**※ 논문에 명시 안 된 implicit 처리 (재현 중 발견)**
- raw price feature 스케일이 극단적(volume ~10⁸ vs movement ~10⁻²)
- **z-score normalize 사실상 필수** — train 통계 기반 per-feature 정규화
- 안 하면 Xavier-초기화된 layer가 즉시 saturation, 학습 불가

### Step 4. 모델 설계 (3 모듈)

#### Step 4-1. Market Information Encoder (MIE)

- **Price Encoder**: 7-dim 또는 11-dim raw → d_p 차원 linear
- **DNE (LLM-based)**: 위 Step 3에서 만든 5-aspect 점수 → 64-dim embedding
- 최종 결합: `[P_t^i, C_t^i]`

#### Step 4-2. Lag-dependent Temporal Causal Discovery (TCD) ⭐

**역할**: 종목 간 인과 그래프 G의 변분 사후분포 `q_φ(G | X_{<T})` 학습

**핵심 차별점** — *lag-dependency*:
- 기존 NOTEARS 류: 시간 무관 단일 G
- 본 논문: lag별 G_l, 그리고 G_l이 이전 G_{l-1}에 의존

**구현 트릭**:
- 각 lag별 Bernoulli 엣지 확률을 **Gumbel-Softmax**로 미분 가능하게 샘플
- Acyclicity는 lag separation에서 자연스럽게 보장 (NOTEARS의 trace constraint 불필요)

#### Step 4-3. Functional Causal Model (FCM)

**역할**: 발견된 G와 입력으로 `p(y_T | X_{<T}, G)` 계산

**구체 형태** (Eq. 10):
- Additive noise SCM: `y_T^i = f_i(parents in G) + ε_i`
- `f_i = ζ_i(concat[ℓ(P), ψ(C)])` 형태의 3-layer MLP
- `ε_i ~ N(0, σ_i²)` (Gaussian)
- 최종 movement 확률: `σ(ζ_i(...))` (sigmoid)

### Step 5. 학습 (ELBO + BCE)

**목적함수** (Eq. 14):

```
L = -ELBO(θ, φ) + λ · BCE(y_true, y_pred)
```

- **ELBO**: variational lower bound on log-likelihood
- **BCE**: 보조 분류 손실로 학습 안정화
- **λ = 0.01** (논문 권장)

**핵심 주의사항**: causal-stationary 가정 (Appendix B)
- 뉴스 C → G 그래디언트를 **detach** 해야 함
- 안 하면 인과 식별성 무너짐

**Hyperparameter** (Appendix C.4)
- Optimizer: Adam, lr=1e-5
- Batch size: 32, max epochs: 100
- Hidden dim: 332, 3-layer MLP
- L=5 lag, l=10 뉴스/일
- Seed: news-driven task 10회 평균, price-only 5회 평균

### Step 6. 평가

**Classification metrics**
- **ACC**: `(tp+tn) / (tp+tn+fp+fn)`
- **MCC**: Matthews correlation — 클래스 불균형에 강건

**Trading simulation** (Appendix C.2)
- 매일 예측확률 **top-3 종목**을 동일 가중치로 매수
- **APV** (Accumulated Portfolio Value): `∏(1 + r_i)`
- **Sharpe Ratio**: `E[APV - R_f] / Std[APV - R_f]`

### Step 7. 결과 분석 (논문 Table 1, 2, 5, Figure 4)

- **Table 1**: 6개 데이터셋 SOTA — ACL18에서 ACC 63.42%, MCC 0.2172
- **Table 2 (ablation)**:
  - w/o TCD → 51%대 (graph가 없으면 사실상 random)
  - w/o news → 5–8%p 하락
  - w/o lag-dependency → 3–4%p 하락
  - DNE-GPT-3.5 > DNE-Llama > DNE-FinGPT > traditional encoder
- **Table 5 (Spearman corr.)**: causal strength ↔ market cap 강한 양의 상관
- **Figure 4**: top-3 portfolio Sharpe 0.369, APV 1.32 (CMIN 대비 우위)

---

## 3부. 재현 작업 진행 현황 (2026-05-21 기준)

### 3.1 완료된 단계 ✅

| 영역 | 내용 |
|---|---|
| 데이터 파이프라인 | ACL18 로더, StockNet tweet parser, DNE mock + GPT(sync+async) + parquet cache |
| 모델 모듈 | MIE, TCD, FCM, CausalStock wrapper 전체 (총 5개 nn.Module) |
| 학습 | ELBO+BCE loss, trainer (early-stop, checkpoint, seed determinism) |
| 평가 | classification(ACC/MCC), trading(APV/Sharpe) |
| 시각화 | causal plot, training curve, comparison plot 등 6개 |
| 단위 테스트 | **13 파일, 총 61개** (G1~G4 검증 완료) |
| 실험 인프라 | configs (full + 4 ablations), 결과 JSON 자동 저장 |
| Sanity report | Phase 9.5에서 critical bug 1건 발견·수정 |

### 3.2 1-seed 학습 결과 (paper-exact lr=1e-5)

| 설정 | best epoch | test ACC | test MCC | 논문 target | 평가 |
|---|---:|---:|---:|---|---|
| **Full** (mock DNE) | 100 | **0.623** | 0.246 | 0.6342 ± 0.13 | **−1.1%p** (1 seed 기준 striking distance) |
| w/o TCD | 14 | 0.540 | 0.073 | ≈ 0.51 | **+3%p** (chance 수준 일치) |
| w/o news | 100 | 0.621 | 0.244 | 0.581 ± 0.01 | **+4%p** (mock DNE가 noise라 full≈no-news, 정상 동작) |
| w/o lag-dep | 11 | 0.504 | 0.009 | ≈ 0.59 | **−9%p — 조사 필요** ⚠️ |
| λ=0 (BCE off) | 100 | 0.626 | 0.252 | (논문 미보고) | 참고용 |

### 3.3 검증 게이트 진행도

| 게이트 | 묻는 질문 | 상태 |
|---|---|---|
| **G1** 데이터 패리티 | 88 종목 × 거래일 수가 논문 Table C.1과 일치? | ✅ |
| **G2** 포워드 패스 | 입력 → 출력 shape, 파라미터 수 정상? | ✅ |
| **G3** 손실 sanity | 단일 배치 loss 감소, gradient 정상? | ✅ |
| **G4** Tiny overfit | 소량 데이터에 overfit 가능? | ✅ (sanity sweep에서 lr↑+bernoulli로 0.72) |
| **G5** Full 재현 | 10 seeds 평균 ACC 62.9–63.9? | ⏳ **진입 직전** (현재 1 seed 0.623) |

### 3.4 발견된 핵심 이슈

#### 이슈 1 — Xavier 초기화 + 비정규화 feature → 학습 실패 (해결됨)

**증상**: 풀데이터 10-epoch 실행에서 모든 ablation이 chance(0.50) 수준

**원인**: raw price feature 중 `feat4`(min −48, max 57)와 `volume`(min 0, max 463M)이 Xavier-초기화된 선형층을 즉시 saturation → gradient 차단

**해결**: train-split 통계 기준 per-feature **z-score normalize** 추가 (`CausalStockDataset.compute_feature_stats`)

**교훈**: 논문이 명시하지 않은 implicit 가정. [reproduction-questions.md](reproduction-questions.md)에 기록 권장.

#### 이슈 2 — w/o lag-dep ablation 결과 불일치 (미해결)

| | 우리 결과 | 논문 |
|---|---:|---:|
| test ACC | 0.504 | ≈ 0.59 |

- epoch 11에서 early-stop → 학습 자체가 거의 안 됨
- 다른 ablation들은 striking distance인데 이것만 chance 수준
- **가설**: lag-dependency 제거 시 어떤 fallback 메커니즘(예: 평균 G 공유) 구현이 누락되었을 가능성
- **다음 액션**: paper-checker subagent로 TCD 모듈 검증

### 3.5 남은 작업 (Phase 1 마무리)

| 우선순위 | 작업 | 예상 기간 |
|---|---|---|
| 🔴 높음 | 실 GPT-3.5 DNE 캐시 생성 (`scripts/score_news.py --gpt`) | 1주 |
| 🔴 높음 | w/o lag-dep 갭 조사 (paper-checker agent) | 2-3일 |
| 🔴 높음 | 10 seeds × 100 epochs sweep (G5 통과 조건) | ~1주 (CPU, 5h × 5 configs) |
| 🟡 중간 | 평균 ± std로 논문 표와 비교 (experiment-analyst agent) | 1일 |
| 🟢 낮음 | CMIN-US/CMIN-CN/KDD17/NI225/FTSE100 확장 | 2-3주 (각 데이터셋 loader 신규 필요) |

### 3.6 작업한 코드 구조

```
causalstock-reproduction/
├── src/
│   ├── data/          (8 파일) ACL18, stocknet, DNE (mock/sync/async), cache
│   ├── models/        (5 파일) MIE, TCD, FCM, CausalStock wrapper
│   ├── training/      (3 파일) loss(ELBO+BCE), trainer
│   ├── evaluation/    (3 파일) classification, trading
│   ├── utils/         (4 파일) seed, config, logging
│   └── visualization/ (6 파일) causal/training/trading plots
├── tests/             (14 파일 / 61 테스트)
├── experiments/
│   ├── configs/       (acl18.yaml + 4 ablations)
│   └── results/       (5 ablation × 1-seed JSON + backtest .npz + sanity report)
├── scripts/           score_news.py (GPT-3.5 DNE batch scoring)
├── docs/              paper summary, expected results, reproduction questions, project status
└── .claude/
    ├── agents/        paper-checker, experiment-analyst
    └── skills/        4 skills (math-rigor, paper-faithful, data-pipeline, experiment-runner)
```

---

## 4부. Phase 2 방향 (2026-05-21 피벗 결정)

기존 [CLAUDE.md](../CLAUDE.md)의 commitment("Korean 시장 + 재벌 prior + GRU-VAR FCM")에서 **cross-asset 인과 발견** 으로 thesis 방향 전환.

**3가지 기여 축**:
1. cross-asset(D≥25 — FX major + sovereign rates + commodities + equity indices) end-to-end causal discovery + LLM macro-news (학계 빈 영역)
2. **GRU-VAR FCM + Student-t likelihood** (수학적 기여, 지도교수 line 보존)
3. **Lead-lag causal prior G^p** (관측된 cross-asset 전파 패턴 기반)

**주요 변경**:
- 평가 지표: **Sharpe primary** (FX 방향 ACC 한계 ~53%)
- Korean+재벌 prior: **비교 appendix 챕터로 demote**
- Target 학회: NeurIPS 2027 또는 ICML 2027

**상세 plan**: `~/.claude/plans/glimmering-percolating-mist.md`

Phase 1(현재 작업)이 G5까지 통과한 후 Phase 2A(cross-asset 데이터 파이프라인)로 진입.

---

## 한 줄 현황 요약

> CausalStock의 3 모듈(MIE/TCD/FCM) + ELBO+BCE 학습 + ACL18 파이프라인 + 61 단위테스트 + 5 ablation 1-seed 실행 완료. Phase 1 잔여: 실 GPT-3.5 캐시 → 10 seeds sweep → 논문 ±0.5% 매칭 확인. 그 후 Phase 2(cross-asset thesis) 진입.
