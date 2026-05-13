# CausalStock 재현 시 불명확한 부분 — 질문 모음

> NeurIPS 2024 CausalStock 논문에서 명시되지 않거나 모순이 있어 **재현 구현 전 반드시 확인이 필요한 항목들**. GitHub 코드를 먼저 찾아 매칭하는 것이 가장 빠른 해결책. 코드 위치 불명 시 본 문서의 "기본 가정" 컬럼을 따라 시작하고 추후 sweep.
>
> **우선순위**: 🔴 critical (모듈 동작에 직결) / 🟡 important (수치 재현에 영향) / 🟢 minor (스타일 / 효율).

---

## A. 데이터 전처리 관련

### A.1 🔴 Price feature 정규화 방법

**문제**: 논문 본문에서는 raw price `P̂_t^i = [P̂^{i,a}, P̂^{i,h}, P̂^{i,l}, P̂^{i,o}, P̂^{i,c}, V_t]` 를 embedding layer에 통과시킨다고만 명시 (Section 4.2). 정규화 방법(z-score? min-max? log-return? Adv-ALSTM 스타일 11-dim 변환?)이 비명시.

**KDD17/NI225/FTSE100은 Adv-ALSTM의 11-dim 변환** 사용 (Appendix C.1: *"11 temporal features for normalizing prices and capturing the interaction between different raw price entries"*)이라고 한 줄로만 언급. 11-dim feature가 구체적으로 무엇인지(공식, 코드)는 [8] (Feng et al. 2019)을 직접 봐야 함.

**ACL18은 7-dim** (date, movement%, OHLCV)이지만 date를 어떻게 feature로 쓰는지(Unix timestamp? day-of-week one-hot?), movement %가 어디 기준인지 비명시.

**기본 가정**:
- ACL18/CMIN: z-score per stock per time window. movement % = `(close_t - close_{t-1}) / close_{t-1}`.
- KDD17/NI225/FTSE100: Adv-ALSTM [8] 공식 그대로 (open/high/low 정규화 비율, close 이동평균 등).
- date는 feature에서 제외하거나 day-of-week embedding으로 대체.

**확인 방법**: stocknet-dataset (https://github.com/yumoxu/stocknet-dataset) 의 preprocessing 코드 + Adv-ALSTM repo (https://github.com/fulifeng/Adv-ALSTM) 의 data loader.

---

### A.2 🔴 News-stock alignment & multiple news per day

**문제**: 하루에 한 종목당 평균 몇 개의 뉴스가 있는가? 그리고 `l = 10` (news count per day)을 초과하면 어떻게 처리하는가? 부족하면 zero-padding? 어떤 우선순위로 선택?

논문 Appendix C.4: *"the maximum word number in one piece of news and news number in one day are set to w=20, l=10"* — top-10 뉴스만 사용한다는 의미인 듯하나, **선택 기준이 비명시** (시간순? 무작위? GPT가 매긴 Importance score 순?).

**기본 가정**:
- 시간순 우선 (그날의 첫 10개 뉴스).
- 10개 미만이면 zero score `(0, 0, 0, 0, 0)` 으로 padding.
- 종목별 매칭: 뉴스 본문이나 metadata에 ticker 명시되어 있는 것만 (StockNet 데이터셋은 이미 ticker별로 분리).

**확인 방법**: StockNet 데이터셋의 raw structure 점검. CMIN dataset의 README.

---

### A.3 🟡 Trading day handling

**문제**: 주말/공휴일은 어떻게 처리하나? Lag L=5는 trading day 기준 5일인가, calendar day 기준 5일인가?

**기본 가정**: trading day 기준 5일 (대부분의 stock prediction 논문 표준).

**확인 방법**: NeurIPS 2024 코드 또는 baseline (CMIN, PEN) 코드의 시계열 구성 로직.

---

### A.4 🟡 ACL18 / CMIN 의 정확한 종목 리스트

**문제**: ACL18은 *"88 stocks in 9 industries"*, CMIN-US는 *"110 stocks"*, CMIN-CN은 *"300 CSI300 stocks"*. 정확한 ticker 리스트가 본문에 없음.

**기본 가정**: 각 dataset의 GitHub repo에서 제공하는 ticker list 그대로 사용.
- ACL18: https://github.com/yumoxu/stocknet-dataset (price/ 디렉토리의 종목 = 88개)
- CMIN: https://github.com/BigRoddy/CMIN-Dataset

---

### A.5 🟢 결측 가격 / 거래정지 처리

거래정지·상폐된 종목, 그날 거래량 0인 경우 처리 방법 비명시.

**기본 가정**: 결측은 직전 값으로 forward-fill, 전체 학습 기간 중 결측 비율 > 20%인 종목은 제외.

---

## B. Denoised News Encoder (DNE) 관련

### B.1 🔴 GPT-3.5 정확한 모델 버전 / 호출 시점

**문제**: 논문은 *"GPT-3.5"*만 명시. 구체적인 model string (`gpt-3.5-turbo`, `gpt-3.5-turbo-0301`, `text-davinci-003`?)이 비명시. 학습/추론 시 매번 API를 호출하는지, 사전에 모든 뉴스를 한 번에 점수화해 캐싱하는지도 비명시.

**기본 가정**:
- `gpt-3.5-turbo-0125` (안정 버전) 사용.
- **모든 뉴스를 사전에 한 번 점수화**해 디스크에 저장 후 재사용 (학습 중 매번 API 호출하면 비용·시간 비현실적).

**구현 시 주의**: ACL18(~수만 건), CMIN-US/CN(수십만 건) 뉴스 전수 점수화 비용 추정 필요. GPT-3.5-turbo 기준 100K 뉴스 × 250 tokens ≈ $50–100 수준.

**한국형 확장 시**: HyperCLOVA-X, EXAONE, Solar 등 한국 LLM 동일 방식 적용. 비용 비교 표 작성 필요.

---

### B.2 🔴 Prompt template 정확한 형태

**문제**: Appendix A의 prompt는 [System] + [Default Prompt] + [Input] 3개 component로 구성. **하지만 실제 OpenAI API 호출에서 이 3개를 어떻게 묶는지가 비명시**.

추정:
- [System] → `messages[0] = {"role": "system", "content": "As a stock trading news analyst, ..."}`
- [Default Prompt] + [Input] → `messages[1] = {"role": "user", "content": "<default prompt>\n\n<input>"}`

**또는** Default Prompt를 별도 user turn으로 두고 Input은 그 다음 turn?

**기본 가정**: 위 첫 번째 형태 (system 1개 + user 1개). temperature는 0 (deterministic output).

**확인 방법**: 코드의 prompt 구성 함수 직접 확인.

---

### B.3 🔴 LLM 출력 파싱과 score embedding 방법

**문제**: LLM이 반환한 텍스트 (e.g., `"Correlation: 8\nSentiment: -0.7\n..."`)를 어떻게 파싱하고, 5-dim score를 어떻게 `R^{l × d_m}` 임베딩으로 변환하는지 비명시. `d_m = 64`는 알지만 변환 함수가 불분명.

추정 가능한 방식:
1. 5-dim raw score를 그대로 5-dim vector → linear projection → 64-dim
2. 각 차원을 별도로 bin 처리 후 별도 embedding (Correlation은 11 bins, Sentiment는 21 bins ...) → concat → linear

**기본 가정**: 방식 1 (단순 linear projection). 5-dim → 64-dim FFN 1-layer.

**파싱 robustness**: LLM이 가끔 형식을 어기는 경우 (e.g., `"Correlation score: 8 (high)"`)에 대비해 regex + 예외 처리 필수. 파싱 실패 뉴스는 all-zero score로 fallback.

---

### B.4 🟡 한 뉴스의 `l × 5` 구조 vs 한 뉴스의 `5-dim`?

**문제**: 논문 본문 *"the i-th text at day t is represented as a five-dimensional representation Ĉ_t^i ∈ R^{l × 5}"* — `l`은 hyperparameter로 정의된 "한 뉴스의 단어 수"인가, 아니면 "하루에 뉴스 수"인가?

Appendix C.4: *"the maximum word number in one piece of news and news number in one day are set to w=20, l=10"* → **l = 하루 뉴스 수**.

따라서 `Ĉ_t^i ∈ R^{10 × 5}` = "하루에 최대 10개 뉴스 × 5차원 점수".

**기본 가정**: 위 해석. 그리고 `C_t^i ∈ R^{l × d_m} = R^{10 × 64}`는 그 day의 10개 뉴스 각각의 64-dim 임베딩 시퀀스.

**Aggregation**: FCM의 `ψ(C^j_{T-l})` 함수에서 `R^{10 × 64}` → 단일 vector로 어떻게 줄이는가? mean pooling? attention pooling? **이것이 비명시**. 기본 가정은 mean pooling.

---

### B.5 🟡 Score range가 다른 차원들의 처리

Correlation/Importance/Impact/Duration: 0–10
Sentiment: –1 ~ +1

5-dim vector 그대로 projection하면 Sentiment의 스케일이 다른 4개에 묻힌다.

**기본 가정**: 각 차원을 자신의 max로 정규화 (0–10 → 0–1, Sentiment는 그대로). 또는 sentiment를 (sentiment + 1) / 2로 0–1 매핑.

---

## C. Lag-dependent TCD 관련

### C.1 🔴 `h_u, h_v` MLP layer 수 모순

**Section 4.3 본문**: *"h_u and h_v are trainable 3-layer MLPs"*
**Appendix C.4**: *"λ_s = 1, h_v and h_u are all 1-layer MLPs"*

→ **모순**. 어느 것이 맞는지 불분명.

**기본 가정**: Appendix가 더 구체적이므로 **1-layer MLP** 우선. 차이를 sweep해서 확인.

---

### C.2 🔴 `h_u, h_v` 입력 형태

`h_u(u_{l,ji}, u_{l-1,ji})` — 두 scalar를 어떻게 결합하나? concat → MLP? 또는 element-wise sum/mul?

**기본 가정**: concat (`[u_{l,ji}, u_{l-1,ji}]` → 2-dim → MLP → 1-dim).

---

### C.3 🔴 Causal weight graph Ĝ 초기화 / 정규화

**문제**: `Ĝ ∈ R^{L × D × D}` 는 learnable. 초기화는? 음수 값도 허용? 활성화 함수로 ReLU 등 적용?

**기본 가정**: Xavier 초기화 (Appendix C.4의 "All parameters of our model are initialized with Xavier Initialization"). 활성화 없음 (raw learnable).

---

### C.4 🟡 Gumbel-Softmax temperature

**문제**: Gumbel-Softmax / Concrete distribution 사용 시 temperature τ 값이 비명시. τ가 작을수록 sample이 discrete에 가까워지고 학습이 어려워짐.

**기본 가정**: τ = 1.0 fixed. 또는 학습 중 annealing (1.0 → 0.5).

---

### C.5 🟡 Edge sampling — train vs inference

학습 시에는 Gumbel-Softmax로 stochastic sampling. **추론 시에는 어떻게 하나**? 확률값 threshold (0.5)? Hard argmax? 다중 sample averaging?

**기본 가정**: 추론 시 σ_{l,ji} > 0.5인 edge만 채택 (hard threshold). 또는 σ_{l,ji}를 그대로 G의 continuous relaxation으로 사용 (논문 본문이 "sampled" 라고만 표현하니 후자 가능성도 있음).

---

### C.6 🟡 Domain-specific prior G^p 가중치 λ_d

**문제**: Eq. 4의 `λ_d ||G_{1:L} - G^p_{1:L}||_F^2` 항. λ_d 값이 본문/Appendix 모두 비명시. λ_s = 1만 명시.

**기본 가정**: λ_d = 0 (즉 G^p 없이 baseline 실험). 한국형 확장(Paper α)에서 재벌 그룹 prior 적용 시 λ_d 를 [0.01, 0.1, 1.0]으로 sweep.

---

## D. FCM 관련

### D.1 🔴 ζ_i, ℓ, ψ 의 입출력 차원

Appendix C.4: *"ζ_i, ℓ and ψ are all 3-layer MLPs with hidden size 332"*

차원 추정:
- `ℓ : R^{d_p} → R^?` (price → hidden). `d_p = 4` → 입력 4-dim, 출력 ? (332?)
- `ψ : R^{d_m_aggregated} → R^?` (news → hidden). 뉴스가 `R^{10 × 64}`인데 어떻게 단일 vector로 줄이고 ψ에 넣는지 (B.4 참조)
- `ζ_i : R^? → R^1` (aggregated → logit). per-stock parameter set.

**기본 가정**:
- ψ 앞에 mean-pool 적용 → 64-dim → ψ(64-dim → 332-dim)
- ℓ(4-dim → 332-dim)
- concat → 664-dim → ζ_i(664-dim → 1-dim, sigmoid 전)

**확인 방법**: 코드 또는 본문 Eq. 10의 dimension 검산.

---

### D.2 🟡 BCE loss와 ELBO의 단위 일관성

Eq. 14: `L = (1/D) · (-ELBO + λ · BCE)`. ELBO는 log-likelihood scale (음수 큰 값), BCE는 cross-entropy scale (양수). 1/D 정규화로 stock 수 차이를 흡수하지만, **각 항의 magnitude 차이**가 큼.

**기본 가정**: λ=0.01이 이 magnitude 차이를 흡수하도록 fine-tune된 값. 다른 데이터셋(다른 D)에서는 λ가 재조정 필요할 수도 있음.

---

### D.3 🟡 Noise variance σ^i 의 초기화·정규화

`z^i ~ N(0, (σ^i)^2)`, σ^i는 learnable. 음수 방지 어떻게? softplus? exp?

**기본 가정**: σ^i = softplus(raw_σ^i) (음수 방지). 초기값 0.

---

### D.4 🟡 학습 중 noise 처리

학습 시 `y_T^i = f_i(...) + z^i`로 noise를 더하는가, 아니면 `y_T^i = f_i(...)` 만 쓰고 z는 BCE 계산에서만 사용?

논문 Eq. 12: `log p_θ(y_T | X_{<T}, G) = Σ_i log p_{z^i}(z_T^i)` — z_T는 학습 시 y_T (관측치) - f_i(...) 로 계산.

**기본 가정**: 학습 시 `z_T = g_T - f_i(...)` (잔차)로 두고 log_pdf 계산. 추론 시에는 z=0으로 두고 f_i만 사용.

---

## E. 학습 / 평가 관련

### E.1 🔴 Validation 사용 방법

**문제**: validation set의 정확한 용도가 비명시. Early stopping criterion? best epoch 선택? hyperparam search?

**기본 가정**:
- Early stopping: validation ACC가 N epoch (e.g., N=10) 동안 개선되지 않으면 중단.
- Best epoch on validation을 test에 적용.
- Hyperparameter grid search도 validation 기준.

---

### E.2 🟡 학습 epoch 수 / iteration 수

명시 없음. batch size 32만 알려짐.

**기본 가정**: max 100 epoch + early stopping patience 10. 데이터셋 별로 1 epoch 당 step 수가 다르니 wall-clock으로는 1–6시간 추정.

---

### E.3 🟡 Random seed 처리

논문 표에서 ACL18은 10 runs, KDD17은 5 runs 평균이라 명시. **각 run의 seed는?** seed 0~9? 시드별 결과 분포 보고 없음.

**기본 가정**: seed = 0~9 (또는 0~4). 모델/dataloader 모두 seed 적용.

---

### E.4 🟢 GPU 메모리 사용량

4× V100 (32GB 각) 사용. D=300 (CMIN-CN), L=5, batch=32일 때 G ∈ R^{L × D × D} = 5 × 300 × 300 = 450K elements per sample × 32 batch = 14M tensors. **그렇게 크지 않음**. V100 4개는 약간 over-spec인 듯, 단일 V100/A100에서도 충분히 돌 것으로 추정.

---

## F. Baseline 재현 관련

### F.1 🔴 Baseline 결과의 출처

Table 1의 baseline 수치들이 **원논문 재구현인지, baseline 논문에서 가져온 것인지** 비명시. 같은 split·전처리로 재학습했는지 확인 필요.

**기본 가정**: 본 논문에서 모두 재구현 (논문 standard practice). 단, CMIN 수치는 CMIN 원논문[23]과 매칭하는지 cross-check 필요.

---

### F.2 🟡 HAN, StockNet 등 baseline의 hyperparameter

각 baseline의 hyperparameter는 어떻게 설정했나? 원논문 default? CausalStock과 동일 search?

**기본 가정**: 원논문 default 사용.

---

## G. 코드/Repo 관련

### G.1 🔴 GitHub repo 위치

NeurIPS Paper Checklist에 *"We release code and data in GitHub"* 라고만 답. **본문/Appendix에 URL 없음**. arXiv v1 (2411.06391v1) 기준.

**우선 확인 사항**:
1. NeurIPS 2024 OpenReview 페이지 (논문 supplemental zip에 코드 포함되어 있을 가능성)
2. 저자 GitHub: shuqili (Shuqi Li), sunyuebo (Yuebo Sun). RUC Rui Yan 연구실 GitHub org도 확인.
3. arXiv v2/v3 업데이트 여부.
4. Google Scholar에서 본 논문 인용한 후속 논문이 코드 location 언급한 경우.

**확보 안 되면**: 본 문서의 "기본 가정" 으로 처음부터 구현. 4–6주 작업 추정.

---

### G.2 🟡 라이선스

코드/데이터 라이선스 비명시. 한국형 확장 paper에서 어떻게 cite할지, 코드 derivative work으로 명시할지 등 사전 확인 필요.

---

## H. 한국형 확장(Paper α) 고유 질문

### H.1 🟢 한국 시장에 5-aspect prompt가 한국어로 작동하는가

GPT-3.5/4 / HyperCLOVA-X / EXAONE에서 동일한 5-aspect prompt를 한국어로 옮겨 한국 뉴스에 적용 시:
- 출력 포맷이 깨지지 않는지
- Sentiment 분포가 합리적인지 (한국 뉴스는 영어 뉴스보다 hedging이 강한 경향)
- 한국어 LLM이 5개 점수 모두 0을 매기는 비율

→ **본 paper의 첫 sanity check 실험**. 100개 한국어 뉴스 sample로 사전 확인.

### H.2 🟢 재벌 그룹 prior G^p 구성 방법

DART OpenAPI의 *지배구조* 정보 + KOSPI 200 구성종목.
- Edge weight = 지분율? binary?
- 같은 그룹 내 종목 간 양방향 vs 일방향 (지주회사 → 자회사)?
- λ_d sweep 범위

→ Paper α의 핵심 novelty이므로 별도 design doc 필요.

### H.3 🟢 한국 시장 trading day와 시차

미국 시장이 닫힌 시간 동안 발생한 한국 뉴스가 다음 날 미국 시장 open에 영향을 주는 cross-market effect. 한국 시장만 다루면 simpler 하지만 cross-market 확장도 가능.

→ Paper α는 한국 시장 내부만, cross-market은 향후 sub-paper로.

---

## I. 재현 구현 중 추가로 발견된 ambiguity (Phase 0–9 작업 중 기록)

### I.1 🟡 ACL18 ticker 수: paper "88" vs 실제 cover "85"
StockNet의 88개 price file 중 GMRE(2016-07 IPO), BABA(2014-09 상장), AGFS(2014-11 상장)는 train 시작일(2014-01-02) 기준으로 history가 없거나 부족함. 다중 종목 모델은 D를 고정해야 하므로 fully-covered 85개만 사용.

**기본 가정**: 85 stocks. 결과가 paper와 어긋나면 D=88로 padded variant도 시도.

### I.2 🔴 q_φ(G | X_{<T})의 input 의존성 (paper-summary §4.4)
표기는 conditional on X지만 본문은 U, V를 free parameter로 설명. 두 해석이 코드로 완전히 다른 모델을 만듦.

**기본 가정**: input-independent (U, V free). `src/models/tcd.py`가 이 가정으로 작성됨. 미달 시 amortized variant sweep.

### I.3 🔴 FCM 노이즈 모델과 binary y의 충돌 (paper-summary §5.1, Eq. 9–12)
`y = Sigmoid(...) + z` with `z ~ N(0, σ²)` + change-of-variables(Eq. 11)는 binary y와 호환 안 됨. 두 가지 reduction:
- **Gaussian**: `z = g_T - f_i`, log N(z; 0, σ²) (paper-literal)
- **Bernoulli**: `f_i = p(y=1)`, log p(y|f) = BCE form

**기본 가정**: Gaussian (default). `src/training/loss.py`의 `likelihood_form` 플래그로 둘 다 지원. λ=0 ablation 결과 보고 결정.

### I.4 🟢 Energy-based prior Z(λ_s, λ_d)
Eq. 4의 정규화 상수 Z는 λ_s, λ_d 고정 시 G에 무관 → ELBO gradient에 영향 없음. 코드에서 무시.

### I.5 🟢 ψ(C̄_{T-l}^j)의 l-news reduction
`C_{T-l}^j ∈ R^{l × d_m}`을 단일 vector로 줄이는 방식 미명시. 본 구현은 **mean-pool** 사용 (B.4 default와 일치). attention-pool sweep은 미달 시.

### I.6 🟡 DNE LLM 변경 (paper deviation, 2026-05-13)
Paper는 `gpt-3.5-turbo` 사용. 본 구현은 비용·속도·품질 모두 우월한 **`gpt-5.4-mini`** 사용 (2026년 기준 표준 model). Paper Table 2c가 "더 좋은 LLM → 약간 더 좋은 ACC"를 보였으므로, 결과는 paper 63.42에서 +0~+1%p 정도 차이 예상.

**Why**: gpt-3.5-turbo Tier 1 한도(60K TPM)로 ACL18 전체 점수화에 ~16시간 소요. gpt-5.4-mini는 더 빠르고 저렴.

**How to apply**: paper 정확 매칭이 필요한 비교는 별도 `dne_acl18_gpt35.parquet` 캐시로 추가 실험 가능.

---

## 우선 해결해야 할 Top-5

1. **G.1 GitHub repo 위치 확보** — 다른 모든 질문을 한 번에 해결.
2. **A.1 Price 정규화** — 모든 후속 실험에 영향.
3. **B.1–B.3 DNE 구현 detail** — 핵심 모듈, API 비용 사전 계획 필요.
4. **C.1–C.3 TCD 모듈 detail** — 본 논문의 novelty 핵심.
5. **D.1 FCM 입출력 차원** — 코드 작성 시 가장 먼저 막힐 지점.
