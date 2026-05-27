# CausalStock Reproduction — 종합 정리

> **목적**: 논문(CausalStock, Li et al. NeurIPS 2024)의 데이터·방법·결론을 한눈에 정리하고, 현 시점(2026-05-21)까지 본 재구현 프로젝트가 어디까지 와 있는지 요약한다.
>
> **상세 자료**:
> - 논문 전체 요약 → [paper-summary.md](paper-summary.md)
> - 재현 목표 수치 → [expected-results.md](expected-results.md)
> - 미해결 질문 → [reproduction-questions.md](reproduction-questions.md)
> - 의존성 → [dependency-list.md](dependency-list.md)

---

## Part 1. 논문(CausalStock) 정리

### 1.1 데이터 선정

저자들은 **6개 데이터셋, 4개국**(US·CN·JP·UK)에서 검증해 일반화를 입증한다. 데이터를 두 task로 분리:

| Task | 데이터셋 | 종목 수 | 텍스트 입력 | Price dim |
|---|---|---:|---|---|
| News-driven multi-stock | **ACL18** (US) | 88 | Twitter | 7 |
| News-driven multi-stock | **CMIN-US** | 110 | Yahoo News | 7 |
| News-driven multi-stock | **CMIN-CN** | 300 (CSI300) | Wind | 7 |
| Price-only multi-stock | **KDD17** (US) | 50 | — | 11 |
| Price-only multi-stock | **NI225** (JP) | 51 | — | 11 |
| Price-only multi-stock | **FTSE100** (UK) | 24 | — | 11 |

**선정 기준**: (a) 시장 다양성, (b) 뉴스 유무 비교, (c) 선행 연구들과 baseline 비교 가능. ACL18 88 stocks × 64 test days ≈ 5,600여 개 예측이 단일 ACC·MCC로 평가된다.

### 1.2 데이터 처리

```
원시 가격 (OHLCV)          뉴스 corpus
        │                      │
        ▼                      ▼
[Price Encoder]      [Denoised News Encoder (DNE)]
  · 7-dim raw or         · LLM (GPT-3.5) 점수화
    11-dim Adv-ALSTM       · 5 aspect: Correlation,
  · linear → d_p           Sentiment, Importance,
                           Impact, Duration
        │                      │
        └─────── concat ───────┘
                  │
                  ▼
            [C_t^i, P_t^i] for each (stock i, day t)
```

**Split**: 시간 순서대로 chronological. ACL18 기준 — train(2014-01 ~ 2015-08) / valid(2015-08 ~ 09) / test(2015-10 ~ 2016-01).

**Label**: rise=1 if close_t > close_{t-1}, else 0 (이진 분류).

**중요한 implicit 처리** (논문에 명시 없음, 본 재구현에서 발견): raw price feature는 스케일이 극단적(volume ~10⁸, mvmt ~10⁻²)이라 Xavier 초기화에서 saturate. **z-score normalize**가 사실상 필수. (자세히는 §2.4 참조)

### 1.3 연구 프로세스 (방법론)

end-to-end 학습. 3개 모듈로 분해:

| 모듈 | 역할 | 학습 대상 |
|---|---|---|
| **MIE** (Market Information Encoder) | 가격+뉴스 임베딩 생성 | Price/News encoder weights |
| **Lag-dep TCD** (Temporal Causal Discovery) ⭐ | 종목 간 인과 그래프 G를 latent로 학습 | Variational posterior `p(G\|X)` (Gumbel-Softmax) |
| **FCM** (Functional Causal Model) | 발견한 G로 다음날 movement 예측 | Additive noise SCM `f_i, ζ_i` |

**손실**: ELBO + λ·BCE (Eq. 14). λ=0.01.

**평가 지표**:
- Classification: ACC, MCC
- Trading: APV(누적수익), Sharpe Ratio — top-3 일별 매수 전략

**Hyperparameter** (Appendix C.4): lr=1e-5, batch=32, hidden=332, lag L=5, news/day l=10, optimizer=Adam, 10 seeds로 평균±std 보고.

### 1.4 최종 결론

**저자들의 핵심 주장**:
> 종목 간 관계를 *상관* 대신 *인과*로 모델링하고, 뉴스 노이즈를 LLM으로 제거하면 multi-stock 방향 예측이 일관되게 개선된다.

**입증 4가지**:
1. 6개국 시장 전부에서 SOTA (Table 1)
2. TCD 제거 시 ACC가 51%로 폭락 → 인과 모듈이 성능의 원천 (Table 2)
3. 발견된 causal strength ↔ 시가총액 강한 상관(0.79~0.89) → **해석가능성** (Table 5)
4. 투자 시뮬레이션 Sharpe·APV에서도 baseline 능가 (Figure 4)

**저자 자체 인정 한계 (Appendix E)**:
- **시간 불변 그래프** G가 학습 후 고정 → regime change 반영 불가
- **Bernoulli edge**만 모델링 → multi-level causal strength 불가
- **LLM safety** 우려

→ 박사 thesis(Paper α)는 이 한계들을 정확히 공략한다: 재벌 prior G^p, regime-switching, GRU-VAR FCM.

---

## Part 2. 재구현 현황 (2026-05-21 기준)

### 2.1 구현 완료된 코드

```
src/
├── data/        ─ ACL18 loader, StockNet tweet parser,
│                  DNE: mock + GPT (sync, async) + parquet cache, 종목명 매핑
├── models/      ─ MIE, TCD, FCM, CausalStock 전체 wrapper
├── training/    ─ loss (ELBO+BCE), trainer (early-stop, checkpoint)
├── evaluation/  ─ classification(ACC/MCC), trading(APV/Sharpe)
├── utils/       ─ seed determinism, config loader, logging
└── visualization/ ─ causal/comparison/score/trading/training plots
```

총 26개 Python 모듈. [src/](../src/) 참조.

### 2.2 검증 상태 — 테스트

| 영역 | 파일 | 테스트 수 |
|---|---|---:|
| Data: ACL18 loader | test_acl18_loader.py | 5 |
| Data: DNE mock + cache | test_dne_mock_and_cache.py | 5 |
| Data: DNE GPT (mocked client) | test_dne_gpt.py | 4 |
| Data: DNE GPT async | test_dne_gpt_async.py | 4 |
| Model: MIE | test_mie.py | 4 |
| Model: TCD | test_tcd.py | 6 |
| Model: FCM | test_fcm.py | 5 |
| Training: loss | test_loss.py | 7 |
| Training: trainer smoke | test_trainer_smoke.py | 1 |
| Eval: classification | test_metrics.py | 9 |
| Eval: trading | test_backtest.py | 4 |
| Config: ablations | test_ablation_configs.py | 4 |
| Seed determinism | test_seed_determinism.py | 3 |
| **합계** | **13 파일** | **61** |

→ G1·G2·G3 단계의 unit/component 검증은 통과. 실제 API를 호출하는 GPT 호출은 mocked client로 격리(`tests/test_dne_gpt.py:1`).

### 2.3 실제 학습 결과 (1 seed, 100 epochs, paper-exact lr=1e-5)

| 설정 | best epoch | test ACC | test MCC | 논문 target | 평가 |
|---|---:|---:|---:|---|---|
| **Full** (mock DNE) | 100 | **0.623** | 0.246 | 0.6342 ± 0.13 | -1.1%p, 1 seed 기준 striking distance |
| w/o TCD | 14 | 0.540 | 0.073 | ≈ 0.51 | +3%p (chance에 가까운 수준 일치) |
| w/o news | 100 | 0.621 | 0.244 | 0.581 ± 0.01 | +4%p — mock DNE가 noise라 full≈no-news (정상) |
| w/o lag-dep | 11 | 0.504 | 0.009 | ≈ 0.59 | **−9%p — 조사 필요** |
| λ=0 (BCE off) | 100 | 0.626 | 0.252 | (논문 미보고) | 참고 |

전체 history JSON: [experiments/results/](../experiments/results/)

### 2.4 발견·수정한 중요 버그 (Phase 9.5 sanity)

[experiments/results/sanity_5epoch.md](../experiments/results/sanity_5epoch.md) 요약:

- **증상**: 초기 10-epoch 풀데이터 실행에서 모든 ablation이 chance(0.50)에 갇힘
- **원인**: raw price feature 중 `feat4`(min −48.4, max 57.6) 와 `volume`(min 0, max 463M)이 Xavier-초기화된 선형층을 즉시 saturation. 그래디언트 흐름 차단.
- **수정**: train-split 통계 기준 **z-score per-feature normalize** 추가 (`CausalStockDataset.compute_feature_stats`)
- **결과**: 정상 학습 복귀. 위 §2.3 수치는 모두 fix 적용 후.

이 처리는 논문이 명시하지 않은 implicit 가정이므로 [reproduction-questions.md](reproduction-questions.md)에 별도 기록 권장.

### 2.5 게이트 진행도 (재구현 verification cadence)

| 게이트 | 내용 | 상태 |
|---|---|---|
| **G1** 데이터 패리티 | 88 stocks × 거래일 수가 논문 Table C.1과 일치? | ✅ ACL18 loader 통과 |
| **G2** 포워드 패스 | 입력 shape → 출력 shape, 파라미터 수 sanity | ✅ unit tests (MIE/TCD/FCM) |
| **G3** 손실 sanity | 단일 배치 loss 감소, gradient 정상 | ✅ trainer smoke test |
| **G4** Tiny overfit | 소규모 데이터에 overfit 가능? | ✅ sanity sweep에서 lr↑ + bernoulli로 0.72 달성 (모델 표현력 확인) |
| **G5** Full reproduction | 10 seeds 평균이 0.6342 ± 0.005? | ⏳ **진행 직전** — 1 seed 결과 0.623, 본격 sweep 대기 |

### 2.6 Phase 10 budget (sanity report 기반)

- 풀 universe(85 stocks) × 100 epoch × 1 seed ≈ **5–6분** (CPU)
- 10 seeds × 5 configs = **약 5시간 CPU**
- 실제 GPT-3.5 DNE 점수화: [scripts/score_news.py](../scripts/score_news.py)로 별도 캐싱(비용 별도)

### 2.7 다음에 할 일

순서대로:

1. **Real GPT-3.5 DNE 캐시 완성**: 현재 mock으로 학습 중. 진짜 점수 캐시 만들고 full 재학습 → no_news와 full 사이 +5%p gap 재현 확인
2. **w/o lag-dep 조사**: 논문 ≈0.59인데 우리 0.504. 구현 차이 또는 hyperparameter 문제 가능성. [paper-checker](../.claude/agents/paper-checker.md) agent로 TCD 모듈 비교 검증
3. **10 seeds × 100 epochs sweep** (G5)
4. **평균±std로 논문 표와 비교** → [experiment-analyst](../.claude/agents/experiment-analyst.md) agent로 gap 분석
5. 재현 성공 시 → Phase 2(한국 시장 + 재벌 prior) 진입

### 2.8 외부 자산 (git에 미포함)

- `reference_data/` (3GB): StockNet / CMIN / Adv-ALSTM 원본
- `paper/`: PDF 원문
- `CausalStock_code/`: 저자 공개 코드 (참고용, leaked API key 포함이라 별도 격리)
- `.venv/`, `.env` (OPENAI_API_KEY)
- `data/processed/dne_acl18.parquet`: 점수 캐시 (작아서 git 포함 가능)

`.gitignore` 정책은 [.gitignore](../.gitignore) 참조.

---

## 한 줄 현황

> 논문의 3개 모듈(MIE/TCD/FCM) + ELBO+BCE 학습 + ACL18 데이터 파이프라인 + 61개 단위 테스트 + 5개 ablation 1-seed 실행까지 완료. 핵심 normalize 버그 1건 해결. Full ACL18 1-seed test ACC **0.623** (논문 0.634 대비 −1.1%p). 다음은 GPT-3.5 실제 점수 캐시 → 10 seeds sweep → 재현 성공 판정.
