# CausalStock Reproduction — Master Plan & Progress Tracker

> **이 문서의 목적**: Phase 1 reproduction(원논문 재현) 완료까지 해야 할 모든
> 작업·검증·결정을 한곳에 추적하는 *살아있는 문서*. 작업 시작/완료/이슈 발생
> 시 이 문서를 먼저 갱신한 뒤 코드 변경한다.
>
> **Reproduction 대상**: Li et al., *CausalStock*, NeurIPS 2024 (arXiv:2411.06391).
> Paper Tables 1, 2, 4, 5 + Figure 4의 모든 정량 결과를 ±tolerance 안에 재현.
>
> **마지막 갱신**: 2026-05-13

---

## 📑 목차

1. [업데이트 규칙 (반드시 읽기)](#0-업데이트-규칙)
2. [진척도 대시보드](#1-진척도-대시보드)
3. [완료된 작업](#2-완료된-작업)
4. [진행 중인 작업](#3-진행-중인-작업)
5. [대기 중인 작업 — 우선순위 순](#4-대기-중인-작업)
6. [검증 상태](#5-검증-상태)
7. [지속 작업 (Continuous)](#6-지속-작업)
8. [학교PC 이전 후 작업](#7-학교pc-이전-후-작업)
9. [위험 요소 / Open Questions](#8-위험-요소--open-questions)
10. [마일스톤 & 일정](#9-마일스톤--일정)
11. [업데이트 로그](#10-업데이트-로그)

---

## 0. 업데이트 규칙

**작업 시작할 때마다:**
1. 해당 항목을 `⏸ pending` → `🟡 in_progress`로 변경
2. "진행 중인 작업" 섹션에 한 줄 추가 (시작 날짜 + 책임자)

**작업 완료할 때마다:**
1. `🟡` → `✅` + 완료 날짜
2. 산출물 경로 명시 (`src/data/kdd17.py`, `experiments/results/...` 등)
3. 새 unit test 추가했으면 카운트 갱신 (현재 57)
4. 결과가 수치라면 "검증 상태" 표에 기록
5. "업데이트 로그"에 한 줄 추가

**새 발견·이슈 생길 때마다:**
- 가정 깨짐 / paper와 다른 점 → `docs/reproduction-questions.md`의 I.x로 추가, 여기 "위험 요소"에 링크
- 측정값(시간, 비용, 처리량) → "마일스톤" 일정 재검토

**커밋 메시지에 본 문서 변경 포함하기:**
- 작업 항목 완료마다 본 문서도 같이 업데이트하고 함께 commit

---

## 1. 진척도 대시보드

| Phase | 상태 | 완료율 | 다음 액션 |
|---|---|---|---|
| Phase 0  Scaffold | ✅ | 100% | — |
| Phase 1  ACL18 loader | ✅ | 100% | — |
| Phase 2  MIE | ✅ | 100% | — |
| Phase 3a DNE mock | ✅ | 100% | — |
| Phase 3b DNE GPT code | ✅ | 100% | — |
| **Phase 3b DNE 실제 점수화** | 🟡 | **0.4%** (200/44,625) | 학교PC C1 |
| Phase 4  TCD | ✅ | 100% | — |
| Phase 5  FCM | ✅ | 100% | — |
| Phase 6  Loss (ELBO+BCE) | ✅ | 100% | — |
| Phase 7  Trainer | ✅ | 100% | — |
| Phase 8  Metrics | ✅ | 100% | — |
| Phase 9  Ablation harness | ✅ | 100% | — |
| Phase 9.5 normalization fix | ✅ | 100% | — |
| Phase Transfer (git push) | ✅ | 100% | — |
| **Phase 10 ACL18 reproduction** | ⏸ | 0% | 학교PC C2 |
| **Phase 11 5-dataset 일반화** | ⏸ | 0% (어댑터 미작성) | 노트북 A1 |
| **보조: Tables 4/5, Figure 4** | ⏸ | 0% | 노트북 A2 |
| **최종 reproduction report** | ⏸ | 0% | M5 |

**현재 시점 핵심 차단점:**
- (a) **Phase 11 어댑터 코드 미작성** — 노트북에서 즉시 시작 가능
- (b) **Phase 3b 점수화 미완료** — 학교PC 야간 진행 예정
- (c) **device='auto' on CUDA 검증 안 됨** — 학교PC 첫 셋업 시 확인

---

## 2. 완료된 작업

### 2.1 코드 (Phase 0–9)

산출물 — `src/`, `tests/` (57 tests pass), `experiments/configs/`, `requirements.txt`.

| 모듈 | 파일 | unit tests | 핵심 검증 |
|---|---|---|---|
| Seed/Config | `src/utils/{seed,config,logging}.py` | 3 | RNG 재현성 |
| Data loader | `src/data/{stocknet,acl18,dataset}.py` | 5 | shape, 누설, 85 stocks |
| DNE mock + cache | `src/data/{dne_mock,dne_cache}.py` | 5 | 결정성, parquet roundtrip |
| DNE GPT (sync+async) | `src/data/{dne_gpt,dne_gpt_async}.py` | 8 | prompt 파싱, 인터넷 끊김 안전 |
| MIE | `src/models/mie.py` | 4 | shape, gradient flow |
| TCD (lag-dep) | `src/models/tcd.py` | 6 | Gumbel diff, l=1 boundary |
| FCM | `src/models/fcm.py` | 5 | **sparsity (Eq.9)**, per-node 독립 |
| CausalStock model | `src/models/causalstock.py` | (smoke) | 통합 |
| Loss (ELBO+BCE) | `src/training/loss.py` | 7 | finite, toy overfit 감소 |
| Trainer | `src/training/trainer.py` | (smoke) | 1-epoch 완주 |
| Metrics | `src/evaluation/{classification,trading}.py` | 9 | **MCC=sklearn**, APV, Sharpe |
| Ablation flags | (configs) | 4 | flag → 모델 정확히 변경 |

### 2.2 Phase 9.5 — Sanity & Bug fix

- **발견**: `volume` (~10⁷), `feat4` (std ~3.15) feature가 z-score 없이 들어가 Xavier-init linear 폭주 → 모든 config가 chance level
- **수정**: `CausalStockDataset.compute_feature_stats` 추가, train-split 통계로 z-score 정규화
- **검증 (full 85 stocks, 30 epoch, 1 seed)**:
  | Config | test ACC | test MCC |
  |---|---|---|
  | paper-exact (price-only, lr=1e-5, gaussian) | **0.560** | 0.123 |
  | price-only, lr=1e-3, bernoulli | 0.719 | 0.438 |
  | full (mock DNE), lr=1e-3 | 0.717 | 0.432 |
- **결론**: paper-exact가 paper's no-news 0.581에 근접. 아키텍처 동작 확인.
- 산출물: `experiments/results/sanity_5epoch.md`, `experiments/results/sanity_sweep.json`

### 2.3 Phase 3b — DNE GPT scoring infrastructure

- Sync `GPTDNEScorer` + Async `AsyncGPTDNEScorer` (concurrency-limited semaphore)
- 기본 모델 `gpt-5.4-mini` (cost/speed 우월, paper-deviation I.6 기록)
- Paper-faithful prompt: `str(token_list)` 그대로 prompt에 (token 리스트), 회사명 매핑, 20 news/day 점수화
- **인터넷 끊김 안전**: `APIConnectionError`/`Timeout` 시 무한 wait + exponential backoff, zero 캐싱 안 함
- 200 pair 시험 점수화 완료 (gpt-5.4-mini, AAPL):
  | 차원 | 범위 (paper) | 우리 결과 (mean ± std) |
  |---|---|---|
  | Correlation | [0, 10] | +6.64 ± 2.25 |
  | Sentiment | [-1, 1] | +0.13 ± 0.35 |
  | Importance | [0, 10] | +3.63 ± 1.87 |
  | Impact | [0, 10] | +2.72 ± 1.66 |
  | Duration | [0, 10] | +1.97 ± 1.25 |
- 비용/시간 (200건 wall-clock): **8.8분, ~$0.5**
- 산출물: `data/processed/dne_acl18.parquet` (200건), `dne_acl18_gpt35.parquet` (200건 backup)

### 2.4 Transfer 준비 → GitHub push

- `requirements.txt` 모든 의존성 버전 명시
- `SETUP.md` 9-step 학교PC 셋업 가이드
- `scripts/run_phase10.sh`, `run_scoring_overnight.sh`
- `Trainer`의 `device='auto'` (cuda > mps > cpu)
- Repo: <https://github.com/peanutdanny/causalstock-reproduction> (private)
- 최신 commit: `a057944`

---

## 3. 진행 중인 작업

| 항목 | 시작일 | 다음 단계 |
|---|---|---|
| (없음 — 다음 작업 선택 대기) | — | A1 또는 A2 시작 |

---

## 4. 대기 중인 작업

우선순위 순서. 각 항목에 **위치(노트북/학교PC)** 와 **차단(blockers)** 표기.

### A. 노트북에서 — 코드 작업 (우선)

#### A1. Phase 11 어댑터 작성 ⭐⭐⭐
- **위치**: 노트북 (CPU만 필요)
- **차단**: 없음
- **이유**: 학교PC가서 모든 실험을 한꺼번에 돌리려면 이 코드가 먼저 있어야 함
- **세부 작업**:
  - [ ] A1-a. `src/data/kdd17.py` (50 stocks, no news, 11-dim Adv-ALSTM)
  - [ ] A1-b. `src/data/dtml.py` (NI225/FTSE100/CSI300/NDX100, 11-dim)
  - [ ] A1-c. `src/data/cmin.py` (CMIN-US 110 / CMIN-CN 300, with news)
  - [ ] A1-d. `src/data/adv_alstm_features.py` (11-dim 변환 함수, A1-a/b가 의존)
  - [ ] A1-e. 각 dataset config: `experiments/configs/{kdd17,ni225,ftse100,cmin_us,cmin_cn}.yaml`
  - [ ] A1-f. 각 어댑터 unit tests (`tests/test_kdd17_loader.py` 등)
- **예상**: 2-3일 코드. 끝나면 57 → ~75 tests.
- **검증 게이트**: 각 dataset의 `build_*_splits()`가 paper §7.1 표대로 split 크기를 만들고, 첫 sample shape 확인.

#### A2. 보조 runner / analyzer ⭐⭐
- **위치**: 노트북
- **차단**: 없음
- **세부**:
  - [ ] A2-a. `scripts/run_investment_sim.py` — Figure 4 (top-3 portfolio SR/APV)
  - [ ] A2-b. `scripts/run_spearman_corr.py` — Table 5 (causal strength vs market cap)
  - [ ] A2-c. `scripts/run_hyperparam_sweep.py` — Table 4 (lr/L/λ sensitivity)
  - [ ] A2-d. `scripts/aggregate_results.py` — paper Table 1/2 자동 비교 리포트
- **예상**: 1-2일.

#### A3. (선택) paper-checker 감사
- **위치**: 노트북
- **차단**: 없음
- **작업**: `paper-checker` agent로 `src/models/`, `src/training/loss.py`가 paper Eq. 1-18과 정확히 매칭되는지 감사
- **예상**: 30분.
- **언제**: A1 마무리 후, 학교PC로 push 직전.

#### A4. ✅ 노트북에서 minimal 학습 사전 검증 (완료 2026-05-13)
- **위치**: 노트북 (5분)
- **검증 명령**: `python -m experiments.train --config experiments/configs/acl18.yaml --max-epochs 1 --tiny --seed 0`
- **결과**:
  - ✅ train.py가 GPT cache 인식 — 로그 `using GPT DNE cache: data/processed/dne_acl18.parquet (200 entries)`
  - ✅ 학습 정상 동작 (1 epoch 완료, 결과 JSON 저장)
  - 🟡 Cache hit 비율 **4.7%** (935/19,960) — 200건이 AAPL 2013-12-16~2014-10-01만 cover
  - 나머지 95.3%는 mock score로 fallback (의도된 동작)
- **결론**: wiring 100% 정상. GPT 영향력 측정은 C1 완료 후 (~100% cache hit) 다시 검증 필요.

### B. 학교PC에서 — 셋업 (1회)

#### B1. SETUP.md 따라 셋업
- **위치**: 학교PC
- **차단**: A1, A2 완료 후 (또는 ACL18만 검증할 거면 지금도 가능)
- **세부**: SETUP.md 0-6단계 (clone, venv, requirements, reference_data, .env, smoke test, GPU 확인)
- **검증 게이트**:
  - [ ] `pytest tests/ -q` → 57 passed
  - [ ] `torch.cuda.is_available() == True`, `get_device_name(0) == "NVIDIA RTX A6000"`
  - [ ] tiny smoke 1 epoch 완주

### C. 학교PC에서 — 실험 (GPU, 야간/장시간)

#### C1. Phase 3b 점수화 완료 ⭐⭐⭐ (M1 prerequisite)
- **위치**: 학교PC, 야간
- **차단**: B1 완료
- **명령**: `bash scripts/run_scoring_overnight.sh`
- **예상**: 5-10시간, ~$15 (Tier 1)
- **검증 게이트**:
  - [ ] cache size ≥ 19,000 nonzero pairs (paper 학습 범위 거의 모두 cover)
  - [ ] all-zero rate < 10%
  - [ ] 점수 분포 sanity (200건 결과와 유사)

#### C2. Phase 10 — ACL18 reproduction
- **위치**: 학교PC GPU
- **차단**: C1 완료
- **세부**:
  - [ ] C2-a. 1 seed × 5 configs (sanity): `bash scripts/run_phase10.sh 1` (예상 5-30분 GPU)
  - [ ] C2-b. 10 seeds × 5 configs (paper-quality): `bash scripts/run_phase10.sh 10` (예상 1-5h GPU)
- **검증 게이트** (M1):
  | 항목 | paper 목표 | tolerance | 미달 시 |
  |---|---|---|---|
  | ACL18 ACC (full) | 63.42 | ±0.5%p | lr/L/λ sweep, FCM 노이즈 모델 재검토 |
  | ACL18 MCC | 0.2172 | ±0.015 | 동일 |
  | w/o TCD | 51.08 | ±1.0%p | TCD 모듈 의심 |
  | w/o news | 58.10 | ±1.0%p | 정규화/DNE 의심 |
  | w/o lag-dep | 59.19 | ±1.0%p | h_u/h_v 의심 |
  | λ=0 | 58.26 | ±1.0%p | likelihood form 의심 |

#### C3. Phase 11 — 5-dataset 일반화
- **위치**: 학교PC GPU
- **차단**: A1 (어댑터 코드), B1, M1 달성(아키텍처 검증)
- **세부**:
  - [ ] CMIN-US: 점수화 + 학습 (paper 54.64, 10 seeds)
  - [ ] CMIN-CN: 점수화 + 학습 (paper 56.19, 10 seeds, Chinese prompt 검증)
  - [ ] KDD17: 학습만 (paper 56.09, 5 seeds, no news)
  - [ ] NI225: 학습만 (paper 53.01, 5 seeds)
  - [ ] FTSE100: 학습만 (paper 52.88, 5 seeds)
- **예상**: 점수화 +6-10h, 학습 5-10h GPU.

#### C4. 보조 결과 (Tables 4, 5, Figure 4)
- **위치**: 학교PC
- **차단**: A2 코드, C2 결과
- **세부**: A2의 4개 script 실행 + 결과 JSON.

### D. 종합 분석 / 보고서

#### D1. 결과 집계
- **명령**: `python scripts/aggregate_results.py`
- **출력**: paper Table 1/2/4/5 + Figure 4 매칭표

#### D2. Reproduction report 작성
- 산출물: `docs/reproduction-report.md`
- 내용: 매칭 표, 미달 항목 원인, paper-deviation 명세, Phase α로의 bridge

---

## 5. 검증 상태

### 5.1 ✅ 검증 완료

| 항목 | 검증 방식 | 통과 기준 | 결과 |
|---|---|---|---|
| Unit tests | pytest | 모두 PASS | **57/57** ✅ |
| RNG 재현성 | seed 두 번 → 동일 | bit-exact | ✅ |
| Tiny smoke 1 epoch | trainer 1 epoch + checkpoint | 무에러 + 체크포인트 저장 | ✅ |
| Full 85 stocks × 30 epoch | 4가지 setting 비교 | finite, paper-exact ≈ 0.56 | ✅ |
| GPT 200건 점수 sanity | 범위/분포 | 범위 위반 0 | ✅ |
| **GPT cache → train.py wiring** | tiny 1 epoch (A4) | 로그 `using GPT DNE cache (200 entries)` 출력 + 학습 완주 | ✅ (cache hit 4.7%, 나머지 mock fallback) |
| Async client | fake client 단위 | 4 tests pass | ✅ |
| 인터넷 끊김 처리 | 코드 로직 | 무한 wait + zero 안 캐싱 | ✅ (코드 검증, 실측 미실시) |
| FCM sparsity (Eq. 9) | gradient mask | G=0 이면 ∂f/∂X=0 | ✅ |
| MCC vs sklearn | 200 random sample | 일치 | ✅ |
| Ablation flag wiring | 4 configs | 각 flag 정확 적용 | ✅ |

### 5.2 ⏸ 검증 필요 / 검증 대기

| 항목 | 검증 방법 | 차단 |
|---|---|---|
| **paper Table 1 ACL18 (63.42)** | C2-b (10 seeds × 100 epoch) | C1 완료 |
| **paper Table 2 ablations** | C2-b 5 configs 결과 비교 | C1 완료 |
| **paper Table 1 1b (KDD17/NI225/FTSE100/CMIN)** | C3 학습 + 비교 | A1, C1 |
| **paper Table 4 (sensitivity)** | A2-c sweep + 결과 | C2 |
| **paper Table 5 (Spearman corr)** | A2-b + ACL18 trained model | C2 |
| **paper Figure 4 (SR/APV)** | A2-a + trained model | C2 |
| `device='auto'` on CUDA | B1 첫 실행 | 학교PC |
| `run_phase10.sh` 셸 실제 실행 | B1 직후 | 학교PC |
| **GPT cache → train.py wiring (대규모 cache)** | C2-a 이후 cache hit ≈100% 인 상태로 재검증 | C1 완료 |
| Phase 11 어댑터 (코드 미작성) | A1 + unit tests | A1 진행 |

### 5.3 🔁 매 변경 시 재실행할 검증

- [ ] `pytest tests/ -q` — push 전 항상
- [ ] `tests/test_trainer_smoke.py` — Trainer 변경 시
- [ ] `tests/test_fcm.py::test_sparsity_respected` — FCM 변경 시
- [ ] `tests/test_loss.py::test_loss_decreases_on_toy_overfit` — Loss 변경 시
- [ ] `experiments/results/sanity_5epoch.md`의 30-epoch sweep — 정규화/loss 형태 변경 시

---

## 6. 지속 작업

### 6.1 매 commit 전
- [ ] `pytest tests/ -q` — 57/57 PASS
- [ ] 본 문서 업데이트 ("진행 중" → "완료" 이동, 날짜 기록)
- [ ] `docs/reproduction-questions.md` — 새 deviation/가정 발견 시 I.x 추가

### 6.2 주간
- [ ] 마일스톤 일정 재검토
- [ ] OpenAI 사용량 / 비용 확인
- [ ] 학교PC와 노트북 git sync (push/pull)

### 6.3 매 실험 후
- [ ] `experiments/results/*.json` 결과 git commit
- [ ] 본 문서의 "검증 상태" 표에 결과 기록
- [ ] paper 수치와 비교 → tolerance 안이면 ✅, 밖이면 위험 요소에 추가

---

## 7. 학교PC 이전 후 작업

학교PC가서 따라야 할 단계는 `SETUP.md` 참조. 본 문서에서는 **학교PC 전용 작업**만 정리:

### 7.1 첫 셋업 (1회, 30분-1시간)
1. GitHub auth (PAT or SSH key)
2. `git clone https://github.com/peanutdanny/causalstock-reproduction.git`
3. venv + `pip install -r requirements.txt`
4. `pytest tests/ -q` → 57 passed
5. `reference_data` clone (ACL18 stocknet-dataset 먼저)
6. `cp .env.example .env` + key 입력
7. `python -c "import torch; print(torch.cuda.is_available())"` → True
8. Smoke test: `python -m experiments.train --config experiments/configs/acl18.yaml --max-epochs 1 --tiny`

### 7.2 일상 작업
- **GPT 점수화** (C1): `bash scripts/run_scoring_overnight.sh` — 야간 5-10h
- **Phase 10 학습** (C2): `bash scripts/run_phase10.sh 10`
- **Phase 11 학습** (C3): A1 완료 후 `bash scripts/run_phase11.sh` (작성 예정)
- **결과 보기**: `cat experiments/results/*.json | python -m json.tool`

### 7.3 학교PC ↔ 노트북 동기화
```bash
# 학교PC에서 작업 후
git add -A && git commit -m "..." && git push
# 노트북에서 받기
git pull
```
- `.env`, `reference_data/`는 양쪽 모두 별도 셋업 (gitignored)
- DNE cache (`data/processed/dne_*.parquet`)는 git에 들어가므로 자동 동기화됨

---

## 8. 위험 요소 / Open Questions

본 문서가 인식하는 reproduction 위험 요소. 새 발견 시 추가.

| ID | 위험 | 영향 | 대응 |
|---|---|---|---|
| R-1 | gpt-5.4-mini 사용 (paper는 gpt-3.5) | Table 1 ACL18 ±0~+1%p 예상 | paper-deviation I.6에 기록, 보조로 gpt-3.5 결과도 비교 가능 |
| R-2 | 85 stocks (paper "88") | 결과 약간 다를 수 있음 | I.1에 기록, full coverage stocks만 사용 |
| R-3 | FCM 노이즈 모델 해석 (gaussian vs bernoulli) | likelihood term 다름 | I.3, `likelihood_form` config로 둘 다 지원 |
| R-4 | q_φ(G\|X) input-independent 가정 | 결과 다를 수 있음 | I.2, U/V free param으로 구현. 미달 시 amortized variant |
| R-5 | OpenAI Tier 1 한도 | 점수화 ~16h | Tier 3 결제 시 ~1h. 현재는 Tier 1 가정 |
| R-6 | NI225/FTSE100 데이터 (snu data) | 다운로드 가능성 | https://datalab.snu.ac.kr/dtml 페이지 확인 필요 |
| R-7 | CMIN-CN Chinese prompt 작동 | 점수 품질 미확인 | C3 진입 시 50건 테스트 후 본격 |
| R-8 | API 키 노출 (대화 기록) | 보안 | 회전 권고. 현재 사용자가 그대로 진행 결정 |

---

## 9. 마일스톤 & 일정

| M# | 정의 | 차단 | 예상 |
|---|---|---|---|
| **M0** | 코드 + 57 tests pass + GitHub push | — | ✅ 2026-05-13 |
| **M1** | ACL18 ACC ≥ 62.9% (10 seed) | C1, C2-b | 학교 이전 + 1-2일 |
| **M2** | Phase 11 5 어댑터 완성 + 75+ tests | A1 | 2-3일 코드 |
| **M3** | Phase 11 5 dataset 학습 완료 | M1, M2, C3 | M1 + 1-2일 |
| **M4** | Tables 4/5 + Figure 4 재현 | A2, M1 | M3 + 1일 |
| **M5** | Reproduction report 발행 | M1-M4 | M4 + 1일 |

**현실적 totals**: 코드 작업 3-5일 (노트북) + 셋업 1시간 + 점수화 1야간 + 학습 1-2일 (학교PC) + 분석 1일 = **약 1-2주**.

---

## 10. 업데이트 로그

| 날짜 | 변경 | by |
|---|---|---|
| 2026-05-13 | 초기 작성. Phase 0-9 ✅, Phase 9.5 ✅, Phase 3b 200건 ✅, GitHub push ✅. M0 달성. | Claude + djhwang |
| 2026-05-13 | A4 ✅ GPT cache → train.py wiring 검증 완료. cache hit 4.7% (tiny config), wiring 정상. 대규모 cache 검증은 C1 완료 후. | Claude |

---

## 부록 A — 핵심 산출물 위치

```
src/                           # 코드
├── data/                      # 데이터 로더 + DNE
├── models/                    # MIE, TCD, FCM, CausalStock
├── training/                  # Loss, Trainer
├── evaluation/                # ACC, MCC, APV, Sharpe
└── utils/                     # seed, config, logging

experiments/
├── configs/                   # acl18.yaml + 4 ablations
├── results/                   # JSON 결과 + sanity report
└── train.py                   # 진입점

scripts/
├── run_phase10.sh             # Phase 10 자동화
├── run_scoring_overnight.sh   # Phase 3b 자동화
├── score_news.py              # sync 점수화
├── score_news_async.py        # async 점수화 (recommended)
└── sanity_sweep.py            # Phase 9.5 검증용

tests/                         # 57 unit tests
docs/
├── paper-summary.md           # 원논문 요약
├── expected-results.md        # paper 수치 + tolerance
├── reproduction-questions.md  # paper-ambiguity + 가정
└── reproduction-roadmap.md    # ← 본 문서

data/processed/
└── dne_*.parquet              # GPT 점수 캐시 (git 추적)

reference_data/                # 3GB raw, gitignored
.env                           # API key, gitignored
```

## 부록 B — paper 수치 빠른 참조

원본: `docs/expected-results.md`. 주요 수치만:

| Dataset | ACC | MCC | std runs |
|---|---|---|---|
| ACL18 (with news) | **63.42** | 0.2172 | 10 |
| CMIN-US | 54.64 | 0.0481 | 10 |
| CMIN-CN | 56.19 | 0.1417 | 10 |
| KDD17 (no news) | 56.09 | 0.1235 | 5 |
| NI225 | 53.01 | 0.0640 | 5 |
| FTSE100 | 52.88 | 0.0534 | 5 |

Ablation (ACL18):
- w/o TCD: 51.08 (chance 근접)
- w/o news: 58.10
- w/o lag-dep: 59.19
- λ=0: 58.26
