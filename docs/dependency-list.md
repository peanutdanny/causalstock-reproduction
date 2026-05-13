# CausalStock 재현 구현 — 의존성 목록

> Paper α (한국형 CausalStock) 구현을 위한 Python 패키지 목록. 원논문은 PyTorch 기반, Adam optimizer, 4× NVIDIA Tesla V100 환경 (Appendix C.4)을 사용했다.
>
> **권장 Python 버전**: 3.10 또는 3.11 (3.12는 PyTorch 호환성 이슈 가능). 3.9 이하는 신규 라이브러리 호환성 떨어져 비권장.
>
> **권장 가상환경**: `conda` (CUDA/MKL 의존성 관리 편리) 또는 `uv`/`poetry` (속도). 본 문서는 두 가지 install 방법 모두 제공.

---

## 0. 환경 사양 권장

| 항목 | 권장 사양 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04 / macOS 14+ | macOS 사용 시 일부 CUDA 의존 lib는 제외 |
| Python | 3.10.x | 3.11도 가능 |
| GPU | NVIDIA RTX 4090 / A100 / V100 (32GB 권장) | D=300 (CMIN-CN) 실험은 24GB+ 필요 |
| RAM | 32GB+ | 데이터 + LLM score caching |
| Storage | 100GB+ | 6개 데이터셋 + LLM score cache + checkpoint |
| CUDA | 12.1 (PyTorch 2.3 기준) | macOS는 MPS backend |

---

## 1. Core ML / DL 스택

### 1.1 PyTorch (필수)
```
torch>=2.3.0          # NeurIPS 2024는 2.x 가정
torchvision>=0.18.0   # 이미지 처리는 안 쓰지만 torch 의존성 위해
torchaudio>=2.3.0     # (선택)
```
**설치 (CUDA 12.1)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
**macOS (MPS)**:
```bash
pip install torch torchvision torchaudio
```

### 1.2 PyTorch Lightning (강력 권장)
```
pytorch-lightning>=2.3.0
```
학습 루프 / multi-GPU / checkpoint / logging 일관성을 위해. 본 재현 구현은 Lightning 기반으로 시작 권장.

### 1.3 수치/과학 계산
```
numpy>=1.26
scipy>=1.13
scikit-learn>=1.4    # MCC, train/val/test split
pandas>=2.2
```
- `scikit-learn`: `matthews_corrcoef`, `accuracy_score`, `train_test_split` (단, time series이므로 chronological split 직접 구현).

---

## 2. 그래프 / 인과추론 라이브러리

### 2.1 그래프 처리
```
networkx>=3.3        # causal graph 시각화 / 분석
```

### 2.2 Causal discovery (참조용 baseline, 선택)
```
causalnex>=0.12      # NOTEARS / DYNOTEARS baseline (Paper β/γ에서 활용)
tigramite>=5.2       # PCMCI+ baseline
causal-learn>=0.1.3  # PC algorithm, GES 등 전통 causal discovery
```

본 CausalStock 재현 자체에는 위 라이브러리들이 **불필요**하다 — 모델을 PyTorch로 처음부터 구현. 다만 baseline 비교나 sanity check에는 유용.

### 2.3 Gumbel-Softmax / 카테고리컬 분포
PyTorch 기본에 포함:
- `torch.nn.functional.gumbel_softmax`
- `torch.distributions.RelaxedBernoulli`
- `torch.distributions.Bernoulli`

별도 패키지 없이 구현 가능.

---

## 3. LLM 및 텍스트 처리

### 3.1 OpenAI API (DNE의 GPT-3.5 호출)
```
openai>=1.40.0       # v1.x API (chat.completions)
tiktoken>=0.7.0      # 토큰 카운팅 / 비용 추정
tenacity>=8.2.0      # API 호출 재시도 (rate limit, transient error)
```
**비용 추정 (참고)**:
- ACL18: ~50K 뉴스 × 250 token avg = ~12.5M tokens → gpt-3.5-turbo ~$25
- CMIN-US: ~100K 뉴스 → ~$50
- CMIN-CN: ~500K 뉴스 (중국어 토큰 많음) → ~$250
- **총 LLM 점수화 1회**: $300~500 (한국형 확장 시 한국어 LLM 비교까지 포함하면 더)

### 3.2 Anthropic API (선택 — 비교/한국형 확장 시)
```
anthropic>=0.34.0
```
Claude 3.5 Sonnet으로 DNE 점수화 비교 실험 (Paper α의 추가 contribution).

### 3.3 한국어 LLM (Paper α 한국형 확장)
```
# HyperCLOVA-X (NAVER)
# → API 키 발급 후 HTTP requests 직접 호출. 별도 SDK 없음 (2026.05 기준 변동 가능).

# EXAONE (LG AI Research)
transformers>=4.42.0  # exaone-3.0-7.8b-instruct 등 HuggingFace에서 로드 가능

# Solar (Upstage)
# → Upstage Solar API (https://api.upstage.ai)

# 한국어 OpenAI 호환
# → 위의 openai 라이브러리로 동일 호출 (base_url만 변경)
```

### 3.4 Pretrained model loading (BERT/RoBERTa/FinBERT — baseline 비교용)
```
transformers>=4.42.0
sentencepiece>=0.2.0   # multilingual tokenizer
huggingface-hub>=0.23.0
accelerate>=0.30.0     # 큰 LM 추론 시
```
필요한 baseline 모델:
- `bert-base-multilingual-cased`
- `roberta-base`
- `ProsusAI/finbert`
- `FinGPT/fingpt-mt_llama2-7b_lora` 또는 v3.3 버전
- `meta-llama/Llama-2-7b-chat-hf` (HuggingFace 액세스 신청 필요)

### 3.5 Word embedding (Glove + Bi-GRU baseline 비교)
```
gensim>=4.3.0
```
또는 GloVe 사전학습 vectors 직접 다운로드 (https://nlp.stanford.edu/projects/glove/).

---

## 4. 금융 데이터 수집

### 4.1 미국 시장 (ACL18, CMIN-US, KDD17 재현용)
```
yfinance>=0.2.40       # Yahoo Finance 무료
# 또는 alpha_vantage, financialmodelingprep — 유료 / 무료 quota
```
원논문 데이터는 GitHub repo에서 이미 가공된 형태로 제공되므로 우선 그것을 사용. yfinance는 보강용.

### 4.2 한국 시장 (Paper α 핵심)
```
pykrx>=1.0.45           # KRX 가격 데이터 무료 — 가장 안정적
finance-datareader>=0.9.84   # 보조
beautifulsoup4>=4.12.0  # 네이버 금융 헤드라인 크롤링용
requests>=2.32.0
selenium>=4.20.0        # JS 렌더링 필요한 페이지 (제한적 사용)
lxml>=5.2.0             # BeautifulSoup parser
```

### 4.3 DART OpenAPI (재벌 prior G^p 구성)
```
OpenDartReader>=0.3.0   # DART OpenAPI Python wrapper
dart-fss>=0.4.0         # 재무제표 / 지배구조 정보
```
DART API key 발급 필요 (https://opendart.fss.or.kr/, 무료).

### 4.4 일본/영국/중국 시장 (전체 재현 시)
```
yfinance>=0.2.40        # NI225, FTSE100 일부 커버
investpy>=1.0.8         # (선택, 종종 정지됨)
```
CMIN-CN은 Wind 데이터 (Chinese 금융 단말기 — 유료, 학교 라이선스 확인 필요).

---

## 5. 실험 / 추적

### 5.1 실험 추적 (선택, 강력 권장)
```
wandb>=0.17.0           # 추천 1순위
# 또는
mlflow>=2.14.0
tensorboard>=2.17.0
```

### 5.2 Hyperparameter search
```
optuna>=3.6.0           # Bayesian / TPE 기반 hyperparam search
# 또는 ray[tune]>=2.30.0
```
원논문은 grid search (Appendix C.4). 6개 데이터셋 × 4 lr × 4 L × 4 λ = 384 실험 → 너무 큼. Optuna 권장.

### 5.3 결과 저장 / 검증
```
pyyaml>=6.0             # config 파일
hydra-core>=1.3.0       # config 관리 (선택, 권장)
omegaconf>=2.3.0
```

---

## 6. 시각화 / 보고서

```
matplotlib>=3.9.0
seaborn>=0.13.0         # heatmap (Figure 3b 재현)
plotly>=5.22.0          # 인터랙티브 시각화 (선택)
```

논문 Figure 3 재현:
- (a) market value vs causal strength scatter — `matplotlib`
- (b) causal strength matrix heatmap — `seaborn.heatmap`
- (c) DNE output examples — 그냥 텍스트 박스

Investment simulation (Figure 4 / Table 4):
- `pandas.DataFrame.plot` + `matplotlib`
- 또는 `quantstats>=0.0.62` (Sharpe, drawdown 등 전문 라이브러리)

---

## 7. 코드 품질 / 협업

```
black>=24.0.0
ruff>=0.5.0             # flake8/isort 통합 대체
mypy>=1.10.0            # 타입 체크
pytest>=8.2.0
pytest-cov>=5.0.0
pre-commit>=3.7.0
```

---

## 8. 한 줄 설치 (uv 권장)

`uv`는 pip보다 10–100배 빠른 Python 패키지 매니저 (Astral 개발).

```bash
# uv 설치 (한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 init
uv init causalstock-reproduction
cd causalstock-reproduction

# 핵심 의존성
uv add torch torchvision pytorch-lightning numpy scipy scikit-learn pandas \
       networkx openai tiktoken tenacity transformers sentencepiece \
       huggingface-hub yfinance pykrx OpenDartReader \
       wandb optuna pyyaml hydra-core matplotlib seaborn \
       beautifulsoup4 requests lxml

# 개발 의존성
uv add --dev black ruff mypy pytest pytest-cov pre-commit jupyter
```

또는 conda + pip 조합:

```bash
conda create -n causalstock python=3.10 -y
conda activate causalstock
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install pytorch-lightning numpy scipy scikit-learn pandas networkx \
            openai tiktoken tenacity transformers sentencepiece \
            yfinance pykrx OpenDartReader wandb optuna hydra-core \
            matplotlib seaborn beautifulsoup4 requests lxml
```

---

## 9. requirements.txt (참고용)

```text
# Core ML
torch>=2.3.0
pytorch-lightning>=2.3.0
numpy>=1.26
scipy>=1.13
scikit-learn>=1.4
pandas>=2.2

# Graph / Causal
networkx>=3.3
# (optional baseline) causalnex>=0.12

# LLM / NLP
openai>=1.40.0
tiktoken>=0.7.0
tenacity>=8.2.0
transformers>=4.42.0
sentencepiece>=0.2.0
huggingface-hub>=0.23.0
accelerate>=0.30.0

# Finance data
yfinance>=0.2.40
pykrx>=1.0.45
OpenDartReader>=0.3.0
finance-datareader>=0.9.84

# Web scraping (네이버 금융)
beautifulsoup4>=4.12.0
requests>=2.32.0
lxml>=5.2.0

# Experiment tracking
wandb>=0.17.0
optuna>=3.6.0
hydra-core>=1.3.0
omegaconf>=2.3.0
pyyaml>=6.0

# Visualization
matplotlib>=3.9.0
seaborn>=0.13.0

# Dev
black>=24.0.0
ruff>=0.5.0
mypy>=1.10.0
pytest>=8.2.0
pre-commit>=3.7.0
jupyter>=1.0.0
ipykernel>=6.29.0
```

---

## 10. 외부 자원 (별도 다운로드 / 가입)

| 자원 | 필수? | 비용 | 비고 |
|---|---|---|---|
| OpenAI API key | 필수 (DNE) | ~$300–500 1회 점수화 | platform.openai.com |
| Anthropic API key | 선택 (비교) | ~$100 | console.anthropic.com |
| HuggingFace token | 필수 (Llama2 가중치) | 무료 | huggingface.co |
| DART OpenAPI key | 필수 (Paper α 한국 prior) | 무료 | opendart.fss.or.kr |
| HyperCLOVA-X / EXAONE API | 선택 (Paper α 한국 LLM 비교) | 무료/유료 다양 | NAVER Cloud / LG AI |
| W&B account | 선택 (실험 추적) | 학생 무료 | wandb.ai |
| KRX 데이터 | 필수 | 무료 (`pykrx`) | 별도 가입 불필요 |
| 네이버 금융 헤드라인 | 필수 (Paper α 텍스트) | 무료 | 크롤링 (robots.txt 준수) |

---

## 11. 설치 검증 스크립트

```python
# tools/check_env.py
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()}")

import lightning
print(f"Lightning: {lightning.__version__}")

import openai
print(f"OpenAI SDK: {openai.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import pykrx
print(f"pykrx: {pykrx.__version__ if hasattr(pykrx, '__version__') else 'installed'}")

# 모델 sanity check — Gumbel-Softmax
g = torch.nn.functional.gumbel_softmax(torch.tensor([[1.0, 1.0]]), tau=1.0, hard=False)
print(f"Gumbel-Softmax sanity check: {g}")

print("\n✅ Environment looks good. Proceed to data prep.")
```

---

## 12. 단계별 install 우선순위

**1단계 (1일차)** — 환경 + PyTorch + data prep만 가능한 최소 구성:
```
torch pytorch-lightning numpy pandas scikit-learn matplotlib
yfinance pykrx beautifulsoup4 requests
```

**2단계 (1–2주차)** — DNE 구현 시작:
```
openai tiktoken tenacity transformers
```

**3단계 (2–3주차)** — 본격 실험 / 추적:
```
wandb optuna hydra-core
```

**4단계 (4주차+)** — 한국형 확장 / 베이스라인 비교:
```
OpenDartReader causalnex tigramite anthropic
```

설치 한 번에 다 하지 말고, 모듈 작성 진행에 맞춰 점진적으로 추가하는 것이 디버깅 비용 가장 적다.
