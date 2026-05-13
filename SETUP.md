# Setup on a fresh machine (e.g. university Linux server)

Tested on macOS (M1 Pro) and Linux (Ubuntu 22.04, x86_64). Should work on any
machine with Python 3.10+ and ~10 GB disk.

## 1) Clone the repo

```bash
git clone <YOUR_REPO_URL> causalstock-reproduction
cd causalstock-reproduction
```

## 2) Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify (should print "57 passed"):

```bash
pytest tests/ -q
```

(Tests don't need the dataset or API key.)

## 3) API key (only needed for Phase 3b — GPT DNE scoring)

```bash
cp .env.example .env
# edit .env and put your real OPENAI_API_KEY
```

Skip this step if you only want to run training on already-scored cache.

## 4) Bring the raw datasets

The 3 GB `reference_data/` is excluded from git. Choose one:

### Option A — copy from another machine (fastest)

On the source machine (e.g. your laptop):

```bash
tar czf reference_data.tar.gz reference_data/
# transfer reference_data.tar.gz to the new machine via scp/rsync/usb
```

On the new machine:

```bash
tar xzf reference_data.tar.gz   # unpacks to ./reference_data/
```

### Option B — fetch each upstream (recommended for fresh school PC)

```bash
mkdir -p reference_data && cd reference_data

# ACL18 (Phase 10 prerequisite, ACL18 only needs this one)
git clone https://github.com/yumoxu/stocknet-dataset.git stocknet-dataset-master

# Phase 11 datasets (needed only when you start Phase 11)
git clone https://github.com/fulifeng/Adv-ALSTM.git Adv-ALSTM-master
git clone https://github.com/BigRoddy/CMIN-Dataset.git CMIN-Dataset-main

# NI225/FTSE100/CSI300 (DTML data, Phase 11):
# Download from https://datalab.snu.ac.kr/dtml manually and unpack to
# reference_data/snu\ data/ ‐ they are not on GitHub.

cd ..
```

Sanity check after clone:

```bash
ls reference_data/stocknet-dataset-master/price/preprocessed | wc -l
# Expected: 88
ls reference_data/stocknet-dataset-master/tweet/preprocessed | wc -l
# Expected: 87
```

After either option, check:

```bash
ls reference_data/stocknet-dataset-master/price/preprocessed | wc -l
# Should print 88 (ACL18 price files)
```

## 5) DNE score cache

The current repo includes a small DNE cache (200 AAPL pairs scored with
gpt-5.4-mini and gpt-3.5 backup) under `data/processed/`. After `git clone` it
will already be there. To complete the rest of ACL18:

```bash
bash scripts/run_scoring_overnight.sh
# Resumes from the cached 200 pairs. Picks up everything else (~99K calls,
# ~$10-15, 5-10h on Tier 1).
```

## 6) Smoke test the pipeline

```bash
python -m experiments.train \
    --config experiments/configs/acl18.yaml \
    --max-epochs 1 --tiny --seed 0
```

Expected output:
```
train=393 valid=42 test=64 D=8
trainer: epoch=001 train_loss=... val_acc=... elapsed=...s
```

Checkpoint saved under `experiments/checkpoints/acl18/`. If GPU is present,
the trainer logs e.g. `device=cuda` since the config uses `device: auto`.
Verify GPU is actually being used:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected on the school PC: `CUDA available: True / Device: NVIDIA RTX A6000`.

## 7) Phase 3b — full DNE scoring (Phase 10 prerequisite)

```bash
bash scripts/run_scoring_overnight.sh
# watch: tail -f experiments/logs/scoring/scoring_*.log
# stop: pkill -f score_news_async.py
```

- 5–10 hours on Tier 1 OpenAI quota for the full ACL18 range
- Resumable: rerun the same command to pick up after a crash/restart
- Survives terminal close (`nohup`) and macOS sleep (`caffeinate -i`)

## 8) Phase 10 — full ACL18 reproduction (after Phase 3b completes)

```bash
# Full + 4 ablations, single seed each, 100 epochs
bash scripts/run_phase10.sh        # to be written — see roadmap
# Or for now, manually:
for cfg in experiments/configs/acl18.yaml \
           experiments/configs/ablations/no_tcd.yaml \
           experiments/configs/ablations/no_news.yaml \
           experiments/configs/ablations/no_lag_dep.yaml \
           experiments/configs/ablations/lambda_0.yaml; do
    python -m experiments.train --config "$cfg" --seed 0
done
```

Each run is ~5–10 min on CPU for 100 epochs. Results land in
`experiments/results/causalstock_acl18_<config>_seed0.json`.

For paper-quality stats (paper uses 10 seeds for news-driven, 5 seeds for
price-only), loop seeds 0..9.

## 9) Phase 11 — five other datasets (CMIN-US/CN, KDD17, NI225, FTSE100)

Not implemented yet. Adapter code (`src/data/cmin.py`, `src/data/kdd17.py`,
`src/data/dtml.py`) needs to be written first. See
`docs/reproduction-questions.md` for open questions per dataset.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENAI_API_KEY` not set | `.env` missing or wrong name | step 3 |
| `FileNotFoundError: reference_data/...` | dataset not transferred | step 4 |
| Loss stays at chance (~0.50) | new bug? | first try `pytest -q tests/` to see what broke |
| `RateLimitError 429: quota exceeded` | OpenAI billing not active | top up OpenAI account |
| `MPS` errors on Linux | MPS is macOS-only | set `runtime.device: cpu` in config |

## Layout cheatsheet

```
src/           training code (do not modify without re-running tests)
experiments/   configs + train.py entrypoint + result JSONs
scripts/       one-off shell utilities (scoring, sweeps)
tests/         57 unit tests, all must pass
docs/          paper summary, expected results, open questions
data/processed/   DNE score caches (.parquet)
reference_data/   raw datasets (gitignored, 3 GB)
.env              your OpenAI key (gitignored)
```
