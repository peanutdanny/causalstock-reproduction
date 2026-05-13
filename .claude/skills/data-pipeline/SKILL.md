---
name: data-pipeline
description: |
  Use when downloading, preprocessing, or validating datasets (ACL18, CMIN-US/CN,
  KDD17, NI225, FTSE100, FNSPID). Triggers on "data", "preprocess", "dataset",
  "stocknet", "FNSPID", "load data".
---

# Data Pipeline Conventions

## Storage
- Raw data: data/raw/{dataset_name}/ — NEVER modify, NEVER commit
- Processed: data/processed/{dataset_name}.pkl or .pt
- Sample (small, committable): data/processed/sample_{dataset_name}.csv

## Download
- Document the download source URL and date in data/README.md
- Use a script (src/data/download_{dataset}.py) when possible — not manual download
- For datasets requiring agreement (Reuters TRC2), document the application status

## Preprocessing
- Each dataset has src/data/preprocess_{dataset}.py
- Output format must be unified: a dict of {ticker, date, price, news_list, label}
- Save train/val/test splits separately with same ra seed across runs

## Validation
- After preprocessing, run src/data/validate_{dataset}.py to check:
  - No NaN in price series
  - News timestamps align with price timestamps
  - Train/val/test splits are non-overlapping
  - Class balance reasonable (50/50 for binary movement)

## ACL18 / StockNet specific
- Time range: 2014-01-01 to 2016-01-01
- 88 tickers across 9 industries
- Labels: up (>0.55%) vs down (<-0.5%), discard rest
- Source: https://github.com/yumoxu/stocknet-dataset
