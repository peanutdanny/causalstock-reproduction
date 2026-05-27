"""Make `src/` importable in tests without installing the package."""
import sys
from pathlib import Path

# Windows: load pandas/sklearn before pytest collects tests that import torch.
# Otherwise torch loads first and leaves too little C stack for pandas/sklearn's
# deep nested importlib resolution, causing access-violation crashes.
import sklearn  # noqa: F401
import pandas  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
