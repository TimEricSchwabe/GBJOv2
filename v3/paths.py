"""Single source of truth for filesystem locations in the v3 package.

All paths are anchored to this file (``__file__``), so commands resolve the same
way regardless of the current working directory. Run-artifacts live under
``v3/artifacts/`` (gitignored); the original implementation and training data
stay at the repo root.
"""

from pathlib import Path

V3 = Path(__file__).resolve().parent                       # .../GBJOv2/v3
REPO_ROOT = V3.parent                                      # .../GBJOv2

ARTIFACTS = V3 / "artifacts"
MODELS = ARTIFACTS / "models"
INDEX = ARTIFACTS / "index"
QUERIES = ARTIFACTS / "queries"
PLANS = ARTIFACTS / "plans"
CACHE = ARTIFACTS / "cache"
STATS = ARTIFACTS / "stats"
LOGS = ARTIFACTS / "logs"
LIB = ARTIFACTS / "lib"

DATA = REPO_ROOT / "data"                                  # training data (unmoved)
PACK_ROOT = Path.home() / "rdflib-joinordering" / "gbjo_pack"
