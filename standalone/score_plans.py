"""Step 2 (GBJOv2 venv): score the deployed plans dumped by
proxy_decode_compare.py with the QLever true-cost oracle. Reports mean log10 C*
+ catastrophes per (model, decode) and the head-to-head deltas.

    cd ~/Projects/GBJOv2 && PYTHONPATH=standalone uv run python -u \
      standalone/score_plans.py
"""
import json
import os

import numpy as np

from finetune_mrt import CStarOracle, CENSORED

PLANS = "standalone/val100_plans.json"
CACHE = "standalone/cstar_cache.json"
ORDER = ["old-det", "new-det", "old-rich8", "new-rich8",
         "old-rich32", "new-rich32"]


def main():
    d = json.load(open(PLANS))
    triples = [[tuple(t) for t in q] for q in d["triples"]]
    oracle = CStarOracle("http://127.0.0.1:7020/", 10.0, CACHE)
    L = {}
    cat = {}
    for n in ORDER:
        c = np.array([oracle.c_out(order, tr)
                      for order, tr in zip(d["plans"][n], triples)])
        L[n] = np.log10(np.maximum(c, 1.0))
        cat[n] = int((c >= CENSORED).sum())
    oracle.save()

    print(f"\n{len(triples)} val queries -- true C* of the DEPLOYED plan:")
    print(f"{'config':>9} | {'mean log10 C*':>13} | {'median':>7} | "
          f"{'catastrophes':>12}")
    print("-" * 52)
    for n in ORDER:
        print(f"{n:>9} | {L[n].mean():>13.3f} | {np.median(L[n]):>7.3f} | "
              f"{cat[n]:>12}")

    def head(a, b):
        dd = L[a] - L[b]
        return int((dd < -1e-6).sum()), int((dd > 1e-6).sum()), float(dd.mean())

    for a, b in [("new-det", "old-det"), ("new-rich8", "old-rich8"),
                 ("new-rich32", "old-rich32"), ("new-rich32", "new-rich8")]:
        bw, ws, mn = head(a, b)
        print(f"{a:>9} vs {b:<9}: {a} cheaper on {bw:>3} / costlier on {ws:>3}"
              f" / mean(a-b) {mn:+.3f} OOM  "
              f"({'A cheaper' if mn < 0 else 'B cheaper'})")


if __name__ == "__main__":
    main()
