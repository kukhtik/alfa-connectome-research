#!/usr/bin/env python3
"""WNN33 Task 4: evaluation harness for a submission directory.

Scores a submission (solution.py + model file) with the official
ScorerStepByStep on a labeled valid parquet, measures per-row inference time
and extrapolates to the full test size, and checks determinism (two runs
produce bit-identical predictions).

Usage:
  python eval.py --submission /mnt/d/wnn_work/submission/m1 \
                 --valid /mnt/d/wnn_work/data/valid.parquet \
                 --out /mnt/d/wnn_work/eval/m1_score.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/d/wnn_starterpack_partial")
from utils import DataPoint, ScorerStepByStep, FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP  # noqa: E402

TEST_ROWS_ESTIMATE = 37_460_000  # "similar size" to valid per organizers


def load_submission(submission_dir: Path, suffix=""):
    spec = importlib.util.spec_from_file_location(f"submission{suffix}", submission_dir / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel()


def predict_pass(model, valid_path, collect=False):
    """Feed every row through the model, optionally collecting predictions."""
    pf = pq.ParquetFile(valid_path)
    seqs = []
    t0 = time.perf_counter()
    for group in range(pf.num_row_groups):
        table = pf.read_row_group(group, use_threads=False)
        seq_ix = int(table["seq_ix"][0].as_py())
        need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool)
        feats = np.column_stack([table[c].to_numpy(zero_copy_only=False) for c in FEATURE_COLUMNS]).astype(np.float32)
        preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        for step in range(SEQUENCE_LENGTH):
            value = model.predict(DataPoint(seq_ix, step, bool(need[step]), feats[step]))
            if need[step]:
                preds[step] = np.asarray(value, dtype=np.float32)
        seqs.append((seq_ix, preds))
    elapsed = time.perf_counter() - t0
    rows = pf.num_row_groups * SEQUENCE_LENGTH
    return {"elapsed_s": elapsed, "rows": rows, "per_row_us": elapsed / rows * 1e6,
            "sequences": seqs if collect else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-scorer", action="store_true", help="skip full scorer pass (slow)")
    args = ap.parse_args()

    sub_dir = Path(args.submission)
    result = {"submission": str(sub_dir), "valid": args.valid,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # --- official scorer ---
    if not args.skip_scorer:
        model = load_submission(sub_dir)
        t0 = time.perf_counter()
        score = ScorerStepByStep(args.valid).score(model)
        result["scorer"] = score
        result["scorer_wall_s"] = time.perf_counter() - t0
        print(f"[eval] scorer: {json.dumps(score)} ({result['scorer_wall_s']:.1f}s wall)", flush=True)

    # --- deterministic timing pass (pure predict loop) ---
    model = load_submission(sub_dir, suffix="_timing")
    timing = predict_pass(model, args.valid, collect=True)
    result["timing"] = {k: v for k, v in timing.items() if k != "sequences"}
    result["timing"]["extrapolated_full_test_s"] = timing["per_row_us"] * TEST_ROWS_ESTIMATE / 1e6
    result["timing"]["budget_s"] = 60 * 60
    result["timing"]["budget_ok"] = result["timing"]["extrapolated_full_test_s"] <= 40 * 60
    print(f"[eval] timing: {timing['per_row_us']:.1f} us/row -> "
          f"{result['timing']['extrapolated_full_test_s']/60:.1f} min on full test "
          f"(budget {result['timing']['budget_ok']})", flush=True)

    # --- determinism: second full pass, compare bit-exact ---
    model2 = load_submission(sub_dir, suffix="_det")
    timing2 = predict_pass(model2, args.valid, collect=True)
    identical = True
    for (_, p1), (_, p2) in zip(timing["sequences"], timing2["sequences"]):
        if not np.array_equal(p1, p2, equal_nan=True):
            identical = False
            break
    result["determinism_two_runs_identical"] = bool(identical)
    print(f"[eval] determinism: two runs identical = {identical}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)) and not obj:
            return []
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with open(out, "w") as fh:
        json.dump(clean(result), fh, indent=1)
    print(f"[eval] wrote {out}")
    if not identical:
        sys.exit(2)


if __name__ == "__main__":
    main()
