#!/usr/bin/env python3
"""Package a submission ZIP and run the pre-submit checklist.

Usage: pack_zip.py [submission_dir] [zip_path]
Default: /mnt/d/wnn_work/submission/m1 -> /mnt/d/wnn_work/submission/m1.zip
"""
import importlib.util
import sys
import zipfile
from pathlib import Path

SUB = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/wnn_work/submission/m1")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/wnn_work/submission/m1.zip")

print("=== 1. files in submission dir ===")
for p in sorted(SUB.iterdir()):
    if p.is_dir():
        continue
    print(f"  {p.name:20s} {p.stat().st_size:>10d} B")

print("=== 2. build zip (solution.py + *.onnx + MODEL.md) ===")
OUT.unlink(missing_ok=True)
included = ["solution.py"] + [p.name for p in SUB.glob("*.onnx")] + \
           [p.name for p in SUB.glob("MODEL.md")]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name in included:
        z.write(SUB / name, name)
with zipfile.ZipFile(OUT) as z:
    for info in z.infolist():
        print(f"  {info.filename:20s} {info.file_size:>10d} B")

print("=== 3. size check (<= 20 MB) ===")
size = OUT.stat().st_size
print(f"zip size: {size} bytes -> {'OK' if size <= 20 * 1024 * 1024 else 'FAIL: too big'}")

print("=== 4. structure check: solution.py at root ===")
with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
    ok_root = "solution.py" in names and not any(n.startswith("/") for n in names)
    print("root solution.py:", "OK" if ok_root else "FAIL", "| entries:", names)

print("=== 5. import check + contract smoke from the zip itself ===")
sys.path.insert(0, "/mnt/d/wnn_starterpack_partial")
TMP = Path("/mnt/d/wnn_work/tmp/zip_extract")
TMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT) as z:
    z.extractall(TMP)

spec = importlib.util.spec_from_file_location("solution", TMP / "solution.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
model = m.PredictionModel()
print("PredictionModel() constructed OK")

import numpy as np


class DP:
    def __init__(self, seq, step, need, state):
        self.seq_ix, self.step_in_seq, self.need_prediction, self.state = seq, step, need, state


v = model.predict(DP(42, 0, False, np.zeros(112, dtype=np.float32)))
assert v is None, "warm-up must return None"
# feed all steps in order; warm-up steps must return None, required (2,)
for s in range(1, 99):
    assert model.predict(DP(42, s, False, np.zeros(112, np.float32))) is None
v = model.predict(DP(42, 99, True, np.zeros(112, dtype=np.float32)))
assert v.shape == (2,) and np.isfinite(v).all(), "required row must return 2 finite"
print("warm-up None + required (2,) finite OK:", v)

# determinism: fresh model, same rows -> bit-identical
model2 = m.PredictionModel()
seq_rows = [DP(7, 0, False, np.zeros(112, np.float32))] + \
           [DP(7, s, True, np.ones(112, np.float32) * (s % 5) / 5) for s in range(1, 300)]


def run(mod):
    out = []
    for dp in seq_rows:
        r = mod.predict(dp)
        out.append(r if r is None else np.array(r))
    return out


r1, r2 = run(m.PredictionModel()), run(m.PredictionModel())
same = all((a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
           for a, b in zip(r1, r2))
print("two runs bit-identical:", same)
assert same
print("ZIP_CHECKLIST_DONE")
