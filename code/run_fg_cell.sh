#!/bin/bash
# N6-CELL — feature-axis gate (N5 finding: the axis both fresh architecture lines lack).
#
# N5 (literature): TLOB uses dual attention INCLUDING the feature axis; CVML uses
# explicit cross-variate mixing (+244.9% on the mid-price-return task). Both say
# inter-feature mixing is the axis our GRU lacks (it mixes features only once, in
# weight_ih). Variant C closed every output/ensemble/regime axis; this is the first
# ARCHITECTURE cell aimed at that finding.
#
# DESIGN (one variable vs champion):
#   g = sigmoid(W_g @ x_t + b_g)    # 112 -> 96, from the normalized feature row
#   h' = h * g                      # per-unit gate before the head
#   +10,848 params (116546 -> 127394), 3-in/3-out contract preserved.
#
# COST: measured x1.2-1.6 per row on 1-thread CPU -> the ensemble must be a
# 2-member class, NOT 3. Single-cell verdict first.
#
# RECIPE: identical to the champion (mask + corr + std + flat 5 epochs, seed 778),
# one variable added: --feat-gate.
set -euo pipefail

cd /mnt/d/wnn_work
PY=~/wnn_venv/bin/python
DATA=$HOME/fastdata/train_valid_combined.parquet
MASK=$HOME/fastdata/scoring/valid_mask.parquet
STATS=/mnt/d/wnn_work/models/feat_stats.npz
PHAT=/mnt/d/wnn_work/tmp/mask_logit_coeffs.json
TAG=mfg
LOG=/mnt/d/wnn_work/logs/train_${TAG}.log
STATUS=/mnt/d/wnn_work/tmp/RUN_FG_STATUS.txt

for f in "$DATA" "$MASK" "$STATS" "$PHAT"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 3; }
done

if [ -s "models/${TAG}_e4.pt" ]; then
  echo "[skip] $TAG already trained" > "$STATUS"; exit 0
fi

# serial GPU discipline: wait for any other trainer
while pgrep -f "train.py.*--tag m" | grep -v "$TAG" | grep -q .; do
  sleep 30
done

printf 'START %s tag=%s\n' "$(date '+%F %T')" "$TAG" > "$STATUS"

$PY -u train.py \
  --data "$DATA" --out models --tag "$TAG" \
  --hid 96 --layers 2 --seg-len 8000 --batch 8 --epochs 5 \
  --lr 1e-3 --seed 778 --max-seq 12360 --log-every 200 --grad-clip 1.0 \
  --standardize --feat-stats "$STATS" --weighted-loss \
  --corr-loss --corr-mix 0.5 --corr-horizon seq --loss-from-step 99 \
  --row-weight-col is_scored --row-weight-hybrid 10607 \
  --row-weight-mask-file "$MASK" --phat-coeffs "$PHAT" \
  --exact-oversample 2 --prefetch-workers 3 --reader-use-threads \
  --feat-gate \
  >> "$LOG" 2>&1
rc=$?
printf 'DONE rc=%d at %s\n' "$rc" "$(date '+%F %T')" >> "$STATUS"
exit $rc
