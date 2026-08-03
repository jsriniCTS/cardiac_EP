#!/usr/bin/env bash
# =============================================================================
# run_feature_validation.sh
# Compute the 51-feature set on (a) STACOM ground-truth labels and (b) the
# TRUNet predicted segmentations, both in parallel, then validate pred vs GT.
#
# Edit the paths below, then:  bash run_feature_validation.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GT_LABELS=./data/stacom_labels          # STACOM ground-truth label maps (.nii.gz)
PRED_LABELS=./data/pred_seg             # predict_trunet.py outputs (*_seg.nii.gz)
WORKERS=8                               # parallel worker processes (8 is safe for 512^3)

echo "==> [1/3] Features on GROUND-TRUTH labels (parallel, ${WORKERS} workers)"
python3 extract_features.py \
  --labels-dir "$GT_LABELS" --out ./features_gt \
  --workers "$WORKERS" --ssm --vae --clusters 4

if compgen -G "${PRED_LABELS}/*.nii.gz" > /dev/null; then
  echo "==> [2/3] Features on PREDICTED segmentations (parallel, ${WORKERS} workers)"
  python3 extract_features.py \
    --labels-dir "$PRED_LABELS" --out ./features_pred \
    --workers "$WORKERS" --ssm --vae --clusters 4

  echo "==> [3/3] Validate predicted vs ground truth"
  python3 validate_features.py \
    --gt ./features_gt/features_full.csv \
    --pred ./features_pred/features_full.csv \
    --out ./validation
else
  echo "==> [2/3] No predicted masks in ${PRED_LABELS} yet — skipping pred + validation."
  echo "         Generate them first, e.g.:"
  echo "         for f in ./data/ImageCAS/images/*.img.nii.gz; do"
  echo "           id=\$(basename \"\$f\" .img.nii.gz)"
  echo "           python3 predict_trunet.py --input \"\$f\" \\"
  echo "             --checkpoint ./runs/thor_run1/best_metric_model.pth \\"
  echo "             --trunet-root ./TRUNet-main --num-classes 11 --img-size 128 \\"
  echo "             --output ${PRED_LABELS}/\${id}_seg.nii.gz"
  echo "         done"
fi
echo "Done."
