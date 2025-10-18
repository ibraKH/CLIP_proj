echo "=== Running Caltech-101 Experiments ==="

# Zero-shot baseline
echo "Running zero-shot on Caltech-101..."
python -m src.train.zeroshot --config configs/caltech101_baseline.yaml

# CoOp
echo "Running CoOp on Caltech-101..."
python -m src.train.fewshot --config configs/caltech101_coop.yaml

# Tip-Adapter
echo "Running Tip-Adapter on Caltech-101..."
python -m src.train.fewshot --config configs/caltech101_tipadapter.yaml

echo "=== Caltech-101 Experiments Complete ==="
