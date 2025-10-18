echo "=== Running CIFAR-10 Experiments ==="

# Zero-shot baseline
echo "Running zero-shot on CIFAR-10..."
python -m src.train.zeroshot --config configs/cifar10_baseline.yaml

# CoOp
echo "Running CoOp on CIFAR-10..."
python -m src.train.fewshot --config configs/cifar10_coop.yaml

# Tip-Adapter
echo "Running Tip-Adapter on CIFAR-10..."
python -m src.train.fewshot --config configs/cifar10_tipadapter.yaml

echo "=== CIFAR-10 Experiments Complete ==="
