echo "=== Running Experiments on All Datasets ==="

# From previous runs, skip datasets that are already completed
echo "Flowers-102 experiments already completed - skipping"

# CIFAR-10
echo "=== Running CIFAR-10 Experiments ==="
python -m src.train.zeroshot --config configs/cifar10_baseline.yaml
python -m src.train.fewshot --config configs/cifar10_coop.yaml
python -m src.train.fewshot --config configs/cifar10_tipadapter.yaml

# Caltech-101
echo "=== Running Caltech-101 Experiments ==="
python -m src.train.zeroshot --config configs/caltech101_baseline.yaml
python -m src.train.fewshot --config configs/caltech101_coop.yaml
python -m src.train.fewshot --config configs/caltech101_tipadapter.yaml

echo "=== All Dataset Experiments Complete ==="
