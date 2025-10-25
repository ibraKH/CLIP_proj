#!/usr/bin/env bash
# Run ablation study for Tip-Adapter++

python -m src.train.ablation_tipadapter_plus --config configs/tipadapter_plus.yaml --output reports/tables
