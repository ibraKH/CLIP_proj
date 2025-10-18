#!/usr/bin/env python3
"""
Test dataset configurations to ensure they work properly.
"""
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_dataset_loading(dataset_name):
    """Test loading a specific dataset."""
    try:
        if dataset_name == "cifar10":
            from src.datasets.cifar10 import get_cifar10_loaders
            train_loader, val_loader, test_loader, classes = get_cifar10_loaders(
                root="./data/cifar10",
                batch_size=32,
                num_workers=0,  # Avoid multiprocessing issues
                img_size=224
            )
        elif dataset_name == "caltech101":
            from src.datasets.caltech101 import get_caltech101_loaders
            train_loader, val_loader, test_loader, classes = get_caltech101_loaders(
                root="./data/caltech101",
                batch_size=16,  # Smaller batch size for Caltech-101
                num_workers=0,
                img_size=224
            )
        elif dataset_name == "flowers102":
            from src.datasets.flowers102 import get_flowers102_loaders
            train_loader, val_loader, test_loader, classes = get_flowers102_loaders(
                root="./data/flowers102",
                batch_size=32,
                num_workers=0,
                img_size=224
            )
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            return False
        
        print(f"✅ {dataset_name.upper()} dataset loaded successfully")
        print(f"   - Classes: {len(classes)}")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
        
        # Test loading a batch
        for images, labels in train_loader:
            print(f"   - Batch shape: {images.shape}")
            print(f"   - Labels shape: {labels.shape}")
            print(f"   - Label range: {labels.min().item()}-{labels.max().item()}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test dataset configurations')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'caltech101', 'flowers102'],
                       help='Datasets to test')
    args = parser.parse_args()
    
    print("=== Testing Dataset Configurations ===")
    
    success_count = 0
    total_count = len(args.datasets)
    
    for dataset in args.datasets:
        print(f"\n--- Testing {dataset.upper()} ---")
        if test_dataset_loading(dataset):
            success_count += 1
    
    print(f"\n=== Test Results ===")
    print(f"Successfully loaded: {success_count}/{total_count} datasets")
    
    if success_count == total_count:
        print("✅ All datasets working correctly!")
        print("🚀 Ready to run experiments on additional datasets!")
    else:
        print("⚠️ Some datasets have issues - check error messages above")

if __name__ == "__main__":
    main()
