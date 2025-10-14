import torch
from src.eval.attacks import apply_attacks

def test_text_overlay_changes_pixels():
    x = torch.rand(2, 3, 224, 224)
    y = apply_attacks(x, attack_name="text_overlay", severity=5, text="TEST")
    assert torch.mean(torch.abs(x - y)) > 0.0

def test_pipeline_runs_noise_blur_lowres():
    x = torch.rand(2, 3, 224, 224)
    for atk in ["gaussian_noise", "gaussian_blur", "lowres"]:
        y = apply_attacks(x, attack_name=atk, severity=3)
        assert y.shape == x.shape
