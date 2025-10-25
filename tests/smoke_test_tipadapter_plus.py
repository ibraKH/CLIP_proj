"""
Smoke test for Tip-Adapter++ - basic functionality check without pytest
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tip_adapter_plus import GateMLP, TipAdapterPlus
from src.models.cache_tools import save_cache, load_cache
import tempfile


def test_gate_mlp():
    """Test GateMLP basic functionality"""
    print("Testing GateMLP...")
    B, D = 8, 512
    gate = GateMLP(in_dim=D, hidden=256)

    q = torch.randn(B, D)
    alpha_hat, beta_hat = gate(q)

    assert alpha_hat.shape == (B, 1), f"Alpha shape mismatch: {alpha_hat.shape}"
    assert beta_hat.shape == (B, 1), f"Beta shape mismatch: {beta_hat.shape}"
    assert torch.all(alpha_hat > 0) and torch.all(alpha_hat < 1), "Alpha not in (0, 1)"
    assert torch.all(beta_hat > 0), "Beta not > 0"
    assert not torch.any(torch.isnan(alpha_hat)), "Alpha has NaNs"
    assert not torch.any(torch.isnan(beta_hat)), "Beta has NaNs"

    print("[PASS] GateMLP passed")


def test_tip_adapter_plus():
    """Test TipAdapterPlus basic functionality"""
    print("Testing TipAdapterPlus...")

    # Mock CLIP wrapper
    class MockCLIP:
        def __init__(self):
            self.device = torch.device('cpu')
            self.model = self

        def logits(self, images):
            return torch.randn(images.shape[0], 10)

        def encode_image(self, images):
            return torch.randn(images.shape[0], 512)

    clip_wrapper = MockCLIP()

    # Mock cache
    Ns, D, C = 40, 512, 10
    keys = torch.randn(Ns, D)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = torch.zeros(Ns, C)
    for i in range(Ns):
        values[i, i % C] = 1.0

    model = TipAdapterPlus(
        clip_wrapper=clip_wrapper,
        keys=keys,
        values=values,
        alpha0=0.5,
        beta0=5.0,
        gate_hidden=128,
        reg=0.05,
        learn_beta=True,
        learn_scale=False,
    )

    # Test forward pass
    B, C_test = 4, 10
    images = torch.randn(B, 3, 224, 224)
    logits, alpha_hat, beta_hat = model(images)

    assert logits.shape == (B, C_test), f"Logits shape mismatch: {logits.shape}"
    assert alpha_hat.shape == (B, 1), f"Alpha shape mismatch: {alpha_hat.shape}"
    assert beta_hat.shape == (B, 1), f"Beta shape mismatch: {beta_hat.shape}"
    assert not torch.any(torch.isnan(logits)), "Logits have NaNs"
    assert not torch.any(torch.isnan(alpha_hat)), "Alpha has NaNs"
    assert not torch.any(torch.isnan(beta_hat)), "Beta has NaNs"

    # Test logits() method for eval compatibility
    logits_only = model.logits(images)
    assert logits_only.shape == (B, C_test), f"Logits method shape mismatch: {logits_only.shape}"

    # Test reg_loss
    loss = model.reg_loss(alpha_hat, beta_hat)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"

    # Test trainable params
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_names:
        assert name.startswith("gate."), f"Unexpected trainable param: {name}"

    print(f"[PASS] TipAdapterPlus passed (trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)})")


def test_cache_tools():
    """Test cache save/load functionality"""
    print("Testing cache tools...")

    Ns, D, C = 20, 512, 5
    keys_orig = torch.randn(Ns, D)
    values_orig = torch.randn(Ns, C)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.pt"
        save_cache(cache_path, keys_orig, values_orig)

        keys_loaded, values_loaded = load_cache(cache_path)

        assert torch.allclose(keys_orig, keys_loaded), "Keys don't match after load"
        assert torch.allclose(values_orig, values_loaded), "Values don't match after load"

    print("[PASS] Cache tools passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Tip-Adapter++ Smoke Tests")
    print("=" * 60)

    try:
        test_gate_mlp()
        test_tip_adapter_plus()
        test_cache_tools()

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAIL] TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
