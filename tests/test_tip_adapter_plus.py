from __future__ import annotations
import pytest
import torch
from torch import nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tip_adapter_plus import GateMLP, TipAdapterPlus
from src.models.clip_wrapper import CLIPWrapper
from src.models.cache_tools import build_cache, save_cache, load_cache


class TestGateMLP:
    """Unit tests for GateMLP"""

    def test_forward_shape(self):
        """Test that GateMLP outputs correct shapes"""
        B, D = 8, 512
        gate = GateMLP(in_dim=D, hidden=256)

        q = torch.randn(B, D)
        alpha_hat, beta_hat = gate(q)

        assert alpha_hat.shape == (B, 1), f"Expected alpha shape {(B, 1)}, got {alpha_hat.shape}"
        assert beta_hat.shape == (B, 1), f"Expected beta shape {(B, 1)}, got {beta_hat.shape}"

    def test_alpha_range(self):
        """Test that alpha is in (0, 1)"""
        B, D = 16, 512
        gate = GateMLP(in_dim=D, hidden=128)

        q = torch.randn(B, D)
        alpha_hat, _ = gate(q)

        assert torch.all(alpha_hat > 0), "Alpha should be > 0"
        assert torch.all(alpha_hat < 1), "Alpha should be < 1"

    def test_beta_positive(self):
        """Test that beta is > 0"""
        B, D = 16, 512
        gate = GateMLP(in_dim=D, hidden=128)

        q = torch.randn(B, D)
        _, beta_hat = gate(q)

        assert torch.all(beta_hat > 0), "Beta should be > 0"

    def test_no_nans(self):
        """Test that outputs don't contain NaNs"""
        B, D = 8, 512
        gate = GateMLP(in_dim=D, hidden=256)

        q = torch.randn(B, D)
        alpha_hat, beta_hat = gate(q)

        assert not torch.any(torch.isnan(alpha_hat)), "Alpha contains NaNs"
        assert not torch.any(torch.isnan(beta_hat)), "Beta contains NaNs"


class TestTipAdapterPlus:
    """Integration tests for TipAdapterPlus"""

    @pytest.fixture
    def setup_model(self):
        """Setup a minimal TipAdapterPlus model for testing"""
        # Mock CLIP wrapper (simplified)
        clip_wrapper = type('MockCLIP', (), {
            'device': torch.device('cpu'),
            'logits': lambda self, x: torch.randn(x.shape[0], 10),
            'model': type('MockModel', (), {
                'encode_image': lambda x: torch.randn(x.shape[0], 512)
            })()
        })()

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

        return model, clip_wrapper, keys, values

    def test_forward_shape(self, setup_model):
        """Test that forward pass produces correct shapes"""
        model, _, _, _ = setup_model
        B, C = 4, 10

        images = torch.randn(B, 3, 224, 224)
        logits, alpha_hat, beta_hat = model(images)

        assert logits.shape == (B, C), f"Expected logits shape {(B, C)}, got {logits.shape}"
        assert alpha_hat.shape == (B, 1), f"Expected alpha shape {(B, 1)}, got {alpha_hat.shape}"
        assert beta_hat.shape == (B, 1), f"Expected beta shape {(B, 1)}, got {beta_hat.shape}"

    def test_no_nans(self, setup_model):
        """Test that forward pass doesn't produce NaNs"""
        model, _, _, _ = setup_model
        B = 4

        images = torch.randn(B, 3, 224, 224)
        logits, alpha_hat, beta_hat = model(images)

        assert not torch.any(torch.isnan(logits)), "Logits contain NaNs"
        assert not torch.any(torch.isnan(alpha_hat)), "Alpha contains NaNs"
        assert not torch.any(torch.isnan(beta_hat)), "Beta contains NaNs"

    def test_logits_method(self, setup_model):
        """Test that logits() method works for eval compatibility"""
        model, _, _, _ = setup_model
        B, C = 4, 10

        images = torch.randn(B, 3, 224, 224)
        logits = model.logits(images)

        assert logits.shape == (B, C), f"Expected logits shape {(B, C)}, got {logits.shape}"
        assert not torch.any(torch.isnan(logits)), "Logits contain NaNs"

    def test_reg_loss(self, setup_model):
        """Test regularization loss computation"""
        model, _, _, _ = setup_model
        B = 4

        alpha_hat = torch.rand(B, 1) * 0.5 + 0.25  # [0.25, 0.75]
        beta_hat = torch.rand(B, 1) * 3 + 3  # [3, 6]

        loss = model.reg_loss(alpha_hat, beta_hat)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss is NaN"

    def test_trainable_params(self, setup_model):
        """Test that only gate params are trainable"""
        model, _, _, _ = setup_model

        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

        # All trainable params should be from gate
        for name in trainable_names:
            assert name.startswith("gate."), f"Unexpected trainable param: {name}"


class TestCacheTools:
    """Tests for cache utilities"""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load preserves cache data"""
        Ns, D, C = 20, 512, 5
        keys_orig = torch.randn(Ns, D)
        values_orig = torch.randn(Ns, C)

        cache_path = tmp_path / "test_cache.pt"
        save_cache(cache_path, keys_orig, values_orig)

        keys_loaded, values_loaded = load_cache(cache_path)

        assert torch.allclose(keys_orig, keys_loaded), "Keys don't match after load"
        assert torch.allclose(values_orig, values_loaded), "Values don't match after load"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
