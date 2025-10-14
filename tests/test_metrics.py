import torch
from src.eval.metrics import macro_f1_from_logits, expected_calibration_error, TemperatureScaler

def test_macro_f1_simple():
    # 3 samples, 3 classes
    logits = torch.tensor([[5., 1., 0.],
                           [0., 5., 0.],
                           [0., 0., 5.]])
    labels = torch.tensor([0, 1, 2])
    f1 = macro_f1_from_logits(logits, labels)
    assert abs(f1 - 1.0) < 1e-6

def test_ece_temperature():
    torch.manual_seed(0)
    logits = torch.randn(50, 3)
    labels = torch.randint(0, 3, (50,))
    scaler = TemperatureScaler()
    T = scaler.fit(logits, labels)
    assert T > 0
    ece = expected_calibration_error(scaler(logits), labels)
    assert 0.0 <= ece <= 1.0
