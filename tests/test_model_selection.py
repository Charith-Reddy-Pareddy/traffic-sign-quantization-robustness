import pytest
import torch

from traffic_sign_quantization.models import create_model


@pytest.mark.parametrize("name", ["baseline_cnn", "mobilenet_v3_small"])
def test_create_model_returns_gtsrb_logits(name):
    model = create_model(name).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 32, 32))

    assert logits.shape == (1, 43)
    assert torch.isfinite(logits).all()


def test_create_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="model_name"):
        create_model("unknown")
