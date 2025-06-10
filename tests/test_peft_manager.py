import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from model import AdaptiveMambaModel
from train import PEFTManager, TrainingConfig


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_peft_adds_trainable_params():
    config = TrainingConfig(enable_peft=True, enable_ia3=False)
    manager = PEFTManager(config)
    model = AdaptiveMambaModel(vocab_size=10, d_model=8, n_layers=1,
                               block_config={"enable_masking": False})

    before = count_trainable(model)
    peft_model, new_params = manager.apply_peft_to_model(model)
    after = count_trainable(peft_model)

    assert len(new_params) > 0
    assert after > before
