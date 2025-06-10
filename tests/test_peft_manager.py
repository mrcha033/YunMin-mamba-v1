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


def test_importance_based_peft(monkeypatch):
    class DummyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_high = torch.nn.Linear(4, 4)
            self.linear_low = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear_low(self.linear_high(x))

        def get_importance_scores(self, method: str = "mask_probability"):
            return {"linear_high": 0.9, "linear_low": 0.1}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([DummyBlock()])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

    from layers.ia3_layers import IA3Layer

    config = TrainingConfig(
        enable_peft=True,
        enable_ia3=True,
        importance_threshold=0.5,
        peft_application_ratio=1.0,
        peft_r=2,
    )

    manager = PEFTManager(config)
    model = DummyModel()
    peft_model, _ = manager.apply_peft_to_model(model)

    high_layer = peft_model.base_model.blocks[0].linear_high
    low_layer = peft_model.base_model.blocks[0].linear_low

    assert hasattr(high_layer, "lora_A")
    assert isinstance(low_layer, torch.nn.Sequential)
    assert isinstance(low_layer[-1], IA3Layer)
