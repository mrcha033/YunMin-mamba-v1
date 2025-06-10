import pytest
torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from peft import get_peft_model, LoraConfig, TaskType

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def test_lora_parameter_count():
    model = TinyModel()
    total_before, trainable_before = count_parameters(model)

    config = LoraConfig(r=4, lora_alpha=8, target_modules=["linear1", "linear2"],
                        lora_dropout=0.0, bias="none", task_type=TaskType.FEATURE_EXTRACTION)
    lora_model = get_peft_model(model, config)

    total_after, trainable_after = count_parameters(lora_model)

    # LoRA adds additional parameters while freezing the base layers
    assert total_after > total_before
    assert trainable_after < total_after
