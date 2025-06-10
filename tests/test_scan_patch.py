import pytest
torch = pytest.importorskip("torch")
from scan_patch import (apply_scan_patch, remove_scan_patch,
                        is_scan_patched,
                        create_scan_permutation_from_model)

class DummyMixer(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = DummyMixer()

class DummyModel(torch.nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyLayer() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.mixer(x)
        return x


def test_apply_and_remove_scan_patch(tmp_path):
    model = DummyModel()
    # Copy permutation files to a temporary directory
    scan = tmp_path / "scan_order.npy"
    rev = tmp_path / "scan_order_inv.npy"
    import shutil
    shutil.copy("scan_order.npy", scan)
    shutil.copy("scan_order_inv.npy", rev)

    apply_scan_patch(model, str(scan), str(rev))
    assert is_scan_patched()

    # Ensure forward runs without error
    x = torch.zeros(1, 4, 3)
    out = model.layers[0].mixer.forward(x)
    assert torch.allclose(out, x + 1)

    remove_scan_patch(model)
    assert not is_scan_patched()


def test_create_scan_permutation_from_model():
    model = DummyModel(num_layers=1)
    sample_input = torch.randn(1, 4, 3)
    fwd, rev = create_scan_permutation_from_model(model, sample_input)
    assert len(fwd) == 3
    assert len(rev) == 3
    assert sorted(fwd.tolist()) == [0, 1, 2]
    assert sorted(rev.tolist()) == [0, 1, 2]
    assert (rev[fwd] == list(range(3))).all()
