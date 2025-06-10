import pytest

torch = pytest.importorskip("torch")

from layers import variable_scan as vs


def test_update_permutation(monkeypatch):
    d_model = 4
    opt = vs.VariableScanOptimizer(d_model=d_model, update_frequency=2)

    def fake_perm(hs):
        return torch.arange(d_model - 1, -1, -1)

    monkeypatch.setattr(vs, "compute_scan_permutation", fake_perm)

    hidden = torch.randn(1, 1, d_model)
    assert not opt.update_permutation(hidden)
    first = opt.get_permutation().clone()
    assert opt.update_permutation(hidden)
    second = opt.get_permutation()
    assert not torch.equal(first, second)
    assert torch.equal(second, torch.arange(d_model - 1, -1, -1))
