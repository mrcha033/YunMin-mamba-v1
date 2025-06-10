import pytest

torch = pytest.importorskip("torch")

from model import FallbackMamba


def test_fallback_mamba_state_update():
    m = FallbackMamba(d_model=4, d_state=3, d_conv=2)
    x = torch.randn(2, 5, 4)
    init_state = m.ssm_state.clone()
    out = m(x)
    assert out.shape == (2, 5, 4)
    assert not torch.allclose(m.ssm_state, init_state)
