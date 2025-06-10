import pytest

torch = pytest.importorskip("torch")

from model import AdaptiveMambaModel


def test_model_forward_shape():
    vocab_size = 10
    d_model = 8
    model = AdaptiveMambaModel(vocab_size=vocab_size, d_model=d_model, n_layers=1,
                               block_config={"enable_masking": False})
    input_ids = torch.randint(0, vocab_size, (2, 4))
    output = model(input_ids)
    assert output.shape == (2, 4, vocab_size)
