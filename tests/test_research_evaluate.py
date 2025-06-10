import pytest

torch = pytest.importorskip("torch")

from research.research_evaluate import evaluate_model_on_task

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        return torch.zeros(batch, seq, self.vocab_size)

    def generate(self, input_ids, **kwargs):
        return torch.zeros((input_ids.size(0), 3), dtype=torch.long)

class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "decoded"

def test_evaluate_language_modeling():
    model = DummyModel()
    batch = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "labels": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    results = evaluate_model_on_task(model, [batch], "language_modeling")
    assert "perplexity" in results


def test_evaluate_other_tasks():
    model = DummyModel()
    tok = DummyTokenizer()

    batch_sum = {"input_ids": torch.ones((1, 2), dtype=torch.long), "target_text": ["a"]}
    sum_res = evaluate_model_on_task(model, [batch_sum], "summarization", tokenizer=tok)
    assert "rouge1_fmeasure" in sum_res

    batch_qa = {"input_ids": torch.ones((1, 2), dtype=torch.long), "answer": ["a"]}
    qa_res = evaluate_model_on_task(model, [batch_qa], "question_answering", tokenizer=tok)
    assert "exact_match" in qa_res

    batch_code = {
        "prompt": "p",
        "test": "assert True",
        "task_id": "t1",
        "input_ids": torch.ones((1, 2), dtype=torch.long),
    }
    code_res = evaluate_model_on_task(model, [batch_code], "code_generation", tokenizer=tok)
    assert "pass_at_1" in code_res

