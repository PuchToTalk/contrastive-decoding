import torch

from contrastive_decoding.config import CDConfig
from contrastive_decoding.decoding import contrastive_decode


class DummyTokenizer:
    eos_token_id = 3
    pad_token = "<pad>"

    def __call__(self, prompt, return_tensors="pt", add_special_tokens=True):
        del prompt, return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([[0, 1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        vocab = {
            0: "<bos>",
            1: "prompt",
            2: "answer",
            3: "",
        }
        pieces = [vocab.get(t, "?") for t in token_ids]
        text = "".join(pieces)
        return text.strip() if skip_special_tokens else text


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = ()


class DummyModel:
    def __init__(self, preferred_token: int):
        self.preferred_token = preferred_token

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        del use_cache, past_key_values
        vocab_size = 4
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, vocab_size), -2.0)
        logits[:, -1, self.preferred_token] = 4.0
        return DummyOutput(logits)


def test_contrastive_decode_returns_text():
    tokenizer = DummyTokenizer()
    expert = DummyModel(preferred_token=2)
    amateur = DummyModel(preferred_token=1)

    cfg = CDConfig(max_new_tokens=3, stream=False)
    device = torch.device("cpu")

    result = contrastive_decode(
        amateur=amateur,
        expert=expert,
        tokenizer=tokenizer,
        prompt="Test prompt",
        cfg=cfg,
        device=device,
    )

    assert isinstance(result, str)
    assert result != ""
