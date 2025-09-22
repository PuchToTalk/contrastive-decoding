# Contrastive Decoding

A lightweight playground for experimenting with contrastive decoding using paired expert/amateur Hugging Face causal language models.

## Features
- Implemented with KV-cache reuse, plausibility masking, top-k pruning, and amateur temperature control
- (`src/contrastive_decoding`) with reusable helpers and a CLI entry point
- Tests, notebook showing prompts and generated outputs for quick review

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Quick Start (CLI)
From the project root:
```bash
PYTHONPATH=src python -m contrastive_decoding.main \
  --prompt "$(cat examples/docstring.txt)" \
  --expert Qwen/Qwen2.5-3B-Instruct \
  --amateur Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --alpha 0.1 --top-k 50 --max-new-tokens 40

```

or run after installing:

```bash
pip install -e .
python -m contrastive_decoding.main \
  --prompt "$(cat examples/haiku.txt)" \
  --expert Qwen/Qwen2.5-3B-Instruct \
  --amateur Qwen/Qwen2.5-Coder-0.5B-Instruct
```



Key flags:
- `--alpha`: expert probability floor (`alpha * max(p_E)`); lower means more exploratory
- `--top-k` : candidate pruning size; set `0` to consider all plausible tokens
- `--tau-amateur`:  temperature on amateur logits ( >1 softens, <1 sharpens )
- `--no-stream` : disable live token printing and emit the full string at the end
- `--no-chat-template`: feed the raw prompt if your tokenizer lacks chat templates


## Repository Layout
```
contrastive-decoding/
├─ src/contrastive_decoding/   # library + CLI
├─ tests/                      # lightweight pytest smoke test
├─ examples/                   # prompts and sample outputs
├─ notebooks/analysis.ipynb    # multi-prompt comparison playground
├─ requirements.txt
├─ pyproject.toml
├─ README.md
├─ LICENSE
└─ response.md
```

## Examples & Notebook
- `examples/outputs.md` : side-by-side amateur vs expert vs contrastive snippets
- `examples/haiku.txt` & `examples/docstring.txt` : ready-to-run prompts
- `notebooks/analysis.ipynb` : load the models once and iterate over multiple prompts to compare strategies


