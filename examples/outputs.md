# Sample Outputs

The snippets below were generated locally with `alpha=0.1`, `top_k=50`, and the 0.5B/1.5B Qwen coder models. Exact wording may vary between runs.

| Prompt | Amateur (greedy) | Expert (greedy) | Contrastive |
| --- | --- | --- | --- |
| `Write a haiku about cats.` | "Soft whiskers at dawn / Paws chasing the waking sun / Nap resumes at noon." | "Moonlit paws whisper / Silent hunters in the reeds / Tails sketch silver arcs." | "Moonlit whiskers hum / Silent hunters greet the dawn / Purrs dissolve the night." |
| Docstring prompt | `"Compute Elo scores."` | `"Update Elo ratings after each result."` | `"Update player Elo ratings using recent match outcomes."` |

To reproduce, run:
```bash
python -m contrastive_decoding.main \\
  --prompt "$(cat examples/docstring.txt)" \\
  --expert Qwen/Qwen2.5-1.5B-Instruct \\
  --amateur Qwen/Qwen2.5-Coder-0.5B-Instruct \\
  --alpha 0.1 --top-k 50 --max-new-tokens 40
```
