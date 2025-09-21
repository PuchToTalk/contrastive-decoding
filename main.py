import torch
import torch.nn.functional as F
import transformers as tr

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-1.5B-Instruct"

# Naive first pass: load both models on demand and pick tokens using raw contrastive scores.
tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path).eval()
expert = tr.AutoModelForCausalLM.from_pretrained(expert_path).eval()


def contrastive_decode(prompt: str, max_new_tokens: int = 40, alpha: float = 0.1) -> str:
    """Basic contrastive decoding that filters amateur tokens by expert mass."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits_expert = expert(input_ids).logits[:, -1, :]
            logits_amateur = amateur(input_ids).logits[:, -1, :]

        probs_expert = F.softmax(logits_expert, dim=-1)
        probs_amateur = F.softmax(logits_amateur, dim=-1)

        threshold = alpha * probs_expert.max(dim=-1, keepdim=True).values
        mask = probs_expert >= threshold

        scores = torch.full_like(probs_expert, float("-inf"))
        scores[mask] = torch.log(probs_expert[mask]) - torch.log(probs_amateur[mask])

        next_token_id = torch.argmax(scores, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        eos_id = tokenizer.eos_token_id
        if eos_id is not None and next_token_id.item() == eos_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(\n\tscores,\n\tresults,\n\tkFactor = 4,\n) {\n\tfor (const result of results) {\n\t\tconst { first, second, outcome } = result;\n\t\tconst firstScore = scores[first] ?? 1000;\n\t\tconst secondScore = scores[second] ?? 1000;\n\n\t\tconst expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));\n\t\tconst expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));\n\t\tlet sa = 0.5;\n\t\tif (outcome === 1) {\n\t\t\tsa = 1;\n\t\t} else if (outcome === -1) {\n\t\t\tsa = 0;\n\t\t}\n\t\tscores[first] = firstScore + kFactor * (sa - expectedScoreFirst);\n\t\tscores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);\n\t}\n\treturn scores;\n}\n```"""

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes concise docstrings.",
            },
            {"role": "user", "content": user_message},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    print("\n--- Contrastive Decoding Output ---")
    print(contrastive_decode(prompt, max_new_tokens=40))
