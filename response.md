## Q1: What should you do if the two models have different tokenizers?

The best approach is to choose model pairs that share the same tokenizer, since contrastive decoding requires comparing token-level probabilities for the same tokens. If the tokenizers differ, simply subtracting logits is invalid because token IDs no longer align. In that case, the only correct method is to align in text space: detokenize each candidate from the expert, retokenize it with the amateur’s tokenizer, compute the amateur’s log-probability over the resulting sequence, and then subtract this from the expert’s score. This method is heavier and less efficient, but batching makes it feasible for small candidate sets. In practice, it is strongly preferred to use models with shared tokenizers.

**Takeaway: if tokenizers differ, scores must be computed in text space rather than token ID space.**



## Q2: Do you think contrastive decoding is used in practice?

Contrastive decoding is sometimes used, but it is not widespread in production. It is particularly useful with small or medium-sized models for tasks such as code generation, docstring summarization, or safety-sensitive outputs, where quality matters more than latency. Its adoption is limited because it doubles compute cost, strong single-model decoding baselines often achieve similar results, and serving two models adds operational complexity. As a result, contrastive decoding is mainly applied in research or offline batch generation, with only niche use cases in production.

**Takeaway: CD is valuable for quality-sensitive or research tasks, but most production systems rely on single-model decoding for efficiency.**