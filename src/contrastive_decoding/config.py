from dataclasses import dataclass


@dataclass
class CDConfig:
    """Hyperparameters and runtime options for contrastive decoding"""

    max_new_tokens: int = 128
    alpha: float = 0.1
    top_k: int = 64
    tau_amateur: float = 1.0
    stream: bool = True
    use_chat_template: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.tau_amateur <= 0:
            raise ValueError("tau_amateur must be positive")
