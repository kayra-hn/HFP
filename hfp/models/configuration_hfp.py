from transformers import PretrainedConfig

class HFPConfig(PretrainedConfig):
    model_type = "hfp"

    def __init__(
        self,
        vocab_size=5000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        bulk_dim=128,
        tunnel_depth=3,
        tunnel_decay=0.8,
        dropout_p=0.1,
        short_len=8,
        medium_freq=32,
        long_freq=128,
        medium_momentum=0.1,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.bulk_dim = bulk_dim
        self.tunnel_depth = tunnel_depth
        self.tunnel_decay = tunnel_decay
        self.dropout_p = dropout_p
        self.short_len = short_len
        self.medium_freq = medium_freq
        self.long_freq = long_freq
        self.medium_momentum = medium_momentum
        self.ENABLE_COHERENCE = kwargs.pop("ENABLE_COHERENCE", False)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @classmethod
    def from_1b_profile(cls, vocab_size=50257):
        """
        Creates a 1 Billion Parameter configuration for Cloud Training.
        """
        return cls(
            vocab_size=vocab_size,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=8192,
            bulk_dim=512,
            short_len=64,
            ENABLE_COHERENCE=False
        )
