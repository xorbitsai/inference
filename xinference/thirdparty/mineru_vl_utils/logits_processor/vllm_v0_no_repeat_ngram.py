# Logits Processor for vLLM V0 Engine.


class VllmV0NoRepeatNGramLogitsProcessor:
    """
    Prevents repeating the same n-gram of specified size in the output.
    Inspired by Hugging Face's NoRepeatNGramLogitsProcessor.
    Args:
        no_repeat_ngram_size (int): Size of the n-gram to avoid repeating.
    """

    def __init__(self, no_repeat_ngram_size: int = 100):
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.cached_ngrams = {}

    def __call__(self, past_token_ids: list[int] | tuple[int], logits):
        """
        Applies repetition constraints to the logits before sampling tokens.
        Args:
            past_token_ids (list[int] | tuple[int]): The previously generated token IDs.
            logits (torch.FloatTensor): A tensor of shape (vocab_size,) containing raw token logits.
        """
        if not past_token_ids:
            assert len(self.cached_ngrams) == 0, "Cached n-grams should be empty for the first call."

        # Skip if there are not enough tokens to form an n-gram
        ngram_size = self.no_repeat_ngram_size
        if ngram_size <= 0 or len(past_token_ids) < ngram_size:
            return logits

        # Get the n-gram prefix (all but the last token)
        prev_ngram = tuple(past_token_ids[-ngram_size:-1])
        last_token = past_token_ids[-1]

        # Store this n-gram occurrence
        self.cached_ngrams.setdefault(prev_ngram, []).append(last_token)

        # Get the next-token candidates to ban based on current prefix
        current_prefix = tuple(past_token_ids[-ngram_size + 1 :])
        banned_tokens = self.cached_ngrams.get(current_prefix, [])

        # Set the logits of banned tokens to negative infinity
        for token in banned_tokens:
            logits[token] = -float("inf")

        return logits
