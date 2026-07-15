"""Process-local compatibility patches for vLLM serving subprocesses."""


def _patch_transformers_tokenizers() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


_patch_transformers_tokenizers()
