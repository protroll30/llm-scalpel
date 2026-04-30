"""Batch tokenization helpers for ``HookedTransformer`` (Phase 2+)."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def get_left_padded_tokens(
    model: HookedTransformer, prompts: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a batch of strings to a common length and return ``input_ids`` + attention mask.

    Configures the model tokenizer for left padding, sets ``pad_token`` to ``eos_token`` if missing,
    then uses :meth:`HookedTransformer.to_tokens` (same BOS / padding rules as the rest of
    TransformerLens). The attention mask is ``1`` on non-padding positions and ``0`` on padding
    (padding uses ``pad_token_id`` on the **left**).

    Returns:
        ``(tokens, attention_mask)`` with shapes ``[batch, pos]``; dtype ``long`` for both.
    """
    tok = model.tokenizer
    if tok is None:
        raise ValueError("Model has no tokenizer; cannot batch-tokenize.")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Explicit left padding; matches tokenizer after the assignments above.
    tokens = model.to_tokens(prompts, padding_side="left")
    attention_mask = (tokens != tok.pad_token_id).to(dtype=torch.long)
    return tokens, attention_mask


if __name__ == "__main__":
    from transformer_lens import HookedTransformerConfig

    _cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=32,
        n_heads=1,
        d_head=32,
        d_mlp=64,
        n_ctx=128,
        d_vocab=50257,
        act_fn="gelu",
        normalization_type="LN",
        default_prepend_bos=False,
        tokenizer_name="gpt2",
    )
    _model = HookedTransformer(_cfg)
    _prompts = [
        "Short.",
        "A bit longer string here.",
        "One",
    ]
    toks, mask = get_left_padded_tokens(_model, _prompts)
    print("tokens shape:", tuple(toks.shape))
    print("attention_mask shape:", tuple(mask.shape))
    print("padding_side:", _model.tokenizer.padding_side, " pad_token_id:", _model.tokenizer.pad_token_id)
    for i, s in enumerate(_prompts):
        print(f"--- example {i} (repr): {s!r}")
        row = toks[i]
        mrow = mask[i]
        for j, (tid, m) in enumerate(zip(row.tolist(), mrow.tolist())):
            dec = _model.tokenizer.decode([int(tid)])
            print(f"  pos {j:2d}  id={tid:5d}  mask={m}  decoded={dec!r}")
