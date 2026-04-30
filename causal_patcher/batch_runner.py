"""Batched baselines and metrics for left-padded prompt batches."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
from transformer_lens import HookedTransformer

from causal_patcher.utils import get_left_padded_tokens

NamesFilter = Optional[Union[str, List[str], Callable[[str], bool]]]


class BatchExperimentRunner:
    """Batched clean/corrupt prompts, left-padded, with shared baseline forward passes."""

    def __init__(
        self,
        model: HookedTransformer,
        clean_prompts: list[str],
        corrupt_prompts: list[str],
        clean_answers: list[str],
        corrupt_answers: list[str],
    ) -> None:
        n = len(clean_prompts)
        if not (
            n == len(corrupt_prompts) == len(clean_answers) == len(corrupt_answers)
        ) or n == 0:
            raise ValueError(
                "clean_prompts, corrupt_prompts, clean_answers, and corrupt_answers must be "
                "non-empty and the same length."
            )

        self.model = model
        self.clean_prompts = list(clean_prompts)
        self.corrupt_prompts = list(corrupt_prompts)
        # Left-pad each batch; lengths match within clean / corrupt, may differ between them.
        self.clean_tokens, self.clean_attention_mask = get_left_padded_tokens(
            model, self.clean_prompts
        )
        self.corrupt_tokens, self.corrupt_attention_mask = get_left_padded_tokens(
            model, self.corrupt_prompts
        )

        clean_ids: list[int] = []
        corrupt_ids: list[int] = []
        for c_ans, u_ans in zip(clean_answers, corrupt_answers, strict=True):
            clean_ids.append(int(self.model.to_single_token(c_ans)))
            corrupt_ids.append(int(self.model.to_single_token(u_ans)))
        self.clean_answer_token_ids = torch.tensor(clean_ids, dtype=torch.long)
        self.corrupt_answer_token_ids = torch.tensor(corrupt_ids, dtype=torch.long)

        self.clean_logits: torch.Tensor | None = None
        self.corrupt_logits: torch.Tensor | None = None
        self.clean_cache = None
        self.corrupt_cache = None

    def run_baselines(self, names_filter: NamesFilter = None) -> None:
        """Run ``run_with_cache`` for clean and corrupt token batches; store logits and caches."""
        self.clean_logits, self.clean_cache = self.model.run_with_cache(
            self.clean_tokens,
            names_filter=names_filter,
            return_type="logits",
        )
        self.corrupt_logits, self.corrupt_cache = self.model.run_with_cache(
            self.corrupt_tokens,
            names_filter=names_filter,
            return_type="logits",
        )

    def _compute_logit_diff(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-row ``logit(clean_answer) - logit(corrupt_answer)`` at the last token position.

        Assumes ``logits`` is ``[batch, pos, d_vocab]``.
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits, got shape {tuple(logits.shape)}")
        batch_size = logits.shape[0]
        r = torch.arange(batch_size, device=logits.device, dtype=torch.long)
        ca = self.clean_answer_token_ids.to(logits.device)
        cu = self.corrupt_answer_token_ids.to(logits.device)
        return logits[r, -1, ca] - logits[r, -1, cu]


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
    _m = HookedTransformer(_cfg)

    _c_prompts = [
        "Q: Hi. A:",
        "Q: A medium length question here. A:",
        "Q: x.",
    ]
    _u_prompts = [
        "Q: There. A:",
        "Q: A different and slightly longer phrasing for batch two. A:",
        "Q: y z.",
    ]
    _c_ans = ["a", "b", "c"]
    _u_ans = ["b", "a", "c"]
    for _s in _c_ans + _u_ans:
        _m.to_single_token(_s)  # fail fast if not a single token in this tokenizer

    _runner = BatchExperimentRunner(_m, _c_prompts, _u_prompts, _c_ans, _u_ans)
    _runner.run_baselines()
    assert _runner.clean_logits is not None
    _ld = _runner._compute_logit_diff(_runner.clean_logits)
    print("logit_diff (on clean run):", _ld)
    print("logit_diff shape:", tuple(_ld.shape))
