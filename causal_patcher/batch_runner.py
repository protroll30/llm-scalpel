"""Batched baselines and metrics for left-padded prompt batches."""

from __future__ import annotations

from typing import Callable, List, Optional, Union, cast, get_args

import torch
from transformer_lens import HookedTransformer

from causal_patcher.runner import _patch_fn, _resolve_index
from causal_patcher.targets import PatchKind, PatchPos, PatchTarget
from causal_patcher.utils import get_left_padded_tokens

NamesFilter = Optional[Union[str, List[str], Callable[[str], bool]]]


def _batch_resolve_patch_pos(
    pos: Optional[PatchPos], p_clean: int, p_corrupt: int
) -> int | slice | tuple[int, int]:
    """Like Phase-1 :func:`_resolve_patch_pos` but *clean* and *corrupt* rows can have different *P*.

    * ``int``: resolved as ``(resolve(int, p_clean), resolve(int, p_corrupt))`` so the last
      prompt token (``-1``) reads/writes the correct time step in each cache.
    * ``tuple[int, int]``: ``(resolve(a, p_clean), resolve(b, p_corrupt))``.
    * ``None`` or ``slice``: use the same on both sides; *requires* ``p_clean == p_corrupt`` so
      the sliced activations are the same shape.
    """
    if pos is None:
        if p_clean != p_corrupt:
            raise ValueError(
                "pos is None (full-sequence copy) but clean and corrupt have different "
                f"lengths: P_clean={p_clean} vs P_corrupt={p_corrupt}."
            )
        return slice(None)
    if isinstance(pos, slice):
        if p_clean != p_corrupt:
            raise ValueError(
                "pos is a slice but clean and corrupt have different lengths: "
                f"P_clean={p_clean} vs P_corrupt={p_corrupt}."
            )
        return pos
    if isinstance(pos, tuple):
        if len(pos) != 2:
            raise ValueError("pos tuple must be (clean_index, corrupt_index)")
        a, b = int(pos[0]), int(pos[1])
        return (_resolve_index(a, p_clean), _resolve_index(b, p_corrupt))
    if isinstance(pos, int):
        return (_resolve_index(pos, p_clean), _resolve_index(pos, p_corrupt))
    raise TypeError(f"Invalid pos spec: {pos!r}")


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
        cl, self.clean_cache = self.model.run_with_cache(
            self.clean_tokens,
            names_filter=names_filter,
            return_type="logits",
        )
        ul, self.corrupt_cache = self.model.run_with_cache(
            self.corrupt_tokens,
            names_filter=names_filter,
            return_type="logits",
        )
        self.clean_logits = cast(torch.Tensor, cl)
        self.corrupt_logits = cast(torch.Tensor, ul)

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

    def _require_baselines(self) -> None:
        if self.clean_cache is None or self.corrupt_tokens is None or self.clean_tokens is None:
            raise RuntimeError("Call run_baselines() first.")

    def patch_clean_into_corrupt(
        self,
        target: PatchTarget,
        *,
        positions: Optional[PatchPos] = None,
    ) -> torch.Tensor:
        """Corrupt forward with a hook that overwrites the target activation from ``clean_cache``.

        Indexing is vectorized: ``src[:, clean_idx, ...]`` and ``[:, corrupt_idx, ...]`` (or
        full-slice) so the patch applies to the whole batch at once.

        Returns:
            Per-row logit-difference (clean vs corrupt answer at last time step) on the **patched**
            corrupt run, shape ``[B]`` (same as :meth:`_compute_logit_diff`).
        """
        self._require_baselines()
        assert self.clean_cache is not None
        assert self.corrupt_tokens is not None
        assert self.clean_tokens is not None

        hook_name = target.hook_name()
        if hook_name not in self.clean_cache:
            raise KeyError(
                f"Clean cache has no entry for {hook_name!r}. "
                "Re-run run_baselines with a names_filter that includes this hook."
            )

        clean_activation = self.clean_cache[hook_name]
        p_clean = int(self.clean_tokens.shape[1])
        p_corrupt = int(self.corrupt_tokens.shape[1])
        if clean_activation.shape[1] != p_clean:
            raise ValueError(
                f"Clean cache {hook_name} pos dim {clean_activation.shape[1]} != clean_tokens {p_clean}."
            )

        effective_pos = target.pos if positions is None else positions
        pos_spec = _batch_resolve_patch_pos(effective_pos, p_clean, p_corrupt)
        hook_fn = _patch_fn(clean_activation, target, pos_spec)
        logits = self.model.run_with_hooks(
            self.corrupt_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )
        return self._compute_logit_diff(cast(torch.Tensor, logits))

    def run_patch_sweep(self, kind: str, pos: PatchPos) -> torch.Tensor:
        """Sweep patch sites of one ``kind`` and record the patched corrupt logit-difference readout.

        For **residual-stream** kinds (``resid_pre``, ``resid_mid``, ``resid_post``) the clean run was
        already cached in :meth:`run_baselines` — that single cache already holds every layer’s
        activations, so the sweep only repeats **corrupt** forward passes (one per layer/head) and
        does not recompute the clean forward.

        Args:
            kind: One of ``"resid_pre"``, ``"resid_mid"``, ``"resid_post"``, ``"mlp_out"``,
                ``"attn_head_z"``.
            pos: Position spec passed to each :class:`PatchTarget` (see Phase-1 /
                :func:`_batch_resolve_patch_pos`).

        Returns:
            If ``kind != "attn_head_z"``: shape ``[n_layers, batch_size]``.
            If ``kind == "attn_head_z"``: shape ``[n_layers, n_heads, batch_size]``.

        Note:
            :meth:`run_baselines` must have been run with a ``names_filter`` that includes the hooks
            you sweep (e.g. ``None`` to cache all), or :meth:`patch_clean_into_corrupt` will raise.
        """
        self._require_baselines()
        assert self.corrupt_tokens is not None
        if self.corrupt_logits is not None:
            dtype = self.corrupt_logits.dtype
        else:
            dtype = torch.float32
        dev = self.corrupt_tokens.device
        batch_size = int(self.corrupt_tokens.shape[0])

        valid: tuple[str, ...] = get_args(PatchKind)
        if kind not in valid:
            raise ValueError(f"kind must be one of {valid}, got {kind!r}")
        k = cast(PatchKind, kind)

        n_layers = int(self.model.cfg.n_layers)
        n_heads = int(self.model.cfg.n_heads)

        with torch.no_grad():
            if k == "attn_head_z":
                out = torch.empty(n_layers, n_heads, batch_size, device=dev, dtype=dtype)
                for layer in range(n_layers):
                    for head in range(n_heads):
                        tgt = PatchTarget("attn_head_z", layer, head=head, pos=pos)
                        out[layer, head] = self.patch_clean_into_corrupt(tgt)
            else:
                out = torch.empty(n_layers, batch_size, device=dev, dtype=dtype)
                for layer in range(n_layers):
                    tgt = PatchTarget(k, layer, pos=pos)
                    out[layer] = self.patch_clean_into_corrupt(tgt)
        return out


if __name__ == "__main__":
    import torch as _torch

    from transformer_lens import HookedTransformerConfig

    _torch.manual_seed(0)

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
    _runner.run_baselines()  # cache all hooks so resid activations are available
    assert _runner.corrupt_logits is not None

    _ld_corrupt = _runner._compute_logit_diff(_runner.corrupt_logits)
    print("logit_diff (corrupt run, pre-patch):", _ld_corrupt)
    print("  shape:", tuple(_ld_corrupt.shape))

    # ``resid_pre`` can leave some rows' logit readout unchanged on tiny random models; ``resid_mid``
    # at the last position reliably moves the first two rows here.
    _t = PatchTarget("resid_mid", 0, pos=-1)
    _ld_patched = _runner.patch_clean_into_corrupt(_t)
    print("logit_diff (corrupt run, patched, last pos):", _ld_patched)
    print("  shape:", tuple(_ld_patched.shape))

    # Patch should change the readout for at least the first two rows.
    for _i in (0, 1):
        assert not _torch.allclose(_ld_corrupt[_i], _ld_patched[_i], rtol=0, atol=1e-5), _i
