"""Baseline runs and clean-into-corrupt activation patching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Union

import torch
import transformer_lens.utilities as utils

from causal_patcher.targets import PatchKind, PatchTarget

if TYPE_CHECKING:
    from transformer_lens.hook_points import HookedRootModule

NamesFilter = Optional[Union[str, List[str], Callable[[str], bool]]]


def _resolve_positions(
    positions: Union[int, slice, None], seq_len: int
) -> Union[slice, int]:
    if positions is None:
        return slice(None)
    if isinstance(positions, int) and positions < 0:
        return positions + seq_len
    return positions


def _patch_fn(
    clean_activation: torch.Tensor,
    target: PatchTarget,
    positions: Union[int, slice, None],
    seq_len: int,
):
    pos = _resolve_positions(positions, seq_len)

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        src = clean_activation.to(device=activation.device, dtype=activation.dtype)
        if target.kind == "attn_head_z":
            h = target.head
            assert h is not None
            if isinstance(pos, slice):
                activation[:, pos, h, :] = src[:, pos, h, :]
            else:
                activation[:, pos, h, :] = src[:, pos, h, :]
        else:
            if isinstance(pos, slice):
                activation[:, pos, ...] = src[:, pos, ...]
            else:
                activation[:, pos, ...] = src[:, pos, ...]
        return activation

    return hook_fn


class ExperimentRunner:
    """Tokenize clean/corrupt strings, cache activations, and patch clean into corrupt runs."""

    def __init__(
        self,
        model: "HookedRootModule",
        clean_prompt: str,
        corrupt_prompt: str,
        clean_answer_token: int,
        corrupt_answer_token: int,
        *,
        run_baselines: bool = True,
        names_filter: NamesFilter = None,
    ) -> None:
        self.model = model
        self.clean_prompt = clean_prompt
        self.corrupt_prompt = corrupt_prompt
        self.clean_answer_token = int(clean_answer_token)
        self.corrupt_answer_token = int(corrupt_answer_token)

        self.clean_tokens: torch.Tensor | None = None
        self.corrupt_tokens: torch.Tensor | None = None
        self.clean_logits: torch.Tensor | None = None
        self.corrupt_logits: torch.Tensor | None = None
        self.clean_cache = None
        self.corrupt_cache = None

        if run_baselines:
            self.run_baselines(names_filter=names_filter)

    def run_baselines(self, names_filter: NamesFilter = None) -> None:
        """Forward clean and corrupt prompts with ``run_with_cache`` and store logits and caches."""
        self.clean_tokens = self.model.to_tokens(self.clean_prompt)
        self.corrupt_tokens = self.model.to_tokens(self.corrupt_prompt)

        if self.clean_tokens.shape != self.corrupt_tokens.shape:
            raise ValueError(
                "clean and corrupt token tensors must have the same shape for patching; "
                f"got {tuple(self.clean_tokens.shape)} vs {tuple(self.corrupt_tokens.shape)}."
            )

        self.clean_logits, self.clean_cache = self.model.run_with_cache(
            self.clean_tokens, names_filter=names_filter, return_type="logits"
        )
        self.corrupt_logits, self.corrupt_cache = self.model.run_with_cache(
            self.corrupt_tokens, names_filter=names_filter, return_type="logits"
        )

    def _require_baselines(self) -> None:
        if self.clean_cache is None or self.corrupt_tokens is None:
            raise RuntimeError("Call run_baselines() first (or pass run_baselines=True).")

    def logit_diff(
        self,
        logits: torch.Tensor,
        *,
        seq_pos: int = -1,
        clean_token: int | None = None,
        corrupt_token: int | None = None,
    ) -> torch.Tensor:
        """``logit(clean) - logit(corrupt)`` at ``seq_pos`` (default: last token)."""
        ct = self.clean_answer_token if clean_token is None else clean_token
        ut = self.corrupt_answer_token if corrupt_token is None else corrupt_token
        if logits.dim() == 3:
            return logits[0, seq_pos, ct] - logits[0, seq_pos, ut]
        if logits.dim() == 2:
            return logits[seq_pos, ct] - logits[seq_pos, ut]
        raise ValueError(f"Expected logits rank 2 or 3, got shape {tuple(logits.shape)}")

    def patch_clean_into_corrupt(
        self,
        target: PatchTarget,
        *,
        positions: Union[int, slice, None] = None,
    ) -> torch.Tensor:
        """Run the corrupt prompt with a forward hook that overwrites the target activation from the clean cache.

        Args:
            target: Patch site (layer, kind, and optional head for ``attn_head_z``).
            positions: Token index (``int``), ``slice``, or ``None`` for all positions.
                Negative ints index from the end of the sequence.

        Returns:
            Logits from the patched corrupt forward pass.
        """
        self._require_baselines()
        assert self.clean_cache is not None
        assert self.corrupt_tokens is not None

        hook_name = target.hook_name()
        if hook_name not in self.clean_cache:
            raise KeyError(
                f"Clean cache has no entry for {hook_name!r}. "
                "Re-run with a names_filter that includes this hook (or use default None to cache all)."
            )

        clean_activation = self.clean_cache[hook_name]
        seq_len = self.corrupt_tokens.shape[-1]
        hook_fn = _patch_fn(clean_activation, target, positions, seq_len)

        logits = self.model.run_with_hooks(
            self.corrupt_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )
        return logits

    @staticmethod
    def all_patch_hook_names(n_layers: int, kinds: Iterable[PatchKind] | None = None) -> list[str]:
        """Convenience list of hook names for ``names_filter`` covering common patch sites."""
        kinds = kinds or (
            "resid_pre",
            "resid_mid",
            "resid_post",
            "mlp_out",
            "attn_head_z",
        )
        names: list[str] = []
        for layer in range(n_layers):
            for k in kinds:
                if k == "attn_head_z":
                    names.append(utils.get_act_name("z", layer))
                else:
                    names.append(utils.get_act_name(k, layer))
        return names
