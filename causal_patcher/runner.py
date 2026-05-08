"""Baseline runs and clean-into-corrupt activation patching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Union

import torch
from transformer_lens import utils as tl_utils

from causal_patcher.targets import PatchKind, PatchPos, PatchTarget

if TYPE_CHECKING:
    from transformer_lens.hook_points import HookedRootModule

NamesFilter = Optional[Union[str, List[str], Callable[[str], bool]]]


def _resolve_index(i: int, seq_len: int) -> int:
    if i < 0:
        return i + seq_len
    return i


def _resolve_patch_pos(
    pos: Optional[PatchPos],
    seq_len: int,
) -> Union[slice, int, tuple[int, int]]:
    """Normalize position spec: aligned ``slice``/``int``, or ``(clean_i, corrupt_i)`` with negative indices fixed."""
    if pos is None:
        return slice(None)
    if isinstance(pos, slice):
        return pos
    if isinstance(pos, tuple):
        if len(pos) != 2:
            raise ValueError("pos tuple must be (clean_index, corrupt_index)")
        a, b = int(pos[0]), int(pos[1])
        return (_resolve_index(a, seq_len), _resolve_index(b, seq_len))
    if isinstance(pos, int):
        return _resolve_index(pos, seq_len)
    raise TypeError(f"Invalid pos spec: {pos!r}")


def _patch_fn(
    clean_activation: torch.Tensor,
    target: PatchTarget,
    pos_spec: Union[slice, int, tuple[int, int]],
):
    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        src = clean_activation.to(device=activation.device, dtype=activation.dtype)
        if target.kind == "attn_head_z":
            h = target.head
            assert h is not None
            if isinstance(pos_spec, tuple) and len(pos_spec) == 2:
                ci, ri = pos_spec
                activation[:, ri, h, :] = src[:, ci, h, :]
            elif isinstance(pos_spec, slice):
                activation[:, pos_spec, h, :] = src[:, pos_spec, h, :]
            else:
                activation[:, pos_spec, h, :] = src[:, pos_spec, h, :]
        else:
            if isinstance(pos_spec, tuple) and len(pos_spec) == 2:
                ci, ri = pos_spec
                activation[:, ri, ...] = src[:, ci, ...]
            elif isinstance(pos_spec, slice):
                activation[:, pos_spec, ...] = src[:, pos_spec, ...]
            else:
                activation[:, pos_spec, ...] = src[:, pos_spec, ...]
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
        prepend_bos: bool | None = None,
    ) -> None:
        self.model = model
        self.clean_prompt = clean_prompt
        self.corrupt_prompt = corrupt_prompt
        self.clean_answer_token = int(clean_answer_token)
        self.corrupt_answer_token = int(corrupt_answer_token)
        self.prepend_bos = prepend_bos

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
        tk_kw = {}
        if self.prepend_bos is not None:
            tk_kw["prepend_bos"] = bool(self.prepend_bos)
        self.clean_tokens = self.model.to_tokens(self.clean_prompt, **tk_kw)
        self.corrupt_tokens = self.model.to_tokens(self.corrupt_prompt, **tk_kw)

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
        positions: Optional[PatchPos] = None,
    ) -> torch.Tensor:
        """Run the corrupt prompt with a forward hook that overwrites the target activation from the clean cache.

        Args:
            target: Patch site (``kind``, ``layer``, ``head`` for z, and optional ``pos``). If
                ``target.pos`` is set, it chooses which token positions to patch. A tuple
                ``(clean_index, corrupt_index)`` reads the clean run at the first index and
                overwrites the corrupt run at the second.
            positions: If not ``None``, overrides ``target.pos`` (same types as :attr:`PatchTarget.pos`).

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
        effective_pos = target.pos if positions is None else positions
        pos_spec = _resolve_patch_pos(effective_pos, seq_len)
        hook_fn = _patch_fn(clean_activation, target, pos_spec)

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
                    names.append(tl_utils.get_act_name("z", layer))
                else:
                    names.append(tl_utils.get_act_name(k, layer))
        return names
