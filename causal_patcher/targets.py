"""Patch targets map to TransformerLens hook names via ``utils.get_act_name``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import transformer_lens.utilities as utils

PatchKind = Literal["resid_pre", "resid_mid", "resid_post", "mlp_out", "attn_head_z"]


def patch_hook_name(kind: PatchKind, layer: int) -> str:
    """Resolve the hook name for a patch site (no head index; use for all non-head-z kinds)."""
    if kind == "attn_head_z":
        return utils.get_act_name("z", layer)
    return utils.get_act_name(kind, layer)


@dataclass(frozen=True)
class PatchTarget:
    """Single activation patch site.

    For ``attn_head_z``, set ``head`` to the head index; only that head is overwritten
    when patching clean activations into the corrupt forward pass.
    """

    kind: PatchKind
    layer: int
    head: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "attn_head_z":
            if self.head is None:
                raise ValueError("attn_head_z requires head index")
        elif self.head is not None:
            raise ValueError("head is only valid for attn_head_z")

    def hook_name(self) -> str:
        """Full TransformerLens hook name for this target."""
        return patch_hook_name(self.kind, self.layer)
