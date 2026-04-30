"""Patch targets map to TransformerLens hook names via :func:`transformer_lens.utils.get_act_name`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformer_lens import utils as tl_utils

PatchKind = Literal["resid_pre", "resid_mid", "resid_post", "mlp_out", "attn_head_z"]

# ``None``: all positions (aligned copy). ``int`` / ``slice``: same indices in clean and corrupt.
# ``tuple[int, int]``: ``(clean_index, corrupt_index)`` — read clean activations at the first index
# and write into the corrupt forward pass at the second.
PatchPos = int | slice | tuple[int, int] | None


def patch_hook_name(kind: PatchKind, layer: int) -> str:
    """Resolve the hook name for a patch site (no head index; use for all non-head-z kinds)."""
    if kind == "attn_head_z":
        return tl_utils.get_act_name("z", layer)
    return tl_utils.get_act_name(kind, layer)


@dataclass(frozen=True)
class PatchTarget:
    """Single activation patch site.

    For ``attn_head_z``, set ``head`` to the head index; only that head is overwritten
    when patching clean activations into the corrupt forward pass.

    ``pos`` selects which token positions participate. A pair ``(clean_index, corrupt_index)``
    copies from the clean run at ``clean_index`` into the corrupt run at ``corrupt_index``.
    """

    kind: PatchKind
    layer: int
    head: int | None = None
    pos: PatchPos = None

    def __post_init__(self) -> None:
        if self.kind == "attn_head_z":
            if self.head is None:
                raise ValueError("attn_head_z requires head index")
        elif self.head is not None:
            raise ValueError("head is only valid for attn_head_z")
        if isinstance(self.pos, tuple):
            if len(self.pos) != 2:
                raise ValueError("pos tuple must be (clean_index, corrupt_index)")
            a, b = self.pos
            if not isinstance(a, int) or not isinstance(b, int):
                raise TypeError("pos tuple must be two integers")

    def hook_name(self) -> str:
        """Full TransformerLens hook name for this target."""
        return patch_hook_name(self.kind, self.layer)
