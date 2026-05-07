"""Glue between `sae_lens.SAE` and `discovery` hook encoders.

`discovery` expects:

- ``encode_fn(act) -> Tensor`` with shape ``[pos, n_features]`` or ``[batch, pos, n_features]``.
- ``decode_fn(f) -> Tensor`` with the **same** shape as ``act`` (typically ``[batch, pos, d_model]``).

SAELens :meth:`sae_lens.SAE.encode` and :meth:`sae_lens.SAE.decode` are applied elementwise over all
but the last dimension, so hook tensors from TransformerLens match without extra reshaping for
common residual hooks. For ``hook_z`` SAEs, use SAELens docs / ``turn_on_forward_pass_hook_z_reshaping``
if your hook activations are head-structured.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, cast

import torch

try:  # pragma: no cover - import guarded for environments without sae_lens
    from sae_lens import SAE as SaeLensSAE
except ImportError:  # pragma: no cover
    SaeLensSAE = None  # type: ignore[misc, assignment]


def load_pretrained_sae(
    *,
    release: str,
    sae_id: str,
    device: Union[str, torch.device] = "cpu",
    dtype: str = "float32",
    force_download: bool = False,
) -> "SaeLensSAE":
    """Load weights via :meth:`sae_lens.SAE.from_pretrained`.

    Args:
        release: SAELens release name (see SAELens pretrained SAE tables), or a Hugging Face repo id.
        sae_id: SAE identifier within that release (often a TransformerLens hook path).
        device: Device string, e.g. ``\"cpu\"`` or ``\"cuda\"``.
        dtype: SAELens dtype string, e.g. ``\"float32\"`` or ``\"bfloat16\"``.
        force_download: Forwarded to SAELens (re-fetch from the hub).
    """

    if SaeLensSAE is None:  # pragma: no cover
        raise ImportError("sae_lens is not installed; add it to project dependencies and pip install.")

    dev = device if isinstance(device, str) else str(device)
    return cast(
        "SaeLensSAE",
        SaeLensSAE.from_pretrained(
            release,
            sae_id,
            device=dev,
            dtype=dtype,
            force_download=force_download,
        ),
    )


def discovery_encode_decode(
    sae: "SaeLensSAE",
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Return ``(encode_fn, decode_fn)`` closures wrapping a loaded SAELens SAE."""

    sae.train(False)

    def encode_fn(x: torch.Tensor, /) -> torch.Tensor:
        return sae.encode(x)

    def decode_fn(f: torch.Tensor, /) -> torch.Tensor:
        return sae.decode(f)

    return encode_fn, decode_fn


def metadata_hook_name(sae: "SaeLensSAE") -> Optional[str]:
    """Hook path stored in SAELens metadata, if any (e.g. ``blocks.8.hook_resid_pre``)."""

    meta = getattr(sae.cfg, "metadata", None)
    if meta is None:
        return None
    name = getattr(meta, "hook_name", None)
    return str(name) if name else None


def assert_d_in_matches_model(
    sae: "SaeLensSAE",
    *,
    d_model: int,
) -> None:
    """Raise if the SAE input width does not match the model residual width at the hook."""

    d_in = int(getattr(sae.cfg, "d_in", -1))
    if d_in != int(d_model):
        raise ValueError(
            f"SAE cfg.d_in={d_in} does not match model d_model={d_model}. "
            "Pick an SAE trained on this model width and hook site."
        )
