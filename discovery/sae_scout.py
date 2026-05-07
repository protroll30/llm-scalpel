"""SAE discovery utilities.

This module is intentionally lightweight: it provides a single entry point,
`sae_loader`, that can fetch SAE weights (typically from a URL) and cache them.

Design goals:
- **Pure utility**: no global project state beyond a small in-process cache.
- **Safe caching**: cache key is derived from the input spec.
- **Format-flexible**: supports common weight formats (`.pt`/`.pth` via torch,
  `.npz` via numpy).

Neuronpedia note:
- Neuronpedia provides an API + public exports, but the exact weight-file URLs
  depend on the dataset/source you’re using. This loader therefore accepts a
  concrete URL/path and focuses on caching + loading.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Union, cast

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SAEWeightBundle:
    """Loaded SAE weights plus minimal metadata.

    `weights` is whatever the underlying file contains:
    - For `.pt`/`.pth`: typically a dict[str, Tensor] or a module state dict.
    - For `.npz`: a dict[str, ndarray] (unless you convert them yourself).
    """

    spec: Mapping[str, Any]
    weights: Any
    local_path: Path


_IN_PROCESS_CACHE: MutableMapping[str, SAEWeightBundle] = {}


def _default_cache_dir() -> Path:
    # `.cache` is already gitignored in this repo.
    return Path(".cache") / "sae"


def _stable_cache_key(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _download_to(url: str, dest: Path, *, timeout_s: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    # We prefer requests (comes transitively via neuronpedia) but keep a stdlib fallback.
    try:
        import requests  # type: ignore

        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except ModuleNotFoundError:  # pragma: no cover
        from urllib.request import urlopen

        with urlopen(url, timeout=timeout_s) as resp:  # nosec - URL is user-provided
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

    tmp.replace(dest)


def _load_weights_file(path: Path, *, map_location: str = "cpu") -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin"}:
        return torch.load(path, map_location=map_location)
    if suffix == ".npz":
        if np is None:
            raise RuntimeError("numpy is required to load .npz SAE weights.")
        data = cast(Any, np.load(path, allow_pickle=False))
        # Convert to a plain dict to avoid keeping the file handle open.
        return {k: data[k] for k in data.files}
    raise ValueError(f"Unsupported SAE weights format: {path.name} (suffix={suffix!r})")


def sae_loader(
    *,
    weights_url: Optional[str] = None,
    weights_path: Optional[PathLike] = None,
    cache_dir: Optional[PathLike] = None,
    cache: bool = True,
    map_location: str = "cpu",
    extra_spec: Optional[Mapping[str, Any]] = None,
) -> SAEWeightBundle:
    """Fetch and cache SAE weights, returning loaded weights.

    Provide either:
    - `weights_url`: HTTP(S) URL to a weight file
    - `weights_path`: local path to a weight file

    Caching:
    - If `cache=True` and `weights_url` is provided, we download into
      `<cache_dir>/<cache_key>/<filename>` and reuse it on subsequent calls.
    - We also keep a small in-process cache keyed by the same `cache_key`.
    """

    if (weights_url is None) == (weights_path is None):
        raise ValueError("Provide exactly one of weights_url or weights_path.")

    spec: Dict[str, Any] = {}
    if weights_url is not None:
        spec["weights_url"] = weights_url
    if weights_path is not None:
        spec["weights_path"] = str(Path(weights_path))
    if extra_spec:
        spec["extra_spec"] = dict(extra_spec)

    cache_key = _stable_cache_key(spec)
    if cache and cache_key in _IN_PROCESS_CACHE:
        return _IN_PROCESS_CACHE[cache_key]

    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    bundle_dir = cache_root / cache_key
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if weights_path is not None:
        local_path = Path(weights_path)
        if not local_path.exists():
            raise FileNotFoundError(str(local_path))
        weights = _load_weights_file(local_path, map_location=map_location)
        bundle = SAEWeightBundle(spec=spec, weights=weights, local_path=local_path)
        if cache:
            _IN_PROCESS_CACHE[cache_key] = bundle
        return bundle

    # URL case
    assert weights_url is not None
    filename = weights_url.split("?")[0].rstrip("/").split("/")[-1] or "weights"
    local_path = bundle_dir / filename

    if not local_path.exists() or not cache:
        _download_to(weights_url, local_path)

    weights = _load_weights_file(local_path, map_location=map_location)

    # Minimal metadata for provenance/debugging.
    meta_path = bundle_dir / "meta.json"
    try:
        meta_path.write_text(json.dumps({"spec": spec, "local_path": str(local_path)}, indent=2))
    except Exception:
        # Best-effort only; cache still works without this.
        pass

    bundle = SAEWeightBundle(spec=spec, weights=weights, local_path=local_path)
    if cache:
        _IN_PROCESS_CACHE[cache_key] = bundle
    return bundle


def patch_features(
    f_corrupt: torch.Tensor,
    f_clean: torch.Tensor,
    feature_indices: Union[torch.Tensor, Sequence[int]],
) -> torch.Tensor:
    """Create f_patched by overwriting selected feature dims from clean into corrupt.

    This implements:
        f_patched = f_corrupt;  f_patched[..., idx] = f_clean[..., idx]

    Works for common shapes like:
    - `[n_features]`
    - `[pos, n_features]`
    - `[batch, pos, n_features]`
    - any tensor with the **last dimension** = `n_features`
    """

    if f_corrupt.shape != f_clean.shape:
        raise ValueError(
            "f_corrupt and f_clean must have the same shape; "
            f"got {tuple(f_corrupt.shape)} vs {tuple(f_clean.shape)}"
        )
    if f_corrupt.dim() < 1:
        raise ValueError(f"Expected at least 1D feature tensor, got shape {tuple(f_corrupt.shape)}")

    if isinstance(feature_indices, torch.Tensor):
        idx = feature_indices.to(device=f_corrupt.device)
        if idx.dtype != torch.long:
            idx = idx.to(dtype=torch.long)
    else:
        idx = torch.tensor(list(feature_indices), device=f_corrupt.device, dtype=torch.long)

    if idx.numel() == 0:
        return f_corrupt.clone()

    n_features = f_corrupt.shape[-1]
    if (idx < 0).any() or (idx >= n_features).any():
        raise IndexError(f"feature_indices out of range for n_features={n_features}")

    f_patched = f_corrupt.clone()
    f_patched.index_copy_(-1, idx, f_clean.index_select(-1, idx))
    return f_patched


def reconstruct_activation(
    *,
    f_patched: torch.Tensor,
    x_corrupt: torch.Tensor,
    f_corrupt: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Reconstruct a patched model activation using the residual from the corrupt example.

    Implements:
        Decode(f_patched) + (x_corrupt - Decode(f_corrupt))

    Shapes:
    - `f_*` can be any shape as long as `decode_fn(f_*)` matches `x_corrupt`.
    - Typically: `f_*` is `[batch, pos, n_features]` and `x_corrupt` is `[batch, pos, d_model]`.
    """

    xhat_patched = decode_fn(f_patched)
    xhat_corrupt = decode_fn(f_corrupt)

    if xhat_patched.shape != x_corrupt.shape:
        raise ValueError(
            "decode_fn(f_patched) must match x_corrupt shape; "
            f"got {tuple(xhat_patched.shape)} vs {tuple(x_corrupt.shape)}"
        )
    if xhat_corrupt.shape != x_corrupt.shape:
        raise ValueError(
            "decode_fn(f_corrupt) must match x_corrupt shape; "
            f"got {tuple(xhat_corrupt.shape)} vs {tuple(x_corrupt.shape)}"
        )

    return xhat_patched + (x_corrupt - xhat_corrupt)


def feature_capture(
    *,
    model: Any,
    prompt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seq_pos: Optional[int] = -1,
    threshold: float = 0.0,
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_residual: bool = False,
) -> torch.Tensor:
    """Run a clean forward pass, encode activations, and return nonzero feature indices.

    Args:
        model: A TransformerLens-style model with `.to_tokens(...)` and `.run_with_cache(...)`.
        prompt: Clean prompt string.
        hook_name: Activation name to read from cache (e.g. `blocks.3.hook_resid_pre`).
        encode_fn: Function mapping activations -> feature activations.
            Expected to return a tensor with last dim = n_features.
        decode_fn: Optional function mapping feature activations -> reconstructed activations.
            If provided and `return_residual=True`, we compute residuals `x - xhat`.
        seq_pos: Token position to capture.
            - `None`: consider all sequence positions and return the union of nonzero feature ids.
            - `int` (default `-1`): capture one position (supports negative indexing).
        threshold: Treat a feature as “present” if `abs(act) > threshold`.
        prepend_bos: Passed through to `model.to_tokens` when supported (TransformerLens).
        device: If set, move tokens to this device before the forward pass.
        return_residual: If True, return a tuple `(indices, residual)` where residual is
            `x_corrupt - xhat_corrupt` at `seq_pos` (or full sequence if `seq_pos=None`).

    Returns:
        If `return_residual=False` (default): 1D `LongTensor` of unique feature indices.
        If `return_residual=True`: `(indices, residual)` where `residual` is a `Tensor`.
    """

    # Tokenize
    to_tokens_kwargs: Dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kwargs["prepend_bos"] = prepend_bos
    tokens = model.to_tokens(prompt, **to_tokens_kwargs)
    if device is not None:
        tokens = tokens.to(device)

    present_mask: torch.Tensor | None = None  # shape: [n_features] boolean
    residual_out: torch.Tensor | None = None

    def _capture_hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        nonlocal present_mask
        nonlocal residual_out

        feats = encode_fn(act)
        if feats.dim() < 1:
            raise ValueError(f"encode_fn returned rank-{feats.dim()} tensor; expected at least 1D.")

        # Normalize to [batch, pos, n_features] if possible.
        if feats.dim() == 2:
            feats_ = feats.unsqueeze(0)
        else:
            feats_ = feats

        if feats_.dim() < 3:
            raise ValueError(
                f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(feats.shape)}"
            )

        if seq_pos is None:
            present = feats_.abs() > threshold  # [batch, pos, n_features]
            cur = present.any(dim=0).any(dim=0)  # [n_features]
        else:
            pos = int(seq_pos)
            present = feats_[:, pos, :].abs() > threshold  # [batch, n_features]
            cur = present.any(dim=0)  # [n_features]

        if present_mask is None:
            present_mask = cur.detach().to(dtype=torch.bool)
        else:
            present_mask |= cur.detach().to(dtype=torch.bool)

        if return_residual:
            if decode_fn is None:
                raise ValueError("return_residual=True requires decode_fn to be provided.")
            xhat = decode_fn(feats_)
            if xhat.shape != act.shape:
                raise ValueError(
                    "decode_fn output must match activation shape. "
                    f"Got xhat={tuple(xhat.shape)} vs act={tuple(act.shape)}"
                )
            resid = act - xhat  # x_corrupt - xhat_corrupt
            if seq_pos is None:
                residual_out = resid.detach()
            else:
                residual_out = resid[:, int(seq_pos), ...].detach()

        # Return the original activation unchanged; we only “peek”.
        return act

    # Run a clean pass with a temporary hook: encode on-the-fly, do not keep raw activations.
    _ = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, _capture_hook)],
        return_type="logits",
    )

    if present_mask is None:
        raise RuntimeError(f"Hook {hook_name!r} never fired (is the name correct for this model?).")

    indices = present_mask.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    if not return_residual:
        return indices
    if residual_out is None:
        raise RuntimeError("return_residual=True but residual was not captured (unexpected).")
    return indices, residual_out


# Back-compat: attribution lives in ``discovery.attribution``.
from discovery.attribution import feature_act_grad_scores  # noqa: E402

