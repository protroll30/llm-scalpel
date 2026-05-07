"""Activation × gradient attribution on SAE latents at a TransformerLens hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class AttributionMetadata:
    """What scalar drove ∂L/∂f and where latents were read."""

    metric: str
    """Human-readable name, e.g. ``logit_diff``, ``kl_divergence``, ``negative_log_prob``."""

    seq_pos: int
    """Token index used for ``f`` and ``∂L/∂f`` (resolved non-negative index into the sequence)."""

    hook_name: str = ""
    """Hook where encoder ran (informational)."""


@dataclass(frozen=True)
class SAEAttributionComponents:
    """Debuggable pieces of a clean-vs-corrupt latent attribution."""

    delta_f: torch.Tensor
    """SAE latent difference ``Δf = f_clean - f_corrupt`` at ``metadata.seq_pos``, shape ``[n_features]``."""

    gradient_g: torch.Tensor
    """``∂L/∂f`` at the corrupt forward’s latents, same shape as ``delta_f``."""


@dataclass(frozen=True)
class SAEAttributionPass:
    """Full output of one attribution pass over SAE latents."""

    indices: torch.Tensor
    """Latent indices ``0 … n_features-1`` (dtype ``long``, aligned with ``scores``)."""

    scores: torch.Tensor
    """Per-latent ``Δf ⊙ g`` (element-wise product); aligns with ``(Δx)·∇`` in feature coordinates."""

    components: SAEAttributionComponents
    metadata: AttributionMetadata


@dataclass(frozen=True)
class LatentGradientSnapshot:
    """Checkpoint of ``∂L/∂f`` before a pruning wave (for cosine drift diagnostics)."""

    gradient_g: torch.Tensor
    """Full latent gradient ``g``, shape ``[n_features]``, detached."""

    forced_zero_indices: frozenset[int]
    """Mask state used for this backward (pruned / ablated latents)."""

    seq_pos_effective: int
    """Resolved token index used when slicing ``f`` and ``∂L/∂f``."""

    hook_name: str = ""
    metric: str = ""
    """Optional label matching the scalar loss used for backward."""


@dataclass(frozen=True)
class LatentGradientCosineDiagnostics:
    """Cosine similarity between two latent-gradient snapshots or vectors."""

    cosine_similarity: torch.Tensor
    """Scalar in ``[-1, 1]`` (shape ``[]``). NaN if either vector has zero norm."""

    norm_original: torch.Tensor
    norm_new: torch.Tensor


def _feat_grad_corrupt_forward(
    *,
    model: Any,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    seq_pos: int,
    prepend_bos: Optional[bool],
    device: Optional[Union[str, torch.device]],
    forced_zero_indices: Optional[AbstractSet[int]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """One corrupt forward + backward.

    Returns ``(f_corrupt, ∂L/∂f, seq_pos_effective)`` at ``seq_pos``; tensors shaped ``[n_features]``.
    """

    to_tokens_kwargs: Dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kwargs["prepend_bos"] = prepend_bos
    tokens = model.to_tokens(prompt_corrupt, **to_tokens_kwargs)
    if device is not None:
        tokens = tokens.to(device)

    feats_for_grad: torch.Tensor | None = None
    forced = frozenset(forced_zero_indices) if forced_zero_indices else frozenset()

    def _grad_hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        nonlocal feats_for_grad
        g = encode_fn(act)
        if g.dim() == 2:
            g = g.unsqueeze(0)
        if g.dim() < 3:
            raise ValueError(
                f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(g.shape)}"
            )
        g = g.clone()
        g.retain_grad()
        feats_for_grad = g

        f_masked = g.clone()
        if forced:
            idx = torch.tensor(sorted(forced), device=g.device, dtype=torch.long)
            f_masked.index_fill_(-1, idx, 0.0)

        xhat_m = decode_fn(f_masked)
        xhat_full = decode_fn(g)
        if xhat_m.shape != act.shape or xhat_full.shape != act.shape:
            raise ValueError(
                "decode_fn output must match activation shape. "
                f"Got masked={tuple(xhat_m.shape)} full={tuple(xhat_full.shape)} vs act={tuple(act.shape)}"
            )
        return xhat_m + (act - xhat_full.detach())

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, _grad_hook)],
            return_type="logits",
        )
        loss = logits_to_scalar_loss(logits)
        loss.backward()

    if feats_for_grad is None or feats_for_grad.grad is None:
        raise RuntimeError(
            f"Hook {hook_name!r} did not produce feature gradients (hook name or loss graph?)."
        )

    pos_effective = int(seq_pos)
    seq_dim = int(feats_for_grad.shape[1])
    if pos_effective < 0:
        pos_effective += seq_dim
    f_co = feats_for_grad[0, pos_effective, :]
    ggrad = feats_for_grad.grad[0, pos_effective, :]
    return f_co.detach(), ggrad.detach(), pos_effective


def feature_act_grad_scores(
    *,
    model: Any,
    prompt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    seq_pos: int = -1,
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    forced_zero_indices: Optional[AbstractSet[int]] = None,
) -> torch.Tensor:
    """Per-feature ``|f| · |∂L/∂f|`` at ``seq_pos`` on the corrupt run (ranking signal for pruning).

    Same graph as :func:`feature_attribution_pass` but **no** clean prompt — magnitude-only salience on corrupt latents.
    """

    f_co, g, _pos_eff = _feat_grad_corrupt_forward(
        model=model,
        prompt_corrupt=prompt,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        seq_pos=seq_pos,
        prepend_bos=prepend_bos,
        device=device,
        forced_zero_indices=forced_zero_indices,
    )
    return (f_co.abs() * g.abs()).detach()


def feature_attribution_pass(
    *,
    model: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    metric: str,
    seq_pos: int = -1,
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    forced_zero_indices: Optional[AbstractSet[int]] = None,
) -> SAEAttributionPass:
    """Full clean-vs-corrupt attribution: ``scores_i = (Δf)_i · (∂L/∂f)_i`` with ``Δf = f_cl - f_co``.

    Runs corrupt forward + backward (same masked substitution as :func:`feature_act_grad_scores`),
    then a **no-grad** clean forward with ``run_with_cache`` to obtain ``f_clean`` at the hook.

    Args:
        metric: Stored in :class:`AttributionMetadata` (e.g. ``\"logit_diff\"``, ``\"kl_div_last_token\"``).
            Must match what ``logits_to_scalar_loss`` implements.
    """

    f_co, g, pos_effective = _feat_grad_corrupt_forward(
        model=model,
        prompt_corrupt=prompt_corrupt,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        seq_pos=seq_pos,
        prepend_bos=prepend_bos,
        device=device,
        forced_zero_indices=forced_zero_indices,
    )

    to_tokens_kwargs: Dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kwargs["prepend_bos"] = prepend_bos
    clean_tokens = model.to_tokens(prompt_clean, **to_tokens_kwargs)
    if device is not None:
        clean_tokens = clean_tokens.to(device)

    seq_len = int(clean_tokens.shape[-1])
    if pos_effective >= seq_len:
        raise IndexError(f"seq_pos_effective={pos_effective} out of range for clean seq_len={seq_len}")

    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(
            clean_tokens,
            names_filter=[hook_name],
            return_type="logits",
        )
        if hook_name not in cache:
            raise KeyError(f"Clean cache missing {hook_name!r}.")
        act_cl = cache[hook_name]
        f_raw = encode_fn(act_cl)
        if f_raw.dim() == 2:
            f_raw = f_raw.unsqueeze(0)
        if f_raw.dim() < 3:
            raise ValueError(
                f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f_raw.shape)}"
            )
        f_cl = f_raw[0, pos_effective, :].to(dtype=f_co.dtype, device=f_co.device)

    f_co_slice = f_co.to(dtype=f_cl.dtype, device=f_cl.device)
    delta_f = f_cl - f_co_slice
    scores = delta_f * g.to(dtype=delta_f.dtype, device=delta_f.device)

    n_features = int(scores.numel())
    indices = torch.arange(n_features, device=scores.device, dtype=torch.long)

    meta = AttributionMetadata(metric=metric, seq_pos=pos_effective, hook_name=hook_name)
    components = SAEAttributionComponents(delta_f=delta_f.detach(), gradient_g=g.detach())

    return SAEAttributionPass(
        indices=indices,
        scores=scores.detach(),
        components=components,
        metadata=meta,
    )


def capture_latent_gradient_snapshot(
    *,
    model: Any,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    seq_pos: int = -1,
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    forced_zero_indices: Optional[AbstractSet[int]] = None,
    metric: str = "",
) -> LatentGradientSnapshot:
    """Micro backward pass: save ``g = ∂L/∂f`` at ``seq_pos`` for the current mask (before / after a prune)."""

    _f_co, g, pos_eff = _feat_grad_corrupt_forward(
        model=model,
        prompt_corrupt=prompt_corrupt,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        seq_pos=seq_pos,
        prepend_bos=prepend_bos,
        device=device,
        forced_zero_indices=forced_zero_indices,
    )
    frozen_mask = frozenset(forced_zero_indices) if forced_zero_indices else frozenset()
    return LatentGradientSnapshot(
        gradient_g=g.clone(),
        forced_zero_indices=frozen_mask,
        seq_pos_effective=pos_eff,
        hook_name=hook_name,
        metric=metric,
    )


def latent_gradient_cosine_similarity(
    g_original: torch.Tensor,
    g_new: torch.Tensor,
    *,
    subset_indices: Optional[Iterable[int]] = None,
    eps: float = 1e-12,
) -> LatentGradientCosineDiagnostics:
    """Cosine similarity ``⟨g_orig, g_new⟩ / (‖g_orig‖ ‖g_new‖)`` over matching latent dimensions.

    If ``subset_indices`` is set (e.g. latents still ``alive`` after a wave), cosine is computed only
    on those columns (useful when comparing gradients restricted to survivors).
    """

    go = g_original.flatten()
    gn = g_new.flatten()
    if go.numel() != gn.numel():
        raise ValueError(
            f"g_original and g_new must have the same length; got {go.numel()} vs {gn.numel()}."
        )
    if subset_indices is not None:
        idx = torch.tensor(list(subset_indices), device=go.device, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("subset_indices must be non-empty when provided.")
        go = go.index_select(0, idx)
        gn = gn.index_select(0, idx)
    n_orig = go.norm()
    n_new = gn.norm()
    out_dtype = torch.result_type(go, gn)
    if float(n_orig.item()) < eps or float(n_new.item()) < eps:
        cos = torch.full((), float("nan"), device=g_original.device, dtype=out_dtype)
    else:
        cos = ((go * gn).sum() / (n_orig * n_new)).to(dtype=out_dtype)
    return LatentGradientCosineDiagnostics(
        cosine_similarity=cos,
        norm_original=n_orig.detach(),
        norm_new=n_new.detach(),
    )


def calculate_gradient_drift(
    pass_orig: SAEAttributionPass,
    pass_new: SAEAttributionPass,
    *,
    eps: float = 1e-12,
) -> float:
    """Cosine similarity between corrupt-run gradients ``∂L/∂f`` from two :class:`SAEAttributionPass` runs.

    Uses ``pass_*.components.gradient_g`` (same shape). Typical use: ``pass_orig`` at mask ``R`` before a
    wave, ``pass_new`` at ``R ∪ chunk`` after a proposed prune — high drift similarity ⇒ gradient field
    barely rotated ⇒ safer to commit the wave (cf. budget pruner: chunk-size-aware cosine cutoff).

    Returns:
        Scalar cosine in ``[-1, 1]``, or ``nan`` if either gradient has negligible norm.

    Raises:
        ValueError: Mismatched tensor shapes or incompatible metadata (hook / seq pos).
    """

    if pass_orig.components.gradient_g.shape != pass_new.components.gradient_g.shape:
        raise ValueError(
            "Gradient shapes differ between passes: "
            f"{tuple(pass_orig.components.gradient_g.shape)} vs "
            f"{tuple(pass_new.components.gradient_g.shape)}."
        )
    if pass_orig.metadata.seq_pos != pass_new.metadata.seq_pos:
        raise ValueError(
            f"seq_pos mismatch: {pass_orig.metadata.seq_pos} vs {pass_new.metadata.seq_pos}."
        )
    ho, hn = pass_orig.metadata.hook_name, pass_new.metadata.hook_name
    if ho and hn and ho != hn:
        raise ValueError(f"hook_name mismatch: {ho!r} vs {hn!r}.")

    diag = latent_gradient_cosine_similarity(
        pass_orig.components.gradient_g,
        pass_new.components.gradient_g,
        eps=eps,
    )
    return float(diag.cosine_similarity.detach().cpu().item())


def latent_gradient_cosine_after_prune_wave(
    snapshot_before: LatentGradientSnapshot,
    *,
    model: Any,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    forced_zero_indices_after: AbstractSet[int],
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    subset_indices: Optional[Iterable[int]] = None,
    eps: float = 1e-12,
) -> LatentGradientCosineDiagnostics:
    """After pruning: one micro backward with the **new** mask; cosine vs ``snapshot_before.gradient_g``.

    Typical workflow:
        1. ``snap = capture_latent_gradient_snapshot(..., forced_zero_indices=R_before)``
        2. Update circuit ``R_after ⊇ R_before``
        3. ``latent_gradient_cosine_after_prune_wave(snap, ..., forced_zero_indices_after=R_after)``
    """
    if snapshot_before.hook_name and hook_name != snapshot_before.hook_name:
        raise ValueError(
            f"hook_name {hook_name!r} does not match snapshot {snapshot_before.hook_name!r}."
        )

    after = capture_latent_gradient_snapshot(
        model=model,
        prompt_corrupt=prompt_corrupt,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        seq_pos=snapshot_before.seq_pos_effective,
        prepend_bos=prepend_bos,
        device=device,
        forced_zero_indices=forced_zero_indices_after,
        metric=snapshot_before.metric,
    )

    return latent_gradient_cosine_similarity(
        snapshot_before.gradient_g,
        after.gradient_g,
        subset_indices=subset_indices,
        eps=eps,
    )
