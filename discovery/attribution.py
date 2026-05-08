"""Activation × gradient attribution on SAE latents at a TransformerLens hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

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
    """For Taylor passes: ``∂L/∂f`` at corrupt latents. For IG: weighted sum approximating ``∫ ∂L/∂f\\,dα``."""


@dataclass(frozen=True)
class SAEAttributionPass:
    """Full output of one attribution pass over SAE latents (+ residual ``e = x - \\hat{x}`` channel)."""

    indices: torch.Tensor
    """Latent indices ``0 … n_features-1`` (dtype ``long``, aligned with ``scores``)."""

    scores: torch.Tensor
    """Per-latent ``Δf ⊙ g`` (element-wise product); aligns with ``(Δx)·∇`` in feature coordinates."""

    residual_score: float
    """``⟨∇_x \\mathcal{L}, e_{clean} - e_{corrupt}⟩`` at the hook (scalar ``e`` / activation gradient dot)."""

    components: SAEAttributionComponents
    metadata: AttributionMetadata

    def get_completeness_report(self, actual_delta_loss: float) -> dict[str, float]:
        """Compare observed ``Δℒ`` to latent sum plus ``residual_score`` (linear decomposition diagnostic)."""

        latent_sum = float(self.scores.sum().detach().cpu().item())
        total_attributed = latent_sum + float(self.residual_score)
        error = abs(float(actual_delta_loss) - total_attributed)
        denom = total_attributed + 1e-9
        return {
            "actual_delta": float(actual_delta_loss),
            "total_attributed": float(total_attributed),
            "latent_contribution": latent_sum / denom,
            "residual_contribution": float(self.residual_score) / denom,
            "approximation_error": float(error),
        }


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


@dataclass(frozen=True)
class IGCompletenessDiagnostics:
    """Compare summed IG (latents + residual channel) to ``Δℒ`` on **unhooked** forwards."""

    metric_clean: float
    """Scalar ``ℒ(logits)`` on a plain forward of ``prompt_clean`` (no SAE intervention)."""

    metric_corrupt: float
    """Scalar ``ℒ(logits)`` on a plain forward of ``prompt_corrupt``."""

    delta_metric: float
    """``metric_clean - metric_corrupt``."""

    sum_latent_ig: float
    """``Σ_i IG_i`` over SAE latents only."""

    residual_score: float
    """Residual attribution ``⟨∇̄_x \\mathcal{L}, Δe⟩`` accumulated along the IG path."""

    total_attributed: float
    """``sum_latent_ig + residual_score``."""

    gap_abs: float
    """``|delta_metric - total_attributed|``. Remaining gap ⇒ higher-order / linearization error."""


def _scalar_metric_plain_forward(
    *,
    model: Any,
    tokens: torch.Tensor,
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """Evaluate ``logits_to_scalar_loss(model(tokens))`` without hooks (TransformerLens-style)."""

    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        return float(logits_to_scalar_loss(logits).detach().cpu().item())


def _encode_latents_at_hook_cached(
    *,
    model: Any,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    pos_effective: int,
) -> torch.Tensor:
    """Encode hook activation at ``pos_effective`` → 1D latent ``[n_features]`` (detached)."""

    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type="logits")
        if hook_name not in cache:
            raise KeyError(f"Cache missing {hook_name!r}.")
        act = cache[hook_name]
        f_raw = encode_fn(act)
        if f_raw.dim() == 2:
            f_raw = f_raw.unsqueeze(0)
        if f_raw.dim() < 3:
            raise ValueError(
                f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f_raw.shape)}"
            )
        seq_len = int(f_raw.shape[1])
        if pos_effective >= seq_len:
            raise IndexError(f"seq_pos_effective={pos_effective} out of range for seq_len={seq_len}")
        return f_raw[0, pos_effective, :].detach()


def _resolve_seq_pos_index(seq_pos: int, seq_len: int) -> int:
    pos_effective = int(seq_pos)
    if pos_effective < 0:
        pos_effective += seq_len
    if pos_effective < 0 or pos_effective >= seq_len:
        raise IndexError(f"seq_pos resolved to {pos_effective}, invalid for seq_len={seq_len}")
    return pos_effective


def _hook_residual_vector_at_pos(
    act_bpdm: torch.Tensor,
    *,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    pos_effective: int,
) -> torch.Tensor:
    """``e = x - \\hat{x}`` at ``pos_effective``; ``act_bpdm`` is ``[batch, pos, d_model]``."""

    f_raw = encode_fn(act_bpdm)
    if f_raw.dim() == 2:
        f_raw = f_raw.unsqueeze(0)
    if f_raw.dim() < 3:
        raise ValueError(
            f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f_raw.shape)}"
        )
    xhat = decode_fn(f_raw)
    if xhat.shape != act_bpdm.shape:
        raise ValueError(
            "decode_fn output must match activation shape. "
            f"Got xhat={tuple(xhat.shape)} vs act={tuple(act_bpdm.shape)}"
        )
    return (act_bpdm - xhat)[0, pos_effective, :].detach().reshape(-1)


def _residual_delta_e_clean_minus_corrupt(
    *,
    model: Any,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    pos_effective: int,
) -> torch.Tensor:
    """``e_clean - e_corrupt`` at ``hook_name``, shape ``[d_model]`` (detached)."""

    model.eval()
    with torch.no_grad():
        _, cache_cl = model.run_with_cache(
            clean_tokens,
            names_filter=[hook_name],
            return_type="logits",
        )
        _, cache_co = model.run_with_cache(
            corrupt_tokens,
            names_filter=[hook_name],
            return_type="logits",
        )
        if hook_name not in cache_cl or hook_name not in cache_co:
            raise KeyError(f"Missing {hook_name!r} in clean or corrupt cache.")
        e_cl = _hook_residual_vector_at_pos(
            cache_cl[hook_name],
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            pos_effective=pos_effective,
        )
        e_co = _hook_residual_vector_at_pos(
            cache_co[hook_name],
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            pos_effective=pos_effective,
        )
    return e_cl - e_co


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
    corrupt_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """One corrupt forward + backward.

    Uses ``act_r = act.detach().clone().requires_grad_(True)`` at the hook so ``∂ℒ/∂act`` is defined at the
    SAE boundary without relying on TransformerLens feeding differentiable hook inputs.

    Returns ``(f_corrupt, ∂L/∂f, seq_pos_effective, ∂L/∂act)`` with latent tensors shaped ``[n_features]``
    and activation gradient ``[d_model]`` at ``seq_pos``.
    """

    to_tokens_kwargs: Dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kwargs["prepend_bos"] = prepend_bos
    if corrupt_tokens is None:
        tokens = model.to_tokens(prompt_corrupt, **to_tokens_kwargs)
        if device is not None:
            tokens = tokens.to(device)
    else:
        tokens = corrupt_tokens

    feats_for_grad: torch.Tensor | None = None
    act_boundary: torch.Tensor | None = None
    forced = frozenset(forced_zero_indices) if forced_zero_indices else frozenset()

    def _grad_hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        nonlocal feats_for_grad, act_boundary
        act_r = act.detach().clone().requires_grad_(True)
        act_boundary = act_r
        g = encode_fn(act_r)
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
        if xhat_m.shape != act_r.shape or xhat_full.shape != act_r.shape:
            raise ValueError(
                "decode_fn output must match activation shape. "
                f"Got masked={tuple(xhat_m.shape)} full={tuple(xhat_full.shape)} vs act={tuple(act_r.shape)}"
            )
        return xhat_m + (act_r - xhat_full.detach())

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
    if act_boundary is None or act_boundary.grad is None:
        raise RuntimeError(
            f"Hook {hook_name!r} did not produce activation gradients on the SAE boundary tensor."
        )

    pos_effective = int(seq_pos)
    seq_dim = int(feats_for_grad.shape[1])
    if pos_effective < 0:
        pos_effective += seq_dim
    f_co = feats_for_grad[0, pos_effective, :]
    ggrad = feats_for_grad.grad[0, pos_effective, :]
    grad_act = act_boundary.grad[0, pos_effective, :].detach().reshape(-1)
    return f_co.detach(), ggrad.detach(), pos_effective, grad_act


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
    """Per-feature signed ``f ⊙ (∂L/∂f)`` at ``seq_pos`` on the corrupt run.

    Same graph as :func:`feature_attribution_pass` but **no** clean prompt: uses corrupt latents ``f`` and
    their gradient w.r.t. the scalar loss. Magnitude for ranking / pruning is applied in
    :mod:`discovery.pruner` (e.g. ``scores.abs()`` before ``topk`` / ``argsort``).
    """

    f_co, g, _pos_eff, _ga = _feat_grad_corrupt_forward(
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
    g_ = g.to(dtype=f_co.dtype, device=f_co.device)
    return (f_co * g_).detach()


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
    """Full clean-vs-corrupt attribution: ``scores_i = (Δf)_i · (∂L/∂f)_i`` plus residual ``⟨∇_xℒ, Δe⟩``.

    Runs corrupt forward + backward (same masked substitution as :func:`feature_act_grad_scores`),
    then a **no-grad** clean forward with ``run_with_cache`` to obtain ``f_clean`` at the hook.
    Residual ``e = x - \\hat{x}(f)`` uses plain encode/decode on cached hook activations.

    Args:
        metric: Stored in :class:`AttributionMetadata` (e.g. ``\"logit_diff\"``, ``\"kl_div_last_token\"``).
            Must match what ``logits_to_scalar_loss`` implements.
    """

    to_tokens_kwargs: Dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kwargs["prepend_bos"] = prepend_bos
    corrupt_tokens = model.to_tokens(prompt_corrupt, **to_tokens_kwargs)
    clean_tokens = model.to_tokens(prompt_clean, **to_tokens_kwargs)
    if device is not None:
        corrupt_tokens = corrupt_tokens.to(device)
        clean_tokens = clean_tokens.to(device)

    f_co, g, pos_effective, grad_act = _feat_grad_corrupt_forward(
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
        corrupt_tokens=corrupt_tokens,
    )

    seq_len = int(clean_tokens.shape[-1])
    if pos_effective >= seq_len:
        raise IndexError(f"seq_pos_effective={pos_effective} out of range for clean seq_len={seq_len}")

    delta_e = _residual_delta_e_clean_minus_corrupt(
        model=model,
        clean_tokens=clean_tokens,
        corrupt_tokens=corrupt_tokens,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        pos_effective=pos_effective,
    )

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

    residual_score = float((grad_act.to(device=delta_e.device, dtype=delta_e.dtype) * delta_e).sum().item())

    n_features = int(scores.numel())
    indices = torch.arange(n_features, device=scores.device, dtype=torch.long)

    meta = AttributionMetadata(metric=metric, seq_pos=pos_effective, hook_name=hook_name)
    components = SAEAttributionComponents(delta_f=delta_f.detach(), gradient_g=g.detach())

    return SAEAttributionPass(
        indices=indices,
        scores=scores.detach(),
        residual_score=residual_score,
        components=components,
        metadata=meta,
    )


@overload
def feature_integrated_gradients_pass(
    *,
    model: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    metric: str = "integrated_gradients",
    seq_pos: int = -1,
    n_steps: int = 20,
    ig_alpha_schedule: Literal["midpoint", "linspace", "trapezoidal"] = "midpoint",
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    forced_zero_indices: Optional[AbstractSet[int]] = None,
    empty_cache_between_steps: bool = False,
    check_completeness: Literal[False] = False,
) -> SAEAttributionPass: ...


@overload
def feature_integrated_gradients_pass(
    *,
    model: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    metric: str = "integrated_gradients",
    seq_pos: int = -1,
    n_steps: int = 20,
    ig_alpha_schedule: Literal["midpoint", "linspace", "trapezoidal"] = "midpoint",
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    forced_zero_indices: Optional[AbstractSet[int]] = None,
    empty_cache_between_steps: bool = False,
    check_completeness: Literal[True],
) -> Tuple[SAEAttributionPass, IGCompletenessDiagnostics]: ...


def feature_integrated_gradients_pass(
    *,
    model: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    metric: str = "integrated_gradients",
    seq_pos: int = -1,
    n_steps: int = 20,
    ig_alpha_schedule: Literal["midpoint", "linspace", "trapezoidal"] = "midpoint",
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    forced_zero_indices: Optional[AbstractSet[int]] = None,
    empty_cache_between_steps: bool = False,
    check_completeness: bool = False,
) -> Union[SAEAttributionPass, Tuple[SAEAttributionPass, IGCompletenessDiagnostics]]:
    """Integrated Gradients on **SAE latents** ``f`` (not raw hook activations ``x``) at ``seq_pos``.

    Interpolation is applied to the latent vector ``f`` at one sequence position; the hook injects that
    choice through ``decode_fn`` while preserving the corrupt residual term (same substitution semantics as
    :func:`_feat_grad_corrupt_forward` / :func:`discovery.sae_scout.reconstruct_activation`). Other positions
    keep ``encode_fn(act)`` from the corrupt forward (detached); gradients flow only through the injected
    ``f(α)`` slice.

    If ``encode_fn`` encodes a **non-linear** map from ``x → f`` (e.g. JumpReLU / Top-K inside the encoder),
    the straight-line path ``f_corrupt → f_clean`` in ``f``-space can cut across discontinuities in the
    true ``x → f`` composition — IG along ``f`` is still well-defined for ℒ as a function of injected latents,
    but may disagree with paths defined in ``x``.

    Discrete quadrature for ``∫_0^1 ∂ℒ/∂f_i|_{f(α)} dα``:

    - ``midpoint``: ``n_steps`` evaluations at ``(k+½)/n_steps``, weights ``1/n_steps`` (often efficient per step).
    - ``linspace``: ``n_steps`` evaluations at ``torch.linspace(0, 1, n_steps)``, uniform weights ``1/n_steps``.
    - ``trapezoidal``: ``n_steps`` **subintervals**, ``n_steps + 1`` evaluations at boundaries ``k/n_steps``,
      composite trapezoid weights (endpoints half-weight). Better alignment with endpoint-inclusive Riemann
      intuition when checking completeness with ``check_completeness``.

    ``scores_i = (Δf)_i × \\hat{∫} ∂ℒ/∂f_i\\, dα`` where ``Δf = f_clean - f_corrupt`` and ``\\hat{∫}`` is the
    weighted gradient sum (stored in ``components.gradient_g``).

    **Residual / completeness:** only latent coordinates are attributed; SAE residual is not a summed
    dimension, so ``Σ_i IG_i`` need not equal plain ``Δℒ`` (see ``check_completeness``).

    Args:
        n_steps: For ``midpoint`` / ``linspace``: number of gradient evaluations. For ``trapezoidal``: number
            of subintervals (``n_steps + 1`` evaluations).
        ig_alpha_schedule: Quadrature rule (see above).
        empty_cache_between_steps: If True, call ``torch.cuda.empty_cache()`` after each backward (slower;
            can reduce peak VRAM when ``n_steps`` is large).
        check_completeness: If True, also return :class:`IGCompletenessDiagnostics` comparing ``Δℒ_natural``
            to ``Σ_i IG_i + residual_score`` (plain forwards for ``ℒ``, latent IG plus averaged ``∇_xℒ`` dot
            ``Δe``). Large ``gap_abs`` implicates curvature / linearization beyond missing residual attribution.
    """

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    to_tokens_kwargs: Dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kwargs["prepend_bos"] = prepend_bos

    corrupt_tokens = model.to_tokens(prompt_corrupt, **to_tokens_kwargs)
    clean_tokens = model.to_tokens(prompt_clean, **to_tokens_kwargs)
    if device is not None:
        corrupt_tokens = corrupt_tokens.to(device)
        clean_tokens = clean_tokens.to(device)

    metric_clean_natural: Optional[float] = None
    metric_corrupt_natural: Optional[float] = None
    if check_completeness:
        metric_clean_natural = _scalar_metric_plain_forward(
            model=model,
            tokens=clean_tokens,
            logits_to_scalar_loss=logits_to_scalar_loss,
        )
        metric_corrupt_natural = _scalar_metric_plain_forward(
            model=model,
            tokens=corrupt_tokens,
            logits_to_scalar_loss=logits_to_scalar_loss,
        )

    model.eval()
    with torch.no_grad():
        _, corrupt_cache = model.run_with_cache(
            corrupt_tokens,
            names_filter=[hook_name],
            return_type="logits",
        )
        if hook_name not in corrupt_cache:
            raise KeyError(f"Corrupt cache missing {hook_name!r}.")
        act_co_ref = corrupt_cache[hook_name]
        seq_len_co = int(act_co_ref.shape[1])
        pos_effective = _resolve_seq_pos_index(seq_pos, seq_len_co)

    seq_len_cl = int(clean_tokens.shape[-1])
    if pos_effective >= seq_len_cl:
        raise IndexError(f"seq_pos_effective={pos_effective} out of range for clean seq_len={seq_len_cl}")

    f_corrupt = _encode_latents_at_hook_cached(
        model=model,
        tokens=corrupt_tokens,
        hook_name=hook_name,
        encode_fn=encode_fn,
        pos_effective=pos_effective,
    )
    f_clean = _encode_latents_at_hook_cached(
        model=model,
        tokens=clean_tokens,
        hook_name=hook_name,
        encode_fn=encode_fn,
        pos_effective=pos_effective,
    )

    delta_f = f_clean - f_corrupt
    dev = delta_f.device
    dt = delta_f.dtype
    forced = frozenset(forced_zero_indices) if forced_zero_indices else frozenset()

    delta_e = _residual_delta_e_clean_minus_corrupt(
        model=model,
        clean_tokens=clean_tokens,
        corrupt_tokens=corrupt_tokens,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        pos_effective=pos_effective,
    )
    gx_accum = torch.zeros(int(delta_e.numel()), dtype=torch.float32, device=delta_e.device)

    if ig_alpha_schedule == "midpoint":
        alphas = (torch.arange(n_steps, device=dev, dtype=torch.float64) + 0.5) / float(n_steps)
        weights = torch.full((n_steps,), 1.0 / float(n_steps), device=dev, dtype=torch.float64)
    elif ig_alpha_schedule == "linspace":
        alphas = torch.linspace(0.0, 1.0, n_steps, device=dev, dtype=torch.float64)
        weights = torch.full((n_steps,), 1.0 / float(n_steps), device=dev, dtype=torch.float64)
    elif ig_alpha_schedule == "trapezoidal":
        alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=dev, dtype=torch.float64)
        h = 1.0 / float(n_steps)
        w_mid = [h] * max(0, n_steps - 1)
        w_list = [h / 2.0] + w_mid + [h / 2.0]
        weights = torch.tensor(w_list, device=dev, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown ig_alpha_schedule={ig_alpha_schedule!r}.")

    if alphas.numel() != weights.numel():
        raise RuntimeError("internal error: alphas and weights length mismatch")

    g_accum = torch.zeros_like(delta_f, dtype=torch.float32)

    for step_i in range(int(alphas.numel())):
        alpha = float(alphas[step_i].item())
        w_step = float(weights[step_i].item())
        model.zero_grad(set_to_none=True)

        f_alpha = f_corrupt + delta_f * alpha
        f_leaf = f_alpha.detach().clone().requires_grad_(True)

        hook_act_r: Optional[torch.Tensor] = None

        def _ig_hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
            nonlocal hook_act_r
            act_r = act.detach().clone().requires_grad_(True)
            hook_act_r = act_r
            g_cor = encode_fn(act_r)
            if g_cor.dim() == 2:
                g_cor = g_cor.unsqueeze(0)
            if g_cor.dim() < 3:
                raise ValueError(
                    f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(g_cor.shape)}"
                )
            g_cor_det = g_cor.detach()
            g_work = g_cor_det.clone()
            g_work[0, pos_effective, :] = f_leaf.to(dtype=g_work.dtype, device=g_work.device)
            if forced:
                idx = torch.tensor(sorted(forced), device=g_work.device, dtype=torch.long)
                g_work.index_fill_(-1, idx, 0.0)

            xhat_m = decode_fn(g_work)
            xhat_full = decode_fn(g_cor_det)
            if xhat_m.shape != act_r.shape or xhat_full.shape != act_r.shape:
                raise ValueError(
                    "decode_fn output must match activation shape. "
                    f"Got masked={tuple(xhat_m.shape)} full={tuple(xhat_full.shape)} vs act={tuple(act_r.shape)}"
                )
            return xhat_m + (act_r - xhat_full.detach())

        with torch.enable_grad():
            logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, _ig_hook)],
                return_type="logits",
            )
            loss = logits_to_scalar_loss(logits)
            loss.backward()

        if f_leaf.grad is None:
            raise RuntimeError(
                f"Integrated gradients step (alpha={alpha}) did not produce gradients on f_leaf "
                f"(hook {hook_name!r} or loss graph?)."
            )
        if hook_act_r is None or hook_act_r.grad is None:
            raise RuntimeError(
                f"Integrated gradients step (alpha={alpha}) did not produce ∂ℒ/∂act on hook boundary "
                f"(hook {hook_name!r})."
            )
        gx_step = hook_act_r.grad[0, pos_effective, :].detach().reshape(-1).float()
        gx_accum += gx_step * float(w_step)
        g_accum += f_leaf.grad.detach().float() * float(w_step)

        del loss, logits, f_leaf
        if empty_cache_between_steps and torch.cuda.is_available():
            torch.cuda.empty_cache()

    g_integral_hat = g_accum.to(dtype=dt, device=dev)
    delta_f_det = delta_f.detach()
    scores = delta_f_det * g_integral_hat

    n_features = int(scores.numel())
    indices = torch.arange(n_features, device=scores.device, dtype=torch.long)

    meta = AttributionMetadata(metric=metric, seq_pos=pos_effective, hook_name=hook_name)
    components = SAEAttributionComponents(delta_f=delta_f_det, gradient_g=g_integral_hat.detach())

    residual_ig = float((gx_accum * delta_e.float()).sum().item())

    pass_out = SAEAttributionPass(
        indices=indices,
        scores=scores.detach(),
        residual_score=residual_ig,
        components=components,
        metadata=meta,
    )

    if check_completeness:
        if metric_clean_natural is None or metric_corrupt_natural is None:
            raise RuntimeError("Completeness metrics missing despite check_completeness=True.")
        sum_lat = float(pass_out.scores.sum().detach().cpu().item())
        delta_m = metric_clean_natural - metric_corrupt_natural
        total_attr = sum_lat + residual_ig
        gap_abs = abs(delta_m - total_attr)
        diag = IGCompletenessDiagnostics(
            metric_clean=metric_clean_natural,
            metric_corrupt=metric_corrupt_natural,
            delta_metric=delta_m,
            sum_latent_ig=sum_lat,
            residual_score=residual_ig,
            total_attributed=total_attr,
            gap_abs=gap_abs,
        )
        return pass_out, diag

    return pass_out


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

    _f_co, g, pos_eff, _ga = _feat_grad_corrupt_forward(
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
