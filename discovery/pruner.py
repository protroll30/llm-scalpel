"""SAE circuit pruning with activation×gradient ranking and KL control.

Two modes:

- **Threshold** (:func:`prune_sae_circuit`): fixed greedy order from one attribution pass,
  remove lowest-ranked latents while ``KL(reference || masked) ≤ τ`` per latent.

- **Budget + refresh** (:func:`prune_sae_circuit_budget`): outer loop on a total
  ``KL(reference || current_masked)`` budget; inner loop takes the bottom ``N``
  latents by fresh scores and **shrinks failures by binary-splitting** batches
  until only singleton refusals stop progress (toward the budget-limited causal frontier).

Optional **gradient drift** gating (when ``prompt_clean`` is set, or ``ExperimentRunner.clean_prompt``):
before each wave snapshot ``pass_orig = feature_attribution_pass(...)`` at the current mask; each trial
mask ``pass_new`` must satisfy ``calculate_gradient_drift(pass_orig, pass_new) ≥`` a chunk-size-aware
threshold (large chunks stricter, small chunks ``len < 5`` looser or optionally skipped) **and** KL
budget — otherwise the wave is rejected / subdivided like a KL failure. During recursive splitting,
``pass_orig`` is **realigned** when the recursive baseline ``removed`` differs from the mask it was
computed at; after a drift failure on unchanged ``removed``, the baseline is **refreshed** once
(re-run attribution at ``removed``) before splitting to reduce unnecessary subdivision.

**Ranking vs drift:** use optional :func:`discovery.attribution.feature_integrated_gradients_pass` only for
**outer-loop** importance ordering (expensive); keep drift checks on **single-point**
:func:`discovery.attribution.feature_attribution_pass` — cheaper and usually sufficient to detect local
geometry breakdown inside recursive batching.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Set, Union

import torch
import torch.nn.functional as F

from discovery.attribution import (
    SAEAttributionPass,
    calculate_gradient_drift,
    feature_act_grad_scores,
    feature_attribution_pass,
    feature_integrated_gradients_pass,
)
from discovery.sae_scout import reconstruct_activation

if TYPE_CHECKING:
    from causal_patcher.runner import ExperimentRunner


def _logits_with_feature_mask(
    *,
    model: Any,
    tokens: torch.Tensor,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    zero_indices: Set[int],
) -> torch.Tensor:
    """Corrupt-style forward: replace hook output with residual-aware SAE reconstruction.

    Latents in ``zero_indices`` are set to zero before decode; the residual term still
    uses the full ``f_corrupt = encode(act)``.
    """

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        f = encode_fn(act)
        if f.dim() == 2:
            f = f.unsqueeze(0)
        if f.dim() < 3:
            raise ValueError(
                f"encode_fn must return [pos, n_features] or [batch, pos, n_features]; got {tuple(f.shape)}"
            )
        f_full = f
        if zero_indices:
            f_masked = f_full.clone()
            idx = torch.tensor(sorted(zero_indices), device=f_full.device, dtype=torch.long)
            f_masked.index_fill_(-1, idx, 0.0)
        else:
            f_masked = f_full
        return reconstruct_activation(
            f_patched=f_masked,
            x_corrupt=act,
            f_corrupt=f_full,
            decode_fn=decode_fn,
        )

    return model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, _hook)],
        return_type="logits",
    )


def kl_last_token_divergence(
    logits_reference: torch.Tensor,
    logits_trial: torch.Tensor,
    *,
    seq_pos: int = -1,
    reduction: str = "sum",
) -> torch.Tensor:
    """``KL(p_ref || p_trial)`` on the vocabulary distribution at ``seq_pos`` (per batch row)."""
    if logits_reference.dim() != 3 or logits_trial.dim() != 3:
        raise ValueError(
            f"Expected 3D logits [batch, pos, vocab]; got ref {tuple(logits_reference.shape)} "
            f"trial {tuple(logits_trial.shape)}"
        )
    lr = logits_reference[:, seq_pos, :]
    lt = logits_trial[:, seq_pos, :]
    p = F.softmax(lr.float(), dim=-1)
    log_q = F.log_softmax(lt.float(), dim=-1)
    return F.kl_div(log_q, p, reduction=reduction, log_target=False)


def _kl_vs_reference_below_budget(
    *,
    logits_reference: torch.Tensor,
    logits_trial: torch.Tensor,
    kl_budget: float,
    seq_pos: int,
) -> bool:
    kl = kl_last_token_divergence(logits_reference, logits_trial, seq_pos=seq_pos, reduction="sum")
    return float(kl.detach().cpu().item()) <= kl_budget


def _budget_try_remove_recursive(
    *,
    model: Any,
    logits_reference: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    removed: Set[int],
    chunk: list[int],
    kl_budget: float,
    seq_pos: int,
    drift_gate: Optional[WaveGradientDriftGate] = None,
) -> Set[int]:
    """Return an updated ``removed`` set after trying to zero all ``chunk`` latents together.

    If ``KL(reference || masked)`` forbids removing the whole chunk, split roughly in half (~N/2),
    recurse on the **first** half against ``removed``, then on the **second** half against that
    result. Only a **singleton** chunk that fails is dropped (cannot cross the budget here).

    If ``drift_gate`` is set and KL passes, a trial must also satisfy
    ``calculate_gradient_drift(pass_orig, pass_new) >= drift_gate.threshold_for_chunk(len(chunk))``
    (unless small chunks are configured to bypass), where ``pass_new`` is a
    :func:`feature_attribution_pass` at ``trial`` mask; otherwise the gate baseline may be refreshed
    once at the current ``removed`` set and the drift check retried before the chunk is subdivided.

    When recursion continues after the left half succeeds, ``pass_orig`` is updated if needed so it
    matches attribution at the **current** ``removed`` set (avoids a stale linear baseline).
    """

    if not chunk:
        return set(removed)

    if drift_gate is not None:
        drift_gate = _align_drift_gate_to_removed_if_needed(
            drift_gate=drift_gate,
            removed=removed,
            model=model,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
        )

    trial = removed | set(chunk)
    with torch.no_grad():
        logits_r = _logits_with_feature_mask(
            model=model,
            tokens=corrupt_tokens,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            zero_indices=trial,
        )
    kl_ok = _kl_vs_reference_below_budget(
        logits_reference=logits_reference,
        logits_trial=logits_r,
        kl_budget=kl_budget,
        seq_pos=seq_pos,
    )
    if kl_ok:
        if drift_gate is None:
            return trial
        if drift_gate.skip_gradient_drift(len(chunk)):
            return trial

        def _trial_passes_drift(gate: WaveGradientDriftGate) -> bool:
            pass_new = feature_attribution_pass(
                model=model,
                prompt_clean=gate.prompt_clean,
                prompt_corrupt=gate.corrupt_prompt,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                metric=gate.metric,
                seq_pos=gate.attribution_seq_pos,
                prepend_bos=gate.prepend_bos,
                device=gate.device,
                forced_zero_indices=trial,
            )
            drift = calculate_gradient_drift(gate.pass_orig, pass_new)
            thr = gate.threshold_for_chunk(len(chunk))
            return not math.isnan(drift) and drift >= thr

        if _trial_passes_drift(drift_gate):
            return trial
        drift_gate = _refresh_drift_gate_pass_orig_at_removed(
            drift_gate=drift_gate,
            removed=removed,
            model=model,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
        )
        if _trial_passes_drift(drift_gate):
            return trial
    if len(chunk) == 1:
        return set(removed)
    mid = len(chunk) // 2 or 1
    left = chunk[:mid]
    right = chunk[mid:]
    rl = _budget_try_remove_recursive(
        model=model,
        logits_reference=logits_reference,
        corrupt_tokens=corrupt_tokens,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        removed=set(removed),
        chunk=left,
        kl_budget=kl_budget,
        seq_pos=seq_pos,
        drift_gate=drift_gate,
    )
    return _budget_try_remove_recursive(
        model=model,
        logits_reference=logits_reference,
        corrupt_tokens=corrupt_tokens,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        removed=rl,
        chunk=right,
        kl_budget=kl_budget,
        seq_pos=seq_pos,
        drift_gate=drift_gate,
    )


@dataclass
class Circuit:
    """Minimal circuit export: surviving SAE latents after pruning."""

    hook_name: str
    feature_indices: list[int]
    n_features_initial: int
    removed_indices: list[int] = field(default_factory=list)
    attribution_scores: Optional[list[float]] = None
    tau: Optional[float] = None
    kl_budget: Optional[float] = None
    batch_remove_n: Optional[int] = None
    final_kl_vs_reference: Optional[float] = None


@dataclass(frozen=True)
class WaveGradientDriftGate:
    """Holds ``pass_orig`` and settings for wave validation via :func:`calculate_gradient_drift`."""

    pass_orig: SAEAttributionPass
    pass_orig_mask: frozenset[int]
    prompt_clean: str
    corrupt_prompt: str
    metric: str
    attribution_seq_pos: int
    threshold_large: float
    threshold_small: float
    small_chunk_max_exclusive: int
    bypass_small_chunks: bool
    prepend_bos: Optional[bool]
    device: Optional[Union[str, torch.device]]

    def threshold_for_chunk(self, chunk_len: int) -> float:
        if chunk_len < self.small_chunk_max_exclusive:
            return self.threshold_small
        return self.threshold_large

    def skip_gradient_drift(self, chunk_len: int) -> bool:
        return self.bypass_small_chunks and chunk_len < self.small_chunk_max_exclusive


def _refresh_drift_gate_pass_orig_at_removed(
    *,
    drift_gate: WaveGradientDriftGate,
    removed: Set[int],
    model: Any,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
) -> WaveGradientDriftGate:
    """New gate with ``pass_orig`` recomputed at ``removed`` (even if mask unchanged)."""

    mask_f = frozenset(removed)
    pass_orig = feature_attribution_pass(
        model=model,
        prompt_clean=drift_gate.prompt_clean,
        prompt_corrupt=drift_gate.corrupt_prompt,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        metric=drift_gate.metric,
        seq_pos=drift_gate.attribution_seq_pos,
        prepend_bos=drift_gate.prepend_bos,
        device=drift_gate.device,
        forced_zero_indices=set(removed),
    )
    return replace(drift_gate, pass_orig=pass_orig, pass_orig_mask=mask_f)


def _align_drift_gate_to_removed_if_needed(
    *,
    drift_gate: WaveGradientDriftGate,
    removed: Set[int],
    model: Any,
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
) -> WaveGradientDriftGate:
    """If ``pass_orig`` was computed at a different mask than ``removed``, re-run attribution at ``removed``."""

    if drift_gate.pass_orig_mask == frozenset(removed):
        return drift_gate
    return _refresh_drift_gate_pass_orig_at_removed(
        drift_gate=drift_gate,
        removed=removed,
        model=model,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
    )


def _resolve_corrupt_tokens_and_prompt(
    *,
    model: Any,
    corrupt_prompt: Optional[str],
    prepend_bos: Optional[bool],
    device: Optional[Union[str, torch.device]],
    experiment: Optional["ExperimentRunner"],
) -> tuple[torch.Tensor, str]:
    if experiment is not None:
        if corrupt_prompt is not None:
            raise ValueError("Pass corrupt_prompt=None when experiment is provided.")
        exp = experiment
        if exp.corrupt_tokens is None:
            raise RuntimeError("experiment.corrupt_tokens is None; run experiment.run_baselines() first.")
        corrupt_tokens = exp.corrupt_tokens
        if device is not None:
            corrupt_tokens = corrupt_tokens.to(device)
        return corrupt_tokens, exp.corrupt_prompt

    if corrupt_prompt is None:
        raise ValueError("corrupt_prompt is required when experiment is None.")
    to_tokens_kw: dict[str, Any] = {}
    if prepend_bos is not None:
        to_tokens_kw["prepend_bos"] = prepend_bos
    corrupt_tokens = model.to_tokens(corrupt_prompt, **to_tokens_kw)
    if device is not None:
        corrupt_tokens = corrupt_tokens.to(device)
    return corrupt_tokens, corrupt_prompt


def prune_sae_circuit_budget(
    *,
    model: Any,
    corrupt_prompt: Optional[str],
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    kl_budget: float,
    batch_remove_n: int,
    n_features: Optional[int] = None,
    seq_pos: int = -1,
    attribution_seq_pos: int = -1,
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    experiment: Optional["ExperimentRunner"] = None,
    prompt_clean: Optional[str] = None,
    gradient_drift_threshold_large: float = 0.90,
    gradient_drift_threshold_small: float = 0.70,
    gradient_drift_small_chunk_max_exclusive: int = 5,
    gradient_drift_bypass_small_chunks: bool = False,
    drift_attribution_metric: str = "logit_diff",
    ranking_mode: Literal["act_grad", "integrated_gradients"] = "act_grad",
    ig_n_steps: int = 20,
    ig_alpha_schedule: Literal["midpoint", "linspace", "trapezoidal"] = "midpoint",
    ig_empty_cache_between_steps: bool = False,
    drift_residual_mass_warn_fraction: Optional[float] = None,
) -> Circuit:
    """Budgeted pruning with **re-attribution** each outer step.

    **Outer loop**: continue while ``KL(p_ref ‖ p_masked) < kl_budget`` for the masked
    graph at the **start** of the iteration (same reference logits as τ-mode: full latent graph).

    **Inner loop**: with removed set ``R``, compute per-feature scores (see ``ranking_mode``) and take the
    **bottom** ``min(N, |alive|)`` by score as ``chunk``. Try ``R ∪ chunk`` via
    :func:`_budget_try_remove_recursive`: if KL fails on the batch, split ~in half recursively
    (**left** subtree first on ``R``, then **right** on the updated removal set), refusing only when
    a **singleton** cannot be merged into ``R`` — then continue the outer sweep with fresh scores.

    **Recalculate**: one scoring pass each outer iteration on the **current** mask.

    **Ranking** (``ranking_mode``): default ``act_grad`` uses
    :func:`discovery.attribution.feature_act_grad_scores` (signed ``f ⊙ ∂L/∂f``); the pruner ranks by
    ``|score|`` before batching removes. ``integrated_gradients`` uses
    :func:`discovery.attribution.feature_integrated_gradients_pass` once per outer step (slow); ranks by
    ``|IG_i|`` so ordering matches magnitude-style salience. Requires ``prompt_clean`` or
    ``ExperimentRunner.clean_prompt``. Drift gating (below) always uses single-point
    :func:`discovery.attribution.feature_attribution_pass`, not IG.

    **Residual mass** (optional): if ``drift_residual_mass_warn_fraction`` is set (e.g. ``0.4``), emit
    ``warnings.warn`` when ``|residual_score| / (|Σ latent scores| + |residual_score|)`` from the drift
    baseline pass exceeds it — a coarse hint that the SAE leaves task-relevant signal in ``e``.

    **Gradient drift** (optional): pass ``prompt_clean`` or use :class:`ExperimentRunner` (uses
    ``clean_prompt`` when ``prompt_clean`` is omitted). Each accepted removal batch must satisfy
    ``calculate_gradient_drift(pass_orig, pass_new) >=`` a cosine threshold that depends on recursive
    chunk size: ``gradient_drift_threshold_large`` (default ``0.90``) when
    ``len(chunk) >= gradient_drift_small_chunk_max_exclusive``, else ``gradient_drift_threshold_small``
    (default ``0.70``). Set ``gradient_drift_bypass_small_chunks=True`` to skip the drift check entirely
    for chunks smaller than that cutoff (KL still applies).
    """
    if batch_remove_n < 1:
        raise ValueError("batch_remove_n must be >= 1.")
    if kl_budget < 0:
        raise ValueError("kl_budget must be non-negative.")
    if gradient_drift_small_chunk_max_exclusive < 1:
        raise ValueError("gradient_drift_small_chunk_max_exclusive must be >= 1.")
    if ig_n_steps < 1:
        raise ValueError("ig_n_steps must be >= 1.")
    if ranking_mode not in ("act_grad", "integrated_gradients"):
        raise ValueError(f"Unknown ranking_mode={ranking_mode!r}.")

    corrupt_tokens, corrupt_prompt_for_attr = _resolve_corrupt_tokens_and_prompt(
        model=model,
        corrupt_prompt=corrupt_prompt,
        prepend_bos=prepend_bos,
        device=device,
        experiment=experiment,
    )

    model.eval()
    with torch.no_grad():
        logits_ref = _logits_with_feature_mask(
            model=model,
            tokens=corrupt_tokens,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            zero_indices=set(),
        )

    removed: Set[int] = set()
    last_scores: Optional[torch.Tensor] = None
    nf = int(n_features) if n_features is not None else -1

    while True:
        with torch.no_grad():
            logits_now = _logits_with_feature_mask(
                model=model,
                tokens=corrupt_tokens,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                zero_indices=removed,
            )
        kl_now = float(
            kl_last_token_divergence(logits_ref, logits_now, seq_pos=seq_pos, reduction="sum")
            .detach()
            .cpu()
            .item()
        )
        if kl_now >= kl_budget:
            break

        clean_for_rank = prompt_clean
        if clean_for_rank is None and experiment is not None:
            clean_for_rank = experiment.clean_prompt

        if ranking_mode == "act_grad":
            last_scores = feature_act_grad_scores(
                model=model,
                prompt=corrupt_prompt_for_attr,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                seq_pos=attribution_seq_pos,
                prepend_bos=prepend_bos,
                device=device,
                forced_zero_indices=removed,
            )
        else:
            if clean_for_rank is None:
                raise ValueError(
                    "ranking_mode='integrated_gradients' requires prompt_clean or ExperimentRunner.clean_prompt."
                )
            ig_pass = feature_integrated_gradients_pass(
                model=model,
                prompt_clean=clean_for_rank,
                prompt_corrupt=corrupt_prompt_for_attr,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                metric=f"{drift_attribution_metric}_ig_rank",
                seq_pos=attribution_seq_pos,
                n_steps=ig_n_steps,
                ig_alpha_schedule=ig_alpha_schedule,
                prepend_bos=prepend_bos,
                device=device,
                forced_zero_indices=removed,
                empty_cache_between_steps=ig_empty_cache_between_steps,
            )
            last_scores = ig_pass.scores
        inferred = int(last_scores.numel())
        if nf < 0:
            nf = inferred
        elif inferred != nf:
            raise ValueError(
                f"n_features={nf} but attribution returned {inferred} scores; align encode width."
            )

        alive_count = nf - len(removed)
        if alive_count <= 0:
            break

        removed_before = set(removed)

        # IMPORTANT: keep ranking on-device to avoid GPU↔CPU sync from repeated `.item()` calls.
        # We only materialize a small Python list of size `batch_remove_n` for the recursive remover.
        k = min(batch_remove_n, alive_count)
        assert last_scores is not None
        rank_scores = last_scores.abs()
        if removed:
            idx = torch.tensor(list(removed), device=rank_scores.device, dtype=torch.long)
            masked = rank_scores.clone()
            masked.index_fill_(0, idx, float("inf"))
        else:
            masked = rank_scores
        chunk_t = torch.topk(masked, k=k, largest=False).indices
        chunk = [int(i) for i in chunk_t.detach().cpu().tolist()]

        clean_for_drift = prompt_clean
        if clean_for_drift is None and experiment is not None:
            clean_for_drift = experiment.clean_prompt

        drift_gate: Optional[WaveGradientDriftGate] = None
        if clean_for_drift is not None:
            pass_orig = feature_attribution_pass(
                model=model,
                prompt_clean=clean_for_drift,
                prompt_corrupt=corrupt_prompt_for_attr,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                metric=drift_attribution_metric,
                seq_pos=attribution_seq_pos,
                prepend_bos=prepend_bos,
                device=device,
                forced_zero_indices=removed,
            )
            if drift_residual_mass_warn_fraction is not None:
                lat_mag = abs(float(pass_orig.scores.sum().detach().cpu().item()))
                res_mag = abs(float(pass_orig.residual_score))
                denom = lat_mag + res_mag + 1e-12
                mass_frac = res_mag / denom
                if mass_frac >= drift_residual_mass_warn_fraction:
                    warnings.warn(
                        "Residual attribution mass fraction is high (~"
                        f"{mass_frac:.1%}); latent-only pruning may miss "
                        "'dark matter' carried by e = x - decode(f).",
                        UserWarning,
                        stacklevel=2,
                    )
            drift_gate = WaveGradientDriftGate(
                pass_orig=pass_orig,
                pass_orig_mask=frozenset(removed),
                prompt_clean=clean_for_drift,
                corrupt_prompt=corrupt_prompt_for_attr,
                metric=drift_attribution_metric,
                attribution_seq_pos=attribution_seq_pos,
                threshold_large=gradient_drift_threshold_large,
                threshold_small=gradient_drift_threshold_small,
                small_chunk_max_exclusive=gradient_drift_small_chunk_max_exclusive,
                bypass_small_chunks=gradient_drift_bypass_small_chunks,
                prepend_bos=prepend_bos,
                device=device,
            )

        removed = _budget_try_remove_recursive(
            model=model,
            logits_reference=logits_ref,
            corrupt_tokens=corrupt_tokens,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
            removed=removed,
            chunk=chunk,
            kl_budget=kl_budget,
            seq_pos=seq_pos,
            drift_gate=drift_gate,
        )
        if removed == removed_before:
            break

    nf_final = nf if nf >= 0 else -1
    if nf_final < 0:
        if last_scores is not None:
            nf_final = int(last_scores.numel())
        else:
            probe = feature_act_grad_scores(
                model=model,
                prompt=corrupt_prompt_for_attr,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                seq_pos=attribution_seq_pos,
                prepend_bos=prepend_bos,
                device=device,
                forced_zero_indices=removed,
            )
            last_scores = probe
            nf_final = int(probe.numel())

    with torch.no_grad():
        logits_final = _logits_with_feature_mask(
            model=model,
            tokens=corrupt_tokens,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            zero_indices=removed,
        )
    final_kl = float(
        kl_last_token_divergence(logits_ref, logits_final, seq_pos=seq_pos, reduction="sum")
        .detach()
        .cpu()
        .item()
    )

    attr_list = [float(last_scores[i].item()) for i in range(nf_final)] if last_scores is not None else None

    return Circuit(
        hook_name=hook_name,
        feature_indices=sorted(set(range(nf_final)) - removed),
        n_features_initial=nf_final,
        removed_indices=sorted(removed),
        attribution_scores=attr_list,
        tau=None,
        kl_budget=kl_budget,
        batch_remove_n=batch_remove_n,
        final_kl_vs_reference=final_kl,
    )


def prune_sae_circuit(
    *,
    model: Any,
    corrupt_prompt: Optional[str],
    hook_name: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
    tau: float,
    n_features: Optional[int] = None,
    seq_pos: int = -1,
    attribution_seq_pos: int = -1,
    prepend_bos: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
    experiment: Optional["ExperimentRunner"] = None,
) -> Circuit:
    """Carve an SAE circuit by recursive lowest-attribution-first removal under a KL cap.

    1. Full graph = all latent indices ``0 .. n_features-1`` (inferred from scores if omitted).
    2. Attribution pass: signed ``f ⊙ ∂L/∂f`` via :func:`~discovery.attribution.feature_act_grad_scores`;
       removal order uses ``|score|`` (lowest magnitude first).
    3. Greedily process features from **lowest** to highest salience (fixed order): tentatively zero
       that latent together with latents already removed; if ``KL(p_ref_last || p_try_last)``
       weighted by reduction ``sum`` is **≤ τ**, commit the removal.

    Args:
        model: TransformerLens-compatible ``HookedTransformer``.
        corrupt_prompt: Text for attribution and KL (ignored if ``experiment`` is passed; then the
            runner's ``corrupt_prompt`` / ``corrupt_tokens`` are implied via ``experiment`` context).
        hook_name: Activation hook (e.g. ``blocks.L.hook_resid_pre``).
        encode_fn / decode_fn: Differentiable SAE interface on hook activations.
        logits_to_scalar_loss: Loss for attribution backward (same signature as ``feature_act_grad_scores``).
        tau: Allowed KL ``KL(reference || masked)`` on the last-token distribution (summed over vocab).
        n_features: If ``None``, use ``score.numel()`` from attribution.
        seq_pos / attribution_seq_pos: Last-token index for KL and for attribution slicing (defaults ``-1``).
        experiment: Optional :class:`causal_patcher.runner.ExperimentRunner`; when set use
            ``corrupt_prompt=None``. ``corrupt_tokens`` must exist (run ``run_baselines()``).

    Returns:
        :class:`Circuit` listing **kept** feature indices plus metadata.
    """

    corrupt_tokens, corrupt_prompt_for_attr = _resolve_corrupt_tokens_and_prompt(
        model=model,
        corrupt_prompt=corrupt_prompt,
        prepend_bos=prepend_bos,
        device=device,
        experiment=experiment,
    )

    scores = feature_act_grad_scores(
        model=model,
        prompt=corrupt_prompt_for_attr,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        seq_pos=attribution_seq_pos,
        prepend_bos=prepend_bos,
        device=device,
    )
    nf = int(n_features) if n_features is not None else int(scores.numel())
    if scores.numel() != nf:
        raise ValueError(
            f"n_features={nf} but attribution returned {scores.numel()} scores; align encode width."
        )

    model.eval()
    with torch.no_grad():
        logits_ref = _logits_with_feature_mask(
            model=model,
            tokens=corrupt_tokens,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            zero_indices=set(),
        )

    # Sort on-device by |score| (least salient first); scores are signed act×grad.
    order_t = torch.argsort(scores.detach().abs())
    order = [int(i) for i in order_t.detach().cpu().tolist()]
    removed: Set[int] = set()

    for j in order:
        trial_removed = removed | {j}
        with torch.no_grad():
            logits_trial = _logits_with_feature_mask(
                model=model,
                tokens=corrupt_tokens,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                zero_indices=trial_removed,
            )
        kl = kl_last_token_divergence(logits_ref, logits_trial, seq_pos=seq_pos, reduction="sum")
        kl_val = float(kl.detach().cpu().item())
        if kl_val <= tau:
            removed.add(j)

    kept_sorted = sorted(set(range(nf)) - removed)
    return Circuit(
        hook_name=hook_name,
        feature_indices=kept_sorted,
        n_features_initial=nf,
        removed_indices=sorted(removed),
        attribution_scores=[float(scores[i].item()) for i in range(nf)],
        tau=tau,
        kl_budget=None,
        batch_remove_n=None,
        final_kl_vs_reference=None,
    )
