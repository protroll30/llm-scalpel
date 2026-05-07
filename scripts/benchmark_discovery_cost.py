"""Rough latency probe for discovery attribution paths (no pytest).

Optionally measures :func:`discovery.attribution.feature_integrated_gradients_pass` (see ``--ig-steps``).

Use this **before** enabling gradient-drift gating in :func:`discovery.pruner.prune_sae_circuit_budget`:
each KL-successful trial can call :func:`discovery.attribution.feature_attribution_pass` (corrupt
backward **plus** clean ``run_with_cache``), which is much heavier than ranking with
:func:`discovery.attribution.feature_act_grad_scores` alone.

Examples::

    python scripts/benchmark_discovery_cost.py
    python scripts/benchmark_discovery_cost.py --repeats 20 --device cuda
    python scripts/benchmark_discovery_cost.py --d-model 768 --n-features 16384 --n-layers 12 --n-ctx 128 --device cuda

The script builds a tiny ``HookedTransformer`` + linear encode/decode stand-in for an SAE so you
can scale ``d_model`` / ``n_features`` toward your real shapes without loading full GPT-2 weights.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig

from discovery.attribution import (
    capture_latent_gradient_snapshot,
    feature_act_grad_scores,
    feature_attribution_pass,
    feature_integrated_gradients_pass,
)


class LinearSAE(nn.Module):
    """Minimal differentiable encoder/decoder on hook activations ``[..., d_model]``."""

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        self.enc = nn.Linear(d_model, n_features, bias=False)
        self.dec = nn.Linear(n_features, d_model, bias=False)

    def encode(self, act: torch.Tensor) -> torch.Tensor:
        return self.enc(act)

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.dec(feats)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable[[], None], repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
        _sync_cuda()
    samples: list[float] = []
    for _ in range(repeats):
        _sync_cuda()
        t0 = time.perf_counter()
        fn()
        _sync_cuda()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark discovery attribution micro-pass cost.")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--d-head", type=int, default=16)
    p.add_argument("--d-mlp", type=int, default=64)
    p.add_argument("--n-ctx", type=int, default=256)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--hook-layer", type=int, default=0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument(
        "--ig-steps",
        type=int,
        default=20,
        help="If >0, also benchmark feature_integrated_gradients_pass with this many Riemann steps.",
    )
    p.add_argument(
        "--ig-schedule",
        type=str,
        default="midpoint",
        choices=("midpoint", "linspace", "trapezoidal"),
        help="Alpha spacing for integrated gradients (see discovery.attribution).",
    )
    p.add_argument(
        "--ig-empty-cache",
        action="store_true",
        help="Pass empty_cache_between_steps=True for IG (CUDA peak VRAM vs speed tradeoff).",
    )
    p.add_argument(
        "--ig-check-completeness",
        action="store_true",
        help="After timing, run one IG pass with check_completeness (needs --ig-steps > 0).",
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    device = torch.device(args.device)

    cfg = HookedTransformerConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_head,
        d_mlp=args.d_mlp,
        n_ctx=args.n_ctx,
        d_vocab=50257,
        act_fn="gelu",
        normalization_type="LN",
        default_prepend_bos=False,
        tokenizer_name="gpt2",
    )
    model = HookedTransformer(cfg).to(device)
    model.eval()

    sae = LinearSAE(args.d_model, args.n_features).to(device)

    def encode_fn(act: torch.Tensor) -> torch.Tensor:
        return sae.encode(act)

    def decode_fn(feats: torch.Tensor) -> torch.Tensor:
        return sae.decode(feats)

    hook_name = f"blocks.{args.hook_layer}.hook_resid_pre"
    clean_prompt = "The capital of France is"
    corrupt_prompt = "The capital of Germany is"
    tid = 25365  # arbitrary vocab id for a scalar loss

    def logits_to_scalar_loss(logits: torch.Tensor) -> torch.Tensor:
        return -logits[0, -1, tid]

    forced: set[int] = set()

    def run_scores() -> None:
        feature_act_grad_scores(
            model=model,
            prompt=corrupt_prompt,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
            forced_zero_indices=forced,
            device=device,
        )

    def run_snapshot() -> None:
        capture_latent_gradient_snapshot(
            model=model,
            prompt_corrupt=corrupt_prompt,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
            forced_zero_indices=forced,
            device=device,
            metric="bench",
        )

    def run_full_pass() -> None:
        feature_attribution_pass(
            model=model,
            prompt_clean=clean_prompt,
            prompt_corrupt=corrupt_prompt,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
            metric="logit_diff_bench",
            forced_zero_indices=forced,
            device=device,
        )

    rows: list[tuple[str, str, Callable[[], None]]] = [
        ("scores", "feature_act_grad_scores (corrupt fwd+bwd)", run_scores),
        ("snapshot", "capture_latent_gradient_snapshot (same core)", run_snapshot),
        ("full_pass", "feature_attribution_pass (+ clean run_with_cache)", run_full_pass),
    ]

    if args.ig_steps > 0:

        def run_ig() -> None:
            feature_integrated_gradients_pass(
                model=model,
                prompt_clean=clean_prompt,
                prompt_corrupt=corrupt_prompt,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                metric="ig_bench",
                forced_zero_indices=forced,
                device=device,
                n_steps=args.ig_steps,
                ig_alpha_schedule=args.ig_schedule,
                empty_cache_between_steps=args.ig_empty_cache,
            )

        rows.append(
            (
                "ig_pass",
                f"feature_integrated_gradients_pass (n_steps={args.ig_steps}, {args.ig_schedule})",
                run_ig,
            )
        )

    print(f"device={device}  cfg: L={args.n_layers} d_model={args.d_model} ctx={args.n_ctx}  SAE width={args.n_features}")
    print(f"hook={hook_name}  repeats={args.repeats} warmup={args.warmup}\n")

    means: dict[str, float] = {}
    for key, label, fn in rows:
        ms = _time_call(fn, args.repeats, args.warmup)
        mean = statistics.mean(ms)
        means[key] = mean
        stdev = statistics.stdev(ms) if len(ms) > 1 else 0.0
        print(f"{mean:8.2f} +- {stdev:6.2f} ms   ({label})")

    mean_scores = means["scores"]
    mean_pass = means["full_pass"]

    print("\n--- Rule-of-thumb (drift gate on prune_sae_circuit_budget) ---")
    print(
        "Each KL-successful trial that reaches drift checks pays roughly "
        f"`feature_attribution_pass` ~ {mean_pass:.1f} ms here vs ranking-only "
        f"`feature_act_grad_scores` ~ {mean_scores:.1f} ms "
        f"(factor x{mean_pass / max(mean_scores, 1e-9):.2f})."
    )
    if "ig_pass" in means:
        mean_ig = means["ig_pass"]
        print(
            f"\nIntegrated gradients (~{args.ig_steps} steps): ~{mean_ig:.1f} ms per pass "
            f"(vs single-point full_pass x{mean_ig / max(mean_pass, 1e-9):.2f}, "
            f"vs scores x{mean_ig / max(mean_scores, 1e-9):.2f}). "
            "Use `--ig-steps 0` to skip IG timing."
        )

    if args.ig_check_completeness:
        if args.ig_steps <= 0:
            raise SystemExit("--ig-check-completeness requires --ig-steps > 0.")
        _, diag = feature_integrated_gradients_pass(
            model=model,
            prompt_clean=clean_prompt,
            prompt_corrupt=corrupt_prompt,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
            metric="ig_bench",
            forced_zero_indices=forced,
            device=device,
            n_steps=args.ig_steps,
            ig_alpha_schedule=args.ig_schedule,
            empty_cache_between_steps=args.ig_empty_cache,
            check_completeness=True,
        )
        print("\n--- IG completeness (plain ℒ_clean − ℒ_corrupt vs Σ IG) ---")
        print(
            f"  Δℒ_natural = {diag.delta_metric:.6g}   Σ latent IG = {diag.sum_latent_ig:.6g}   "
            f"residual IG = {diag.residual_score:.6g}   total_attr = {diag.total_attributed:.6g}   "
            f"|gap| = {diag.gap_abs:.6g}"
        )
        print(f"  (ℒ_clean={diag.metric_clean:.6g}, ℒ_corrupt={diag.metric_corrupt:.6g})")

    print(
        "Recursive binary splitting retries smaller chunks on KL or drift failure; worst-case "
        "attribution calls per outer wave scale with tree probes (often multiple x batch_remove_n)."
    )
    print("Scale `--d-model`, `--n-features`, `--n-ctx`, `--n-layers` toward your real setup.")


if __name__ == "__main__":
    main()
