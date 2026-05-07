"""Template: run discovery attribution/pruning on GPT-2 small.

This script is meant to be a *real* end-to-end harness for `discovery/`:
- loads `gpt2-small` via TransformerLens
- defines an SAE interface (default: simple linear stub; replace with your real SAE)
- runs:
  - Taylor attribution (`feature_attribution_pass`) including residual_score
  - optional Integrated Gradients (`feature_integrated_gradients_pass`) + completeness diagnostics
  - optional KL-budget pruning (`prune_sae_circuit_budget`)

Example:

  python scripts/run_discovery_real.py --device cuda --hook-layer 0 --n-features 4096
  python scripts/run_discovery_real.py --device cuda --ig-steps 20 --run-prune --kl-budget 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Literal, cast

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from discovery.attribution import (
    feature_attribution_pass,
    feature_integrated_gradients_pass,
)
from discovery.pruner import prune_sae_circuit_budget


class LinearSAE(nn.Module):
    """Minimal differentiable SAE interface on hook activations `[..., d_model]`.

    Replace this with your real SAE module / wrapper.
    """

    def __init__(self, d_model: int, n_features: int) -> None:
        super().__init__()
        self.enc = nn.Linear(d_model, n_features, bias=False)
        self.dec = nn.Linear(n_features, d_model, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.dec(f)


def _scalar_metric_plain_forward(
    *,
    model: HookedTransformer,
    tokens: torch.Tensor,
    logits_to_scalar_loss: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        return float(logits_to_scalar_loss(logits).detach().cpu().item())


def main() -> None:
    p = argparse.ArgumentParser(description="Template discovery run on gpt2-small.")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--hook-layer", type=int, default=0, help="Layer index for blocks.L.hook_resid_pre")
    p.add_argument("--seq-pos", type=int, default=-1, help="Token position for attribution/IG (supports -1).")

    # SAE width (stub)
    p.add_argument("--n-features", type=int, default=4096, help="SAE latent width for the stub encoder/decoder.")

    # Prompts + answers
    p.add_argument("--clean-prompt", type=str, default="The capital of France is")
    p.add_argument("--corrupt-prompt", type=str, default="The capital of Germany is")
    p.add_argument("--clean-answer", type=str, default=" Paris", help="Single token (include leading space).")
    p.add_argument("--corrupt-answer", type=str, default=" Berlin", help="Single token (include leading space).")

    # IG
    p.add_argument("--ig-steps", type=int, default=0, help="If >0, run integrated gradients with n steps.")
    p.add_argument(
        "--ig-schedule",
        type=str,
        default="midpoint",
        choices=("midpoint", "linspace", "trapezoidal"),
    )
    p.add_argument("--ig-check-completeness", action="store_true")

    # Pruning
    p.add_argument("--run-prune", action="store_true", help="Run KL-budget pruning after attribution.")
    p.add_argument("--kl-budget", type=float, default=0.5)
    p.add_argument("--batch-remove-n", type=int, default=32)
    p.add_argument(
        "--ranking-mode",
        type=str,
        default="act_grad",
        choices=("act_grad", "integrated_gradients"),
        help="Ranking signal inside prune_sae_circuit_budget.",
    )
    p.add_argument("--prune-ig-steps", type=int, default=10)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    ig_alpha_schedule = cast(Literal["midpoint", "linspace", "trapezoidal"], args.ig_schedule)

    device = torch.device(args.device)
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    # Hook site
    hook_name = f"blocks.{int(args.hook_layer)}.hook_resid_pre"

    # SAE stub (replace with your real SAE)
    d_model = int(model.cfg.d_model)
    sae = LinearSAE(d_model=d_model, n_features=int(args.n_features)).to(device)

    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        return sae.encode(x)

    def decode_fn(f: torch.Tensor) -> torch.Tensor:
        return sae.decode(f)

    # Scalar metric: prefer clean answer token over corrupt answer token at seq_pos.
    clean_tok = int(model.to_single_token(args.clean_answer))
    corrupt_tok = int(model.to_single_token(args.corrupt_answer))

    def logits_to_scalar_loss(logits: torch.Tensor) -> torch.Tensor:
        pos = int(args.seq_pos)
        if pos < 0:
            pos += int(logits.shape[1])
        # Larger is "more clean-like": logit(clean) - logit(corrupt)
        score = logits[0, pos, clean_tok] - logits[0, pos, corrupt_tok]
        return -score

    # --- Taylor attribution + residual score ---
    print(f"model=gpt2-small device={device} hook={hook_name} seq_pos={args.seq_pos}")
    print(f"clean={args.clean_prompt!r}")
    print(f"corrupt={args.corrupt_prompt!r}")
    print()

    taylor = feature_attribution_pass(
        model=model,
        prompt_clean=args.clean_prompt,
        prompt_corrupt=args.corrupt_prompt,
        hook_name=hook_name,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        logits_to_scalar_loss=logits_to_scalar_loss,
        metric="logit_diff",
        seq_pos=int(args.seq_pos),
        device=device,
    )

    # Plain Δℒ for completeness report
    to_tokens_kwargs: dict[str, Any] = {"prepend_bos": False}
    clean_tokens = model.to_tokens(args.clean_prompt, **to_tokens_kwargs).to(device)
    corrupt_tokens = model.to_tokens(args.corrupt_prompt, **to_tokens_kwargs).to(device)
    delta_loss_natural = _scalar_metric_plain_forward(model=model, tokens=clean_tokens, logits_to_scalar_loss=logits_to_scalar_loss) - _scalar_metric_plain_forward(
        model=model, tokens=corrupt_tokens, logits_to_scalar_loss=logits_to_scalar_loss
    )

    report = taylor.get_completeness_report(actual_delta_loss=delta_loss_natural)
    print("--- Taylor attribution (latents + residual) ---")
    print(f"  residual_score = {taylor.residual_score:.6g}")
    print(
        "  completeness: "
        f"actual_delta={report['actual_delta']:.6g} total_attr={report['total_attributed']:.6g} "
        f"|err|={report['approximation_error']:.6g} "
        f"(latent_frac={report['latent_contribution']:.1%}, residual_frac={report['residual_contribution']:.1%})"
    )

    topk = torch.topk(taylor.scores.abs(), k=min(10, int(taylor.scores.numel())))
    print("  top|score| latents:", [int(i) for i in topk.indices.detach().cpu().tolist()])

    # --- Integrated Gradients (optional) ---
    if int(args.ig_steps) > 0:
        print("\n--- Integrated gradients ---")
        if args.ig_check_completeness:
            ig_pass, diag = feature_integrated_gradients_pass(
                model=model,
                prompt_clean=args.clean_prompt,
                prompt_corrupt=args.corrupt_prompt,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                metric="ig_logit_diff",
                seq_pos=int(args.seq_pos),
                n_steps=int(args.ig_steps),
                ig_alpha_schedule=ig_alpha_schedule,
                device=device,
                check_completeness=True,
            )
            print(
                f"  Δℒ_natural={diag.delta_metric:.6g}  Σ_latent_ig={diag.sum_latent_ig:.6g}  "
                f"residual_ig={diag.residual_score:.6g}  total={diag.total_attributed:.6g}  |gap|={diag.gap_abs:.6g}"
            )
        else:
            ig_pass = feature_integrated_gradients_pass(
                model=model,
                prompt_clean=args.clean_prompt,
                prompt_corrupt=args.corrupt_prompt,
                hook_name=hook_name,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                logits_to_scalar_loss=logits_to_scalar_loss,
                metric="ig_logit_diff",
                seq_pos=int(args.seq_pos),
                n_steps=int(args.ig_steps),
                ig_alpha_schedule=ig_alpha_schedule,
                device=device,
            )
        topk_ig = torch.topk(ig_pass.scores.abs(), k=min(10, int(ig_pass.scores.numel())))
        print("  top|IG| latents:", [int(i) for i in topk_ig.indices.detach().cpu().tolist()])

    # --- KL-budget pruning (optional) ---
    if bool(args.run_prune):
        print("\n--- Prune (KL budget) ---")
        ranking_mode = cast(Literal["act_grad", "integrated_gradients"], args.ranking_mode)
        circuit = prune_sae_circuit_budget(
            model=model,
            corrupt_prompt=args.corrupt_prompt,
            hook_name=hook_name,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            logits_to_scalar_loss=logits_to_scalar_loss,
            kl_budget=float(args.kl_budget),
            batch_remove_n=int(args.batch_remove_n),
            seq_pos=int(args.seq_pos),
            attribution_seq_pos=int(args.seq_pos),
            prompt_clean=args.clean_prompt,  # enables drift-gate baseline if you also enable drift args later
            ranking_mode=ranking_mode,
            ig_n_steps=int(args.prune_ig_steps),
            ig_alpha_schedule=ig_alpha_schedule,
        )
        print(f"  kept={len(circuit.feature_indices)} removed={len(circuit.removed_indices)} final_KL={circuit.final_kl_vs_reference}")


if __name__ == "__main__":
    main()

