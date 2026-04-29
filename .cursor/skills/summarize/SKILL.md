---
name: summarize
description: Produces a structured /summarize project status report (component map, state, tensor shapes, blockers, next goal) for the llm-scalpel and causal-patcher codebase. Use when the user invokes /summarize, asks for a session or project summary in the summarize format, or wants a handoff block to prevent context loss.
---

# /summarize (project status handoff)

When the user requests **`/summarize`** or a summary in this format, **read the relevant source** (`causal_patcher/`, `tests/`, `docs/`) to fill **Active Blockers** and confirm shapes. Then output a single report using **exactly** the five sections below (same headings and order).

## Output template (define /summarize as follows)

**Component Map:** List the current core classes (`ExperimentRunner`, `PatchTarget`) and their primary responsibilities.

**Current State:** Summarize the last successful feature implemented (e.g., "Phase 1: Explicit index mapping and logit-diff viz complete").

**Input/Output Shapes:** Explicitly list the expected tensor shapes (e.g., `[batch, pos, d_model]`) because this is where the LLM usually hallucinates as context fades.

**Active Blockers:** List any specific bugs or "TODOs" currently in the code.

**Next Goal:** State the immediate next task (e.g., "Implementing BatchExperimentRunner with left-padding support").

## Reference: causal-patcher tensor shapes (HookedTransformer, batch size 1)

Use these as the default unless the code has changed. Subscripts: `B` = batch, `P` = sequence length, `L` = layer, `H` = `n_heads`, `D` = `d_model`, `d_head` = `d_head`, `V` = `d_vocab_out`.

| Object | Shape |
|--------|--------|
| `clean_tokens`, `corrupt_tokens` | `[B, P]` (int) |
| `clean_logits`, `corrupt_logits`, patched logits | `[B, P, V]` |
| Cached `resid_pre` / `resid_mid` / `resid_post` / `mlp_out` at a layer | `[B, P, D]` |
| Cached `attn` `hook_z` at a layer | `[B, P, H, d_head]` |
| `logit_diff` (scalar on selected position) | scalar from `logits[b, pos, t_clean] - logits[b, pos, t_corrupt]` |

**Constraint:** `run_baselines` requires **identical** `clean_tokens` and `corrupt_tokens` **shape**; cross-length prompts are not supported without a different design.

## Current defaults (as of this skill; verify in code)

- **Component Map:** `PatchTarget` — one patch site (kind, layer, optional `head` for `attn_head_z`, optional `pos`). `ExperimentRunner` — tokenize clean/corrupt, `run_with_cache`, `logit_diff`, `patch_clean_into_corrupt`.
- **Active Blockers:** If a grep for `TODO`/`FIXME` in `causal_patcher/` is empty, state that none are recorded in-code.
- **Next Goal:** If unknown, state "TBD" or ask the user for one line.

## When to re-read the repo

- Before claiming **blockers** or **current state** after substantive edits.
- To align **Input/Output Shapes** with `causal_patcher/runner.py` and `targets.py` if the patch hook contract changes.
