# llm-scalpel

Automated causal intervention framework for mechanistic interpretability experiments.

## causal-patcher (Python package)

Install from the repository root:

```bash
pip install -e "."
# Optional: notebook + table deps for examples
pip install -e ".[demo]"
```

### Interpreting patch heatmaps (logit difference)

`causal_patcher.viz` plots **logit(clean answer) − logit(corrupt answer)** on the patched **corrupt** run. With the default `RdBu_r` scale (centered at zero):

| Color        | Meaning |
|--------------|---------|
| **Red / warm**  | **Recovery** — the patch *increased* that score; the run looks *more* like the clean-answer side of the comparison. |
| **Blue / cool** | **Suppression** — the patch *decreased* the score; the readout moved toward the corrupt answer or away from the clean one. |
| **Near white**  | little effect. |

A longer guide (layer × position vs. layer × head, baselines, caveats) is in [`docs/heatmap.md`](docs/heatmap.md).

### Token labels on the x-axis

`plot_layer_position_patching` labels each column with decoded subwords from the **corrupt** prompt by default. For custom grids, pass `x_tick_labels=viz.position_tick_labels(runner, "corrupt")` to `plot_heatmap`, or `which="clean"` to align the axis with the clean sequence.

## discovery (experimental)

Code under `discovery/` builds on a TransformerLens `HookedTransformer` with **hook-mounted SAE-style** `encode_fn` / `decode_fn` callables. You supply the SAE (or a toy linear stand-in); the library wires **residual-aware reconstruction** at the hook, attribution, and optional KL-budget pruning.

| Module | Role |
|--------|------|
| `discovery.sae_scout` | SAE weight loading/cache helpers, `reconstruct_activation`, `feature_capture` |
| `discovery.attribution` | `feature_act_grad_scores`, `feature_attribution_pass`, integrated gradients, residual-channel score + completeness helpers |
| `discovery.pruner` | `prune_sae_circuit` (τ-KL) and `prune_sae_circuit_budget` (KL budget, optional drift gate, optional IG ranking) |
| `discovery.labels` | Neuronpedia feature labels + JSON cache |
| `discovery.circuit_graphviz` | Bipartite SAE latent graphs (DOT) + cross-hook edge weights (`scripts/sae_crosslayer_circuit_dot.py`) |

### Cross-layer SAE graphs: direct path vs “scalpel” (attention-in-the-middle)

The DOT export from `discovery.circuit_graphviz` measures edges from **layer‑A SAE latents** to **layer‑B SAE latents** using a **local linear-style summary** (intervention at one hook × gradient at another). Treat that figure as the **direct residual-stack hypothesis**: latent→latent along **`hook_resid_pre`→`hook_resid_pre`** **without** an explicit attention head as an intermediate node.

When factual recall moves **information across token positions**, coupling often travels through **attention** (queries / keys / values), not only through same-position residual channels. So **near-zero edges** in that bipartite graph are not a bug to hide—they are a useful **“before” picture**: they show where the **direct path story fails**, which motivates head-level tools next.

The complementary **three-node** chain is: **latent at layer 8 → one attention head** (IDs like **L9H8** from `find_mover_heads.py`) **→ latent at layer 9.**

Use **`scripts/find_mover_heads.py`** (marginal `hook_z` patching vs your logit objective) and **`scripts/visualize_attention_heads.py`** ( **`hook_pattern`** heatmaps) to nominate and verify heads (e.g. query at **` is`** attending back to the **country** token). Keeping both artifacts—the sparse/zero cross-layer DOT **and** the ranked heads—is the intended workflow.

**Imports:** use an editable install from the repo root (`pip install -e "."`) and run with working directory / `PYTHONPATH` including the repo so `import discovery` resolves. The built wheel currently ships **`causal_patcher` only**; `discovery` is not yet listed as an installable package in the wheel.

**Ranking shape:** attribution returns a **dense** score vector of length `n_features` (full SAE width at one sequence position). Large dictionaries (e.g. 128k latents) mean full-length tensors per pass.

## Scripts

- `scripts/benchmark_discovery_cost.py` — rough timing for attribution/IG paths on a tiny `HookedTransformer` + linear SAE stub; optional integrated-gradients and completeness flags.
- `scripts/test_neuronpedia_label.py` — smoke test for label fetch/cache (see `discovery.labels` and your environment/API setup).
- `scripts/sae_crosslayer_circuit_dot.py` — Graphviz DOT bipartite graph between two SAE hooks (optional `--src-seq-pos` / `--dst-seq-pos` / `--loss-seq-pos`).
- `scripts/find_mover_heads.py` — rank attention heads by marginal clean→corrupt `hook_z` patching effects.
- `scripts/visualize_attention_heads.py` — plot `hook_pattern` for nominated heads (PNG export; optional query/key highlight).

## Tests

```bash
pytest
```

`tests/` currently focus on `causal_patcher` (including viz). Discovery is exercised mainly via scripts and your own notebooks.

## Examples

- **Notebook:** `notebooks/demo_factual_recall.ipynb` — factual recall (Eiffel/Paris vs Colosseum/Rome) on `gpt2-small` with `resid_pre` and `attn_head_z` sweeps.
- **Notebook:** `notebooks/demo_batched_factual_recall.ipynb` — batched / left-padded factual recall using `BatchExperimentRunner`.
- **Heatmap reading:** see [`docs/heatmap.md`](docs/heatmap.md).
