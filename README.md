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

**Imports:** use an editable install from the repo root (`pip install -e "."`) and run with working directory / `PYTHONPATH` including the repo so `import discovery` resolves. The built wheel currently ships **`causal_patcher` only**; `discovery` is not yet listed as an installable package in the wheel.

**Ranking shape:** attribution returns a **dense** score vector of length `n_features` (full SAE width at one sequence position). Large dictionaries (e.g. 128k latents) mean full-length tensors per pass.

## Scripts

- `scripts/benchmark_discovery_cost.py` — rough timing for attribution/IG paths on a tiny `HookedTransformer` + linear SAE stub; optional integrated-gradients and completeness flags.
- `scripts/test_neuronpedia_label.py` — smoke test for label fetch/cache (see `discovery.labels` and your environment/API setup).

## Tests

```bash
pytest
```

`tests/` currently focus on `causal_patcher` (including viz). Discovery is exercised mainly via scripts and your own notebooks.

## Examples

- **Notebook:** `notebooks/demo_factual_recall.ipynb` — factual recall (Eiffel/Paris vs Colosseum/Rome) on `gpt2-small` with `resid_pre` and `attn_head_z` sweeps.
- **Notebook:** `notebooks/demo_batched_factual_recall.ipynb` — batched / left-padded factual recall using `BatchExperimentRunner`.
- **Heatmap reading:** see [`docs/heatmap.md`](docs/heatmap.md).
