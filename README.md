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

## Examples

- **Notebook:** `notebooks/demo_factual_recall.ipynb` — factual recall (Eiffel/Paris vs Colosseum/Rome) on `gpt2-small` with `resid_pre` and `attn_head_z` sweeps.
- **Heatmap reading:** see [`docs/heatmap.md`](docs/heatmap.md).
