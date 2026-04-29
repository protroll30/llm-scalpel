# Reading logit-difference heatmaps

Patching heatmaps from `causal_patcher` use the **logit-difference** metric from `ExperimentRunner.logit_diff`:

> **logit_diff** = logit(clean answer token) − logit(corrupt answer token)  

on the **corrupt** forward run (after patching), usually at the **last prompt position** (where the model predicts the next token).

## Diverging colormap (default: `RdBu_r`)

The default `plot_heatmap` and layer–position / layer–head plot helpers use a **symmetric** color scale (same magnitude for positive and negative) around **zero**, so the sign of the value is what matters.

| Color (typical) | Meaning |
|-----------------|--------|
| **Red / warm** | **Recovery** — patching **raised** the logit difference. The model’s log-odds for the *clean* answer (e.g. “Paris”) increased relative to the *corrupt* answer (e.g. “Rome”) on the **patched** corrupt run. In counterfactual setups, that usually means the intervention **helped** the model look more like the clean behavior. |
| **Blue / cool** | **Suppression** — patching **lowered** the logit difference. The model became **less** likely to output the clean answer and **more** in line with the corrupt answer (or simply more confused on that axis). |
| **Near white** | The patch had **little effect** on this readout. |

> **Caveat:** The words “Recovery” and “Suppression” describe the **effect on the chosen logit-difference readout** (one scalar per cell). They do not, by themselves, name the *mechanism* (attention vs MLP, etc.); the **hook point** and **position/head** define where you intervened.

## Layer × position (e.g. `resid_pre`)

- **Y-axis:** Model **layer** (or depth). Higher layers = later processing.
- **X-axis:** **Token position** in the sequence. By default, `causal_patcher` can label this axis with **decoded subword strings** from the model tokenizer (typically the **corrupt** prompt), so you can see which BPE token each column refers to.
- A **red** cell at (layer *L*, position *p*) means: patching from the clean run at the mapped **clean** index into the corrupt run at *p* (or your explicit `(clean_idx, corrupt_idx)`) at **hook** `resid_pre` in layer *L* **increased** the clean–corrupt logit gap.

## Layer × head (`attn_head_z`)

- **X-axis:** **Head index** (not a token).
- The patch is at a chosen **position** (often the **last** prompt token where the next city is predicted).
- A **red** head means that head’s **z** (pre-output) at that site, when replaced from the clean run, **pushes** the readout toward the **clean** answer in logit-difference space.

## Tips

- Compare against **baseline** `logit_diff` on the corrupt run without patches; heatmap cells are **not** “minus baseline” unless you subtract that yourself.
- If prompts differ in length, use `PatchTarget(pos=(clean_i, corrupt_i))` and align the **x-axis** to the run you are patching (usually **corrupt**), which matches the default token labels from `position_tick_labels(runner, "corrupt")`.

For API details, see `causal_patcher.viz` and the example notebook `notebooks/demo_factual_recall.ipynb`.
