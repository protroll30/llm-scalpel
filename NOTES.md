# Private notes (gitignored)

Personal scratch — flags, follow-ups, and reminders that should not ship in the public repo.

## Open follow-ups

### `BatchExperimentRunner.run_baselines()` is not auto-called

`ExperimentRunner.__init__` runs baselines by default (`run_baselines=True`), but
`BatchExperimentRunner.__init__` does not — callers must invoke `batched.run_baselines()`
explicitly before the first patch call. The README's batched example calls it explicitly
so the snippet works copy-pasted, but the asymmetry is a real footgun.

Options:
- Add `run_baselines: bool = True` to `BatchExperimentRunner.__init__` and mirror the
  Phase-1 behavior. Keeps the two runners interchangeable for users.
- Or document the asymmetry in `BatchExperimentRunner`'s class docstring and leave the
  explicit call as the contract (clearer separation of construction vs. compute).

### `docs/heatmap.md` may not exist yet

The README links to `docs/heatmap.md` in two places (heatmap interpretation table footer
and the Examples section). Verify the file exists before pushing — if it doesn't, either
create a stub or drop the links.

```powershell
Test-Path docs\heatmap.md
```

If missing, simplest fix is removing the two link references from `README.md`. A stub file
with the heatmap reading rules would be more useful long-term.

## Conventions

- New entries go under `## Open follow-ups` as `### Short title` blocks.
- Move resolved items to a `## Resolved` section with a date stamp, or just delete.
- Don't put secrets here; this is gitignored locally but not encrypted.
