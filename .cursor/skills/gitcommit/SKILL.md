---
name: gitcommit
description: Suggest a copy-paste git commit message by analyzing recent git history and current uncommitted/staged changes. Use when the user asks for a commit message, “what should my commit message be”, or invokes /gitcommit.
disable-model-invocation: true
---

# /gitcommit

## Goal

When the user invokes `/gitcommit`, generate a commit message they can copy/paste.

Do **not** create a commit unless the user explicitly asks you to.

## Steps

1. From the repo root, gather context with git:
   - `git status -sb`
   - `git diff` (unstaged)
   - `git diff --staged` (staged)
   - `git log -10 --oneline`

2. Infer the repository’s commit style:
   - Look for prefixes like `feat:`, `fix:`, `docs:`, `refactor:`, etc.
   - Look for casing conventions and whether messages are one-line or multi-paragraph.
   - If unclear, default to a concise one-line subject plus an optional short body.

3. Read the diff(s) to understand the **why**:
   - Identify the core intent (bugfix vs feature vs refactor vs tooling).
   - Identify the primary scope (module/path) and key user-facing behavior changes.
   - Avoid over-listing file-by-file changes.

4. Output:
   - Print **one** recommended commit message in a single code block, ready to paste into `git commit -m ...` (subject line; add a blank line and 1–3 bullet points in body if helpful).
   - If the diff suggests multiple unrelated changes, propose **two** alternative messages and recommend splitting (but still provide the best single message).
   - If there are no changes, say so and do not fabricate a message.

## Constraints

- Never add or modify `git config`.
- Never include secrets or credentials in the suggested message.
- Do not mention internal tool names; just show the commit message.

