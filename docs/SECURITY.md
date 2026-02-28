# Security Guidance

## Publication Scope

This open-source export is intended to be safe to publish. It excludes local secrets and runtime data by design.

## What Was Explicitly Excluded

- `.env`
- `input/`
- `output/`
- `.cache/`
- `logs/`
- `.venv/`
- `tmp/`
- backups and local experiments

## Credential Handling

Use `.env.example` only as a template.

Real credentials must live in `.env`, which must never be committed. Supported variables include:
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `MOONSHOT_API_KEY`

## Important Recommendation

Real credentials were found in the source workspace `.env` during packaging. Before publishing anything derived from that workspace, rotate those credentials.

## Safe Publication Rules

1. Publish only the exported folder, not the original working tree.
2. Run a secret scan on the exported folder before pushing.
3. Add `.gitignore` before the first commit.
4. Verify there are no sample outputs, logs, or input books in the commit.
5. Treat cached run directories as potentially sensitive and non-public.

## Suggested Pre-Push Check

```bash
rg -n --hidden -S '(API_KEY|SECRET|TOKEN|PASSWORD|BEGIN RSA|BEGIN OPENSSH|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]|ghp_[A-Za-z0-9])' .
```

Review matches manually before publishing.
