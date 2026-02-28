# Publishing Checklist

## Before First Commit

- confirm `.env` is absent
- confirm `input/` is absent
- confirm `output/` is absent
- confirm `.cache/` and `logs/` are absent
- confirm only template credentials remain in `.env.example`
- choose and add a `LICENSE`

## Before Pushing To GitHub

- run a secret scan on the repository root
- inspect `git diff --cached`
- verify large binary files are not included
- verify no run directories or generated reports are tracked
- verify README and docs reflect what is actually public

## Recommended Commands

```bash
git init
rg -n --hidden -S '(API_KEY|SECRET|TOKEN|PASSWORD|BEGIN RSA|BEGIN OPENSSH|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]|ghp_[A-Za-z0-9])' .
git status
git add .
git diff --cached --stat
git diff --cached
```

## Release Notes To Mention

- archival-first extraction and assembly pipeline
- explicit blank-vs-failure separation
- checkpointed reruns
- deterministic verification artifacts
- validator-gated LLM operations
