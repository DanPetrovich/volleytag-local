# VolleyTag agent instructions

## Communication

- Address the user informally in Russian: use «ты», never «вы».
- Keep the tone warm, friendly, direct, and non-formal.

## Autonomy

- Do not ask the user for permission for routine, safe, reversible actions that are clearly within the current task. The user explicitly authorizes all such actions.
- Make reasonable technical decisions independently when the intended outcome is clear.
- When a logical stage produces results and those results make the next steps clear, continue with those next steps without stopping for approval or asking whether to proceed.
- Carry work through implementation, validation, comparison, and reporting rather than stopping after an intermediate artifact or a single experiment.
- Prefer doing the next useful experiment or verification over merely proposing it.

## When to stop and ask

Stop only for a real blocker, including:

- an action that requires explicit user confirmation because it is destructive, risky, irreversible, externally visible, transmits sensitive data, or changes access/permissions;
- required manual work that only the user can perform, such as annotation, account verification, CAPTCHA, or choosing between materially different product directions;
- missing information that would substantially change the implementation and cannot be inferred safely;
- unclear or genuinely non-obvious next steps where proceeding would risk wasting substantial time or data.

When blocked, explain the exact blocker and ask one concrete question. Do not ask for confirmation earlier than necessary.

## Project-specific safeguards

- Keep the third set and any other designated evaluation match frozen: do not use them for training, threshold selection, model selection, or tuning.
- Preserve user annotations and existing track IDs unless a migration is explicitly required and backed up.
- Validate datasets before remote training: image/label counts, box bounds, split integrity, negative examples, and domain balance.
- Keep uploaded training datasets private unless the user explicitly requests publication.
- Before any expensive training or full-video evaluation, run a representative sanity check first (normally a few hundred samples or a short video segment).
- Define the sanity-check baseline and success criterion before running it. Continue to the full run only if the candidate method materially improves the intended metric and shows no obvious regression.
- Reuse cached crops, embeddings, detections, and other intermediate artifacts whenever possible. Do not recompute expensive features merely to test a new lightweight head or routing rule.
- Estimate the full run's sample count and duration from the sanity-check throughput. If it is unexpectedly expensive, reduce redundant candidates, move the job to Kaggle GPU, or redesign the experiment before committing hours of compute.

## GitHub maintenance

- Treat `main` as the current, working, documented state of the project. When a completed change is validated, commit it, push it, and fast-forward or merge it into `main` without waiting for a separate reminder.
- Before every push, fetch and inspect the complete diff against `origin/main`, including filenames, tracked file sizes, URLs, secrets, local absolute paths, and generated artifacts.
- Never push match videos, extracted frames or photographs, model weights, embeddings, caches, logs, API keys, credentials, private match links, private source registries, or other personal/private data.
- Keep Kaggle datasets and kernels private. Version only reproducible source code, metadata without credentials, compact aggregate reports, and documentation that does not expose private source locations.
- Use a GitHub noreply address for new commits. Do not rewrite already-public history merely to change old author metadata unless the user explicitly requests a coordinated history rewrite.
- Do not force-push `main`. Prefer a fast-forward update; if the remote moved, fetch, rebase or merge safely, rerun validation, and then push.
- Keep `.gitignore` and repository documentation aligned with the actual generated artifacts and current workflows.

## Required validation

- Run the lightweight automated suite before every push: Python compilation, Python unit tests, the evaluation-schema Node checks, and `git diff --check`.
- Run focused tests for the code being changed and a representative browser/server smoke test when UI or HTTP behavior changes.
- Keep GitHub Actions green. If a check fails, diagnose and fix it before updating `main`; never bypass tests merely to make the branch look current.
- Update `README.md` and compact experiment-status documentation whenever commands, supported workflows, promoted models, limitations, or evaluation conclusions change.
- Report failed acceptance criteria honestly. Do not describe a model or temporal route as promoted when its predeclared criterion failed.
