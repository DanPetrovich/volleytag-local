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
