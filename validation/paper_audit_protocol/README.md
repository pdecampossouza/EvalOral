# Blinded dental audit protocol prepared for the EvalOral paper

The audit is intentionally separated into three tasks so that one type of explanation does not bias another.

## Part A — label-free visual clusters

Reviewer sees five prototypes from each fine (`k=7`) content-deduplicated cluster without source labels. Suggested ratings:

- visual coherence (1–5),
- odontological visual relevance (1–5),
- predominant visible pattern,
- whether the group should be split,
- free-text comment.

## Part B — rule-linked Grad-CAM

Reviewer sees 12 cases with the model claim and anonymized rule ID while the released reference view and correctness are hidden. Suggested ratings:

- localization plausibility,
- salient structure,
- artifact influence,
- explanation usefulness (1–5),
- free-text comment.

## Part C — evolving-rule history

Reviewer sees representative rule images plus birth/support/update/staleness/regime information. Suggested ratings:

- visual coherence (1–5),
- history understandable (1–5),
- history useful (1–5),
- plausibility of reported stability,
- free-text comment.

The returned responses should be analyzed only after unblinding. They must not be used to retrain the model if the goal is an independent evaluation of human understanding.
