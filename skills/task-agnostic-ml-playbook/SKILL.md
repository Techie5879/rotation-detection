# Skill: Task-Agnostic ML Iteration Playbook

This skill captures repeatable practices for building and improving ML systems with an agent, independent of model family, dataset type, or domain.

## Bootstrap First: Environment + Tooling Intake

Before touching model code, collect an environment snapshot. If project-level agent docs (for example `AGENTS.md`) already contain this information, reuse it and only refresh stale/missing fields. If not, gather and write it down in your run notes.

Capture at minimum:

- Host/platform: OS version, kernel, architecture, CPU, RAM.
- Accelerator context: GPU/Metal/CUDA availability and driver/runtime status.
- Python/runtime baseline: required Python version, active interpreter, active env path.
- Package manager contract: which tool is authoritative (`uv`, `poetry`, `pip`, etc.).
- Build/toolchain caveats: compiler/Xcode/toolchain status when relevant.

### `uv` + ML Package Caveats (Default Policy)

Use these as default guardrails unless project docs explicitly override them:

- Use `uv` as the runtime entrypoint (`uv run ...`) for all Python commands.
- Use the project-local environment only (typically `.venv` created by `uv`).
- For GPU-heavy ML packages (`torch`, `torchvision`, `torchaudio`, `mlx`, etc.), prefer:
  - `uv pip install ...`
  - avoid `uv add` unless the project explicitly wants lockfile-managed ML deps.
- After ML package installs/updates, refresh reproducibility output if the repo expects it (for example `uv pip freeze > requirements.txt`).
- Do not assume accelerator backend is usable just because hardware exists; verify framework + backend availability at runtime.

If any of the above is unknown at session start, treat "environment clarification" as task zero.

## Use This Skill When

- You are iterating on an ML pipeline and want faster, safer progress.
- You need high-confidence conclusions (not one-off wins).
- You want clear separation between experimentation and final reporting.
- You need reproducible artifacts and explainable decisions.

## Core Principles

- Optimize for learning speed first, full-scale training second.
- Instrument before guessing.
- Never tune against the final test split.
- Prefer small controlled changes over multi-variable jumps.
- Keep evaluation protocols locked once declared.
- Treat docs and reports as product outputs, not afterthoughts.

## Standard Workflow

1. Establish invariants and success criteria
- Define what is forbidden (for example leakage sources, metadata shortcuts, hidden labels).
- Define what metric(s) matter and on which split.
- Define what "done" means (for example quality threshold + reproducibility checks).

2. Build data + split guardrails early
- Add deterministic seeds and manifest outputs.
- Track class distribution and entity distribution per split.
- Add hard constraints for split quality (for example max entity share per split).
- Fail fast when split quality violates declared limits.

3. Create a tight debug loop
- Use small datasets / capped samples for fast iteration.
- Keep run directories isolated and named by intent.
- Log progress frequently enough to catch stalls and drift.
- Add memory/time observability to identify throughput bottlenecks.

4. Make training observable
- Log train/val loss + accuracy per epoch.
- Log per-class metrics, dominant confusions, and generalization gap.
- Log confidence/entropy proxies for uncertainty trends.
- Save structured history so comparisons are scriptable.

5. Evaluate correctly
- Distinguish clearly between:
  - train-time learning
  - inference-only evaluation
  - tuning evaluation
- Keep test evaluation single-pass and untouched by threshold search.
- For confidence-based systems, report calibration (NLL/Brier/ECE/MCE) and confidence slices.

6. Run error analysis that drives action
- Break down by entity/document/source, not just aggregate metrics.
- Inspect representative successes and failures visually.
- Quantify which confusion pairs dominate and where.
- Convert findings into one explicit next change.

7. Add postprocessing only with strict protocol
- Prototype postprocessing on non-final splits first.
- Select thresholds on validation split only.
- Freeze thresholds.
- Run test once with locked settings.
- Report both base and postprocessed metrics + delta + fix/regress counts.

8. Publish operator-friendly outputs
- Provide copy/paste commands for common paths.
- Ship browser-friendly visualization for qualitative review.
- Keep examples representative (avoid no-op or misleading samples).
- Document exact artifact paths used for reported numbers.

## Guardrails That Prevent Common Failures

- Leakage guardrail
  - No hyperparameter or threshold search on test split.
  - If violated, mark affected test numbers as optimistic and rerun clean protocol.

- Split domination guardrail
  - Track largest-entity share in each split.
  - Block dataset generation if share exceeds configured threshold.

- Interpretation guardrail
  - Never present train-split inference as "model generalization".
  - Label metrics by split in every report and UI.

- Visualization guardrail
  - Exclude trivial/no-op examples by default.
  - Include confidence and correctness indicators in visual outputs.

## Agent Execution Pattern

When acting as an agent, use this loop:

- Read logs and reports first.
- Form one hypothesis.
- Add targeted instrumentation if evidence is insufficient.
- Validate on a small run.
- Compare against prior baseline.
- Promote only proven changes to full run.
- Update docs/commands immediately after behavior changes.

## Decision Rules

- If val is noisy due to split composition, fix split quality before tuning optimizer knobs.
- If dominant errors cluster in one confusion pair, prioritize targeted remedies (features, augmentations, postprocess) over global retraining complexity.
- If confidence is useful but imperfect, add calibration reporting before adding hard confidence policies.
- If throughput is slow, profile pipeline stages (data read, render, batch wait, step compute) before changing model architecture.

## Reusable Deliverables Checklist

- Reproducible dataset manifest with split proof and sampling proof.
- Training report with epoch history and diagnostics.
- Saved-model evaluator with calibration and confidence logs.
- Visual dashboard for before/after inspection.
- README section with:
  - locked commands
  - locked thresholds
  - final metrics
  - explicit test protocol

## Minimal "Definition Of Done"

- Metrics meet target on untouched test split.
- Protocol is leak-free and documented.
- Results are reproducible from commands in docs.
- Error profile is understood and summarized.
- A new operator can run end-to-end without tribal context.
