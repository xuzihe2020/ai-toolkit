# Repository Instructions

## Repository ownership

This fork owns the ai-toolkit training runtime and web UI. Keep trainer behavior
and UI changes here.

Cross-repository training orchestration, RunPod lifecycle automation, task
configurations, and operational procedures belong in the sibling
`../aigc-infra` repository. Read `../aigc-infra/AGENTS.md` before work that
crosses repository boundaries or touches RunPod.

Reusable Python primitives shared by multiple repositories belong in
`../aigc-shared`. This runtime consumes them only through the immutable package
pin managed by `aigc-infra`; do not copy shared modules into this repository.

## Validation

Run focused tests for changed trainer/UI components and verify that `run.py`
can import its configuration path before handing off changes. Dependency
changes must not replace the container image's known-good torch build unless
that upgrade is explicitly requested.
