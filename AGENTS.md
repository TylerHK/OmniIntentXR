# OmniIntentXR contributor instructions

Before editing, read AGENTS.md, README.md and docs/usage.md. Discover any
ancestor or deeper AGENTS.md/CLAUDE.md instructions for files being changed.
Documentation/CONTRIBUTING.md and Documentation/Setup.md are currently absent;
do not treat their setup or contribution steps as verified.

This repository is the XR/sensor-ingestion prototype. TylerHK/IAmTheDrop is the
canonical runtime; keep runtime and first-minute OmniIntent Council UX work in
that repository unless explicitly scoped here. Do not copy runtime code across
repositories or claim headset/Unreal behavior from Python tensor tests.

Run pytest with the documented dependencies and the CLI ingestion smoke test.
Report the browser E2E skip honestly when OMNI_UI_URL is absent; sample ingestion
without model weights is not validated model inference or a headset build.

Unreal .uasset/.umap files use Git LFS. Confirm quota/billing authorization
before adding binary content. Do not rewrite history or upload existing assets
to LFS without approval. Preserve draft PR status; no merge/main push/deployment
without separate approval.
