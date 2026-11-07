# OmniIntentXR
Unified XR project enabling semi-omnipotent intent via gesture, voice, and gaze (Quest 3).

## 60-second Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=./src
python -m omniintent demo --ingest samples/quest3_tiny.csv --seq-len 60
```

See [docs/usage.md](docs/usage.md) for a walkthrough and expected output.

## Vision
Unified XR experience granting users a sense of "semi‑omnipotent intent," intuitively expressed via seamless **gesture, voice, and gaze** inputs. Future roadmap includes EEG‑driven thought control.

## Current Stage (May 27 2025)
- **Engine**: Unreal Engine 5.4 prototype
- **Gesture recognition & playback**: Stable
- **Voice commands**: Integrated
- **Fireball spell**: Complete
- **Telekinesis spell**: Nearly finalized
- **Holographic wrist UI**: Stable
- **Intent Router & Gaze Simulation**: In progress

## Immediate Sprint Goal (v0.3.1 – ETA June 2 2025)
- Gesture‑controlled **Telekinesis** (polished)
- Unified Intent Recognition Router
- Gaze simulation (head‑ray + dwell)
- 2‑minute narrated demo showcase

## Repository Structure
```
/Blueprints      # Unreal Blueprints
/Assets          # 3D models, textures, audio, VFX
/DemoScenes      # Playable demo maps & levels
/Documentation   # Design docs, how‑tos, API refs
/SpellLibrary    # Gesture JSON + thumbnails
```

## Getting Started (Developer)
1. Clone repo: `git clone https://github.com/TylerHK/OmniIntentXR.git`
2. Open `/Blueprints/OmniIntentXR.uproject` in **Unreal 5.4**.
3. Follow `Documentation/Setup.md` (coming soon) for Quest 3 build prerequisites.
4. Run the **DemoScene_Main** map and test gestures with keyboard shortcuts (`F` = Fireball, `T` = Telekinesis).

## Build & Packaging
Automated GitHub Actions (coming soon) will export Quest 3‑ready APKs. Manual build:
```bash
# In Unreal Editor
Platforms ▶ Android ▶ Package Project
```

## Roadmap Highlights
- EEG prototype integration (Q4 2025)
- Multiplayer co‑casting (Q1 2026)
- Procedural spell chaining system

## Contributing
Pull requests are welcome! Read [AGENTS.md](AGENTS.md) and [docs/usage.md](docs/usage.md).
`Documentation/CONTRIBUTING.md` and `Documentation/Setup.md` are not present yet; their setup steps remain unverified.
### OmniIntent — real-sensor ingestion

```python
from omniintent.ingest import load_quest3

batch = load_quest3("my_log.parquet", seq_len=60)
print({k: tuple(v.shape) for k, v in batch.items()})  # {'gaze': (1, 60, 3), 'hand_pose': (1, 60, 1)}
```

### Running the tests

```bash
pip install -r requirements.txt pytest
pytest            # pytest.ini puts ./src on the path; e2e test skips without OMNI_UI_URL
```

### Demo inference with Quest 3 data

```bash
# feed a real headset log into the transformer demo
python -m omniintent demo --ingest samples/quest3_tiny.csv

# fall back to synthetic when omitted
python -m omniintent demo
```

The CLI will invoke MultiModalTransformer when it’s installed, echoing a
JSON blob that lists both input and output tensor shapes so you can verify
live inference end-to-end.

### Training & fine-tuning datasets

Need a PyTorch-native stream of fixed-length Quest 3 windows?  Use
`Quest3Dataset`:

```python
from omniintent.ingest import Quest3Dataset
from torch.utils.data import DataLoader

ds = Quest3Dataset("my_logs/", seq_len=60, stride=30)
loader = DataLoader(ds, batch_size=8, shuffle=True)
for batch in loader:
    # batch["gaze"].shape → (8, 60, 3)
    ...
```

### End-to-End browser test

When a local dev server is running (`OMNI_UI_URL=http://localhost:3000`),
Playwright will upload a tiny Parquet sample, wait for inference, and assert
that the JSON payload contains the expected tensor shapes:

```bash
export OMNI_UI_URL=http://localhost:3000
pytest -k e2e
```

CI skips this test automatically when OMNI_UI_URL is not set.

---
© 2025 OmniIntent XR Team

## Repository boundary and Unreal assets
This repository contains the XR/sensor-ingestion prototype. The canonical
OmniIntent runtime and first-minute Council UX belong to
[TylerHK/IAmTheDrop](https://github.com/TylerHK/IAmTheDrop). Changes here do not
validate or alter that runtime.

`.gitattributes` routes future `.uasset` and `.umap` files through Git LFS.
No such files are currently tracked; this configuration uploads no assets and
rewrites no history. Install Git LFS before adding Unreal packages, and confirm
repository/account quota and any billing approval before uploading binaries.
No quota increase or paid service is authorized by this change. Existing CI
checks Python ingestion and CLI behavior; it does not build Unreal/Quest,
validate trained-model inference, or exercise browser E2E without `OMNI_UI_URL`.
