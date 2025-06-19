# OmniIntentXR
Unified XR project enabling semi-omnipotent intent via gesture, voice, and gaze (Quest 3).

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
Pull requests are welcome! Please read `Documentation/CONTRIBUTING.md`.
### OmniIntent — real-sensor ingestion

```bash
python -m omniintent.ingest.quest3_ingest my_log.parquet
    | jq .gaze   # preview tensor shape
```

# --- Demo inference with Quest 3 data -----------------------------------

# feed a real headset log into the transformer demo
python -m omniintent demo --ingest samples/quest3_tiny.csv

# fall back to synthetic when omitted
python -m omniintent demo

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
