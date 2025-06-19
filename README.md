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

---
© 2025 OmniIntent XR Team
