# ADR-009: Visualization — MuJoCo Viewer with Episode Recording

**Status:** Accepted  
**Date:** 2026-08-17

## Context

Two distinct visualization needs exist:
1. **Training inspection**: observe agent behavior at intervals during training (not real-time) to catch degenerate policies early
2. **Results viewing**: watch trained policies play full rallies with free-camera 3D navigation and scrubable playback

The physics engine is MuJoCo, which ships its own high-quality interactive viewer. Building a custom renderer would duplicate this capability without benefit.

## Decision

Use **MuJoCo's built-in passive viewer** (`mujoco.viewer`) for all 3D visualization. Record episode state to disk at configurable intervals during training; the viewer replays these recordings on demand. Training metrics (reward curves, win rates, contact success rates) are logged to **TensorBoard**.

### Training inspection flow
1. Every N training episodes (configurable, e.g. every 500), the environment runs one episode with state recording enabled
2. Episode is saved as a compact binary (NumPy `.npz`): per-timestep `qpos`, `qvel`, ball position/velocity/spin, phase, reward breakdown
3. Viewer can be launched at any time against the latest (or any) recording without interrupting training
4. Viewer runs as a **separate process** — `python -m volleyball.viewer recordings/episode_00500.npz`

### Results viewing flow
1. Load a policy checkpoint
2. Run N evaluation episodes with recording enabled
3. Launch viewer against any recording
4. Free camera: WASD + mouse look (MuJoCo viewer default controls)
5. Scrub: left/right arrow keys step frames; spacebar pauses

### Training metrics (TensorBoard)
- Per-role episode reward (setter, OH, MB, OPP, libero)
- Rally win rate (self-play: team A vs team B)
- Ball contact rate per agent per episode
- Pass accuracy (% of passes with vector pointing to set zone within threshold)
- Set accuracy (% of sets landing in attack window)
- Attack success rate (% of attacks landing in-bounds)

## Recording format

```
episode_NNNNN.npz
├── timesteps: int              # number of frames
├── qpos: float32[T, nq]       # MuJoCo full joint position state
├── qvel: float32[T, nv]       # MuJoCo full joint velocity state
├── ball_pos: float32[T, 3]
├── ball_vel: float32[T, 3]
├── ball_spin: float32[T, 3]
├── phase: int8[T]              # phase enum per frame
├── contacts: int8[T, 12]      # which agent contacted ball this frame
└── rewards: float32[T, 12]    # per-agent shaped reward per frame
```

## Consequences

**Good:**
- Zero custom renderer work — MuJoCo viewer is production quality with free camera, lighting, and mesh rendering
- Recording is cheap: `qpos`/`qvel` for 12 humanoids + ball is ~few KB per timestep
- Viewer is decoupled from training — inspect any past episode without rerunning
- TensorBoard is the RL community standard for metrics
- Free camera mode works out of the box in MuJoCo viewer (right-click drag to orbit, scroll to zoom, WASD-style with Ctrl+drag)

**Bad:**
- MuJoCo viewer appearance is constrained to MuJoCo's rendering style (no custom shaders, no stylized art)
- Scrub controls are keyboard-only (no timeline scrubber UI)
- Viewer cannot overlay custom annotations (e.g., reward values per agent) without forking the viewer code

## Future option

A lightweight overlay using **Dear ImGui** can be added on top of the MuJoCo viewer to display per-agent reward, observation values, and phase state during playback. This is an enhancement, not a requirement for initial implementation.
