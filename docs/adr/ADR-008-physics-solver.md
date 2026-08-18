# ADR-008: Physics Solver — MuJoCo via C API

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The simulation requires stable articulated humanoid dynamics for 12 agents simultaneously, plus a volleyball with Magnus effect and spin-dependent contact. Writing rigid-body dynamics from scratch in C++ is a multi-year effort. A solver must be chosen.

## Decision

Use **MuJoCo** (MuJoCo Physics, now free under DeepMind ownership) via its **C API**, called directly from the C++ simulation layer. The volleyball's Magnus effect is implemented as a custom external force applied via `mjData.xfrc_applied` each timestep, computed from ball spin × velocity × Magnus coefficient.

## Architecture

```
C++ Simulation Layer
├── mujoco C API (libmujoco.so)
│   ├── Humanoid models ×12 (MJCF XML)
│   └── Volleyball model (sphere, mass ~270g)
├── Magnus force applicator (custom C++)
├── Volleyball rule enforcer (contact counting, out-of-bounds, net fault)
└── pybind11 boundary → Python Gym interface
```

## Why MuJoCo over alternatives

| Criterion | MuJoCo | Bullet | From Scratch |
|---|---|---|---|
| Humanoid stability | Excellent (industry standard) | Requires heavy tuning | N/A |
| RL integration | Native (dm_control, mujoco-py lineage) | PyBullet exists but less maintained | Custom |
| Custom forces (Magnus) | `xfrc_applied` per body | `applyForce` API | Full control |
| Speed (headless) | Very fast | Fast | Depends |
| Licensing | Free (Apache 2.0) | Free (zlib) | N/A |
| Humanoid MJCF models | Freely available | Limited | N/A |

## Model files

MuJoCo uses MJCF (XML) to define body geometry, joint structure, and actuators. Standard humanoid MJCF models are available from DeepMind's `dm_control` suite and can be adapted for volleyball-appropriate proportions and joint limits.

The volleyball is modeled as a sphere (circumference ~65cm, mass 270g) with:
- Restitution coefficient: ~0.8
- Rolling friction tuned for hardwood court
- Spin tracked as a separate 3D angular velocity state
- Magnus force: `F_magnus = k * (ω × v)` where k is tuned to match real ball trajectories

## Consequences

**Good:**
- Months of physics implementation work avoided
- MuJoCo humanoids are already validated for RL locomotion tasks
- Magnus force as external force is clean and tunable
- C API gives full performance — no Python overhead in the physics loop

**Bad:**
- MuJoCo's contact model is generalized; ball deformation on player contact is not physically modeled (contact is impulse-based)
- MJCF model authoring has a learning curve
- MuJoCo's solver may not perfectly match real volleyball ball-floor rebound characteristics without coefficient tuning
