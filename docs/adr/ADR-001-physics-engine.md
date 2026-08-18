# ADR-001: Physics Engine — C++ with Python Bindings

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The simulation engine must run headless at much faster than real-time during RL training (potentially millions of timesteps per hour), while also supporting full real kinematics: Magnus effect on the ball, spin, contact mechanics, and full articulated humanoid bodies. The 3D renderer only attaches post-training for visualization.

## Decision

Implement the physics engine in **C++** and expose it to the RL training loop via **pybind11** Python bindings. The engine runs as a standalone headless simulation; the 3D renderer is a separate process that reads recorded episode state.

## Consequences

**Good:**
- C++ gives maximum simulation throughput — critical for RL sample efficiency
- pybind11 allows the RL loop to live in Python, accessing the full ecosystem (PyTorch, SB3, RLlib)
- Renderer is fully decoupled — no rendering overhead during training
- Physics fidelity is not limited by engine abstractions (unlike Unity/Godot physics)

**Bad:**
- Two languages to maintain across the codebase boundary
- pybind11 binding layer must be kept in sync as the C++ API evolves
- Debugging across the C++/Python boundary is harder than a single-language stack
- Full articulated humanoid physics in C++ is a significant implementation effort (may want to evaluate integrating Bullet or MuJoCo as the rigid-body solver rather than writing from scratch)

## Alternatives Rejected

- **Unity/Godot**: Physics capped at engine timestep, hard to run headless at training speed, limited control over contact model
- **Pure Python (PyBullet/MuJoCo-Python)**: Sufficient for some RL tasks but sacrifices fine-grained control over contact mechanics and Magnus force integration
- **Rust**: Better memory safety than C++ but smaller ecosystem for physics/RL integration
