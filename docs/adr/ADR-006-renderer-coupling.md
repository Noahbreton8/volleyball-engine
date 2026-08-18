# ADR-006: Renderer Coupling — Headless Training, Post-Hoc Visualization

**Status:** Accepted  
**Date:** 2026-08-17

## Context

RL training requires maximum simulation throughput. Rendering is expensive and unnecessary during training. Visualization is needed for debugging trained policies and for the free-camera observation mode described in the product goal.

## Decision

The physics simulation runs **fully headless** during training. Episode state (all agent joint positions, ball position/velocity, phase) is recorded at every timestep to a compact binary format. The **3D renderer is a separate process** that reads recorded episode files and plays them back with a free-camera viewer.

## Data format

Each simulation frame records:
- Timestamp (fixed-timestep index)
- Ball: position (vec3), velocity (vec3), spin (vec3)
- Per agent (×12): root position (vec3), per-joint rotation (quaternion × ~20)
- Phase enum
- Score state

## Renderer requirements (separate from training)

- Free camera mode: user moves through 3D space freely (WASD + mouse look)
- Scrubable playback: pause, rewind, frame-step
- Agent highlighting: click agent to inspect reward/observation at that frame
- Court geometry: standard indoor volleyball court (18m × 9m), net at 2.43m (men's)

## Consequences

**Good:**
- Training throughput is not bottlenecked by GPU/rendering
- Renderer can be developed and iterated independently of physics/RL
- Recorded episodes can be replayed, shared, and analyzed offline
- Renderer can be any technology (OpenGL, Vulkan, even Blender import) without coupling to the physics engine

**Bad:**
- Debugging during training requires reading log files rather than watching the sim live
- A "live view" during training (useful for catching degenerate behavior early) is not available without adding a streaming mode
- Recorded episode files can be large for long episodes with 12 articulated humanoids at high timestep frequency

## Future option

A lightweight debug renderer (lines/spheres only, no skinned meshes) can be added as a streaming mode for live training inspection without the full visualization overhead.
