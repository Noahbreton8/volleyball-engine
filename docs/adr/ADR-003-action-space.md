# ADR-003: Action Space — Low-Level Joint Forces on Full Articulated Humanoid

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The simulation targets full physical realism. The agent body model and action space determine both what behaviors are learnable and how long training takes to converge.

## Decision

Each agent controls a **full articulated humanoid** body via **low-level continuous joint torques** (or target joint velocities, depending on the solver). The body model includes all major joints: spine, shoulders, elbows, wrists, hips, knees, ankles. The agent applies forces/torques each timestep; locomotion, jumping, and arm swing all emerge from learned joint control.

## Consequences

**Good:**
- Maximum physical realism — contact mechanics, body positioning, and arm swing are all physically grounded
- Emergent behaviors (dive, roll, jump serve) can arise naturally from physics rather than being hand-coded
- Magnus effect and spin on the ball directly depend on the velocity of the contacting body part at impact — only possible if the body is physically simulated

**Bad:**
- This is the hardest RL action space that exists for this problem. Locomotion learning alone (walking, running, stopping) can require tens of millions of steps
- Combined locomotion + object manipulation (volleyball contact) is an open research problem — convergence is not guaranteed without curriculum or reward shaping
- Action space dimensionality is high: ~20 joints × 3 DOF = ~60-dimensional continuous action space per agent, times 12 agents
- Requires a robust humanoid physics model in C++ (rigid body dynamics, joint limits, contact with floor and ball)

## Mitigation

Dense reward shaping (see ADR-004) is the primary tool for making this action space tractable. A locomotion pre-training phase (reward agents just for standing, walking, reaching a target) before introducing the ball is strongly recommended even if not formally a curriculum.

## Alternatives Rejected

- **Mid-level (target position + contact intent)**: Easier to train but hides the physics — Magnus effect cannot be computed from intent, only from actual arm velocity
- **High-level (discrete intents)**: Incompatible with real kinematics goal; reduces the sim to a state machine
