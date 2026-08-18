# ADR-005: Agent Policy Structure — Independent Policies with Role-Specific Rewards

**Status:** Accepted  
**Date:** 2026-08-17

## Context

6v6 indoor volleyball has distinct positional roles (libero, setter, outside hitter, middle blocker, opposite) with different responsibilities. Agents must specialize without centralized coordination at execution time.

## Decision

Each of the 6 players on a team has an **independent policy network** trained with role-specific reward functions. All 6 policies share the same network architecture but are initialized and trained independently. The opposing team in self-play is a lagged snapshot of the current team's policies (frozen copy updated periodically).

### Policy per role

| Role | Primary reward signal |
|---|---|
| Libero | Pass vector alignment → set zone |
| Back-row (non-libero) | Pass vector alignment → set zone |
| Setter | Set placement → attack window |
| Outside Hitter | Attack landing position (in-bounds, away from defenders) |
| Middle Blocker | Attack landing position; blocking reward when at net |
| Opposite | Attack landing position |

### Observation space per agent (local)

Each agent observes:
- Own joint positions and velocities (~60 dims)
- Ball position and velocity (6 dims)
- All teammate positions and velocities (5 × ~12 dims simplified)
- All opponent positions (6 × 3 dims)
- Phase indicator (serve/receive/set/attack/rally in play)
- Own role indicator (one-hot)

The **centralized critic** additionally observes global state: all joint states for all 12 agents + ball state.

## Consequences

**Good:**
- Role-specific rewards produce natural specialization
- Independent policies can be swapped, frozen, or replaced per role without retraining the whole team
- Scales to self-play naturally: opponent = frozen copy of own team policies

**Bad:**
- 6 independent networks per team means 6× the training compute relative to a shared policy
- Coordination between setter and attacker is implicit (must emerge from training) rather than explicit
- Rotation rule (players must rotate positions after sideout) complicates role assignment — a setter may end up in a back-row position

## Policy switching on rotation — Resolved

Volleyball rotation means role and court position decouple over time. The decision:

- **Setter**: always uses the setter policy, regardless of court position. The setter's job (setting) only occurs in the front row; in the back row they primarily pass and the reward shaping accounts for this.
- **Outside Hitter**: has two policies — `OH_front` (attack, block) and `OH_back` (pass, cover). Policy selected by current court position.
- **Middle Blocker**: has two policies — `MB_front` (quick attack, block) and `MB_back` (pass). In practice, the Libero replaces the MB in the back row, so `MB_back` is rarely used.
- **Opposite**: has two policies — `OPP_front` (attack, block) and `OPP_back` (pass, pipe attack).
- **Libero**: single back-row policy. Never plays front row.

This gives **8 distinct policies per team**. All are trained simultaneously under MAPPO with the centralized critic observing global state.
