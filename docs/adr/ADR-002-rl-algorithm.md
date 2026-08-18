# ADR-002: RL Algorithm — CTDE with MAPPO

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The system has 12 independent agents (6 per team) with role-differentiated reward functions. Agents must coordinate (e.g., setter and outside hitter must implicitly negotiate ball trajectory) but act on decentralized observations at runtime. Self-play is used to generate the opposing team.

## Decision

Use **Centralized Training with Decentralized Execution (CTDE)** via **Multi-Agent PPO (MAPPO)**. During training, a centralized critic observes global state (all agent positions, ball state, teammate intentions). At execution time, each agent's actor acts on its own local observation only.

## Consequences

**Good:**
- Centralized critic solves the non-stationarity problem: each agent's policy improves against a stable value baseline that accounts for all teammates
- Decentralized actors at execution time means policies are deployable without a communication oracle
- PPO's clipped surrogate objective is stable under the high-variance reward signals expected from reward shaping on a sparse contact sport
- Self-play is straightforward: opponent team is a frozen or lagged copy of the current policy

**Bad:**
- Global state during training means the centralized critic must ingest a large observation vector (12 agents × N joint states + ball state)
- MAPPO has higher implementation complexity than IPPO (independent PPO)
- Credit assignment remains hard: when a rally is won, which of the 6 actions in the chain deserves reward?

## Alternatives Rejected

- **IPPO (Independent PPO)**: Simpler but treats teammates as part of the environment — coordination between setter and attacker is unlikely to emerge reliably
- **MADDPG**: Off-policy, better sample efficiency in theory, but notoriously unstable with full articulated humanoid action spaces
- **Centralized single policy**: One policy controlling all 6 players loses role specialization and produces uniform behavior
