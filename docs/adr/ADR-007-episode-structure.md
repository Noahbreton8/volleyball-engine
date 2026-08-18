# ADR-007: Episode Structure — One Rally Per Episode

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The RL training loop must balance exploration speed (more resets = more diverse starting states) against context length (agents need enough time to experience multi-contact sequences and learn coordination). The episode boundary determines both training throughput and what behaviors agents can learn.

## Decision

One **rally** equals one training episode. An episode begins with the serve and ends when the ball hits the floor, goes out of bounds, or a fault is committed. Score state and rotation state do not persist across episodes during early training.

## Consequences

**Good:**
- Maximum reset frequency — agents encounter many different starting states per wall-clock hour
- Reward signal arrives quickly (rally resolution is the terminal reward)
- Avoids the need for agents to learn rotation strategy, score-state awareness, or set/match-level tactics in early training
- Simpler episode bookkeeping: no persistent match state

**Bad:**
- Agents never experience the full consequences of rotation — a setter rotating to the back row is a state that only emerges mid-match, not within a rally
- Agents cannot learn serve-receive tactics that depend on score or set context (e.g., risky jump serve when down match point)
- Once policies are mature, extending to set-level or match-level episodes will require re-evaluation of reward scaling

## Future extension

When policies have converged at the rally level, extend episodes to a full **set** (25 points, win by 2) to allow rotation consequences and serve strategy to emerge. Match-level episodes are unlikely to be necessary for the research goal.

## Starting state distribution

Each episode initializes with:
- Serving team chosen randomly or by round-robin
- Players in valid rotational positions (randomized rotation offset to ensure all rotations are visited during training)
- Ball at server's position
- All agents initialized to their default standing pose
