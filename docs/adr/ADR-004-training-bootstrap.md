# ADR-004: Training Bootstrap — Dense Reward Shaping

**Status:** Accepted  
**Date:** 2026-08-17

## Context

Low-level joint control + 12 independent agents + sparse terminal reward (point scored/lost) creates an extreme cold-start problem. Agents will fail to make contact with the ball for millions of steps, receiving no useful gradient signal from the game outcome alone.

## Decision

Use **dense reward shaping** as the primary bootstrap mechanism. Each agent receives a shaped reward at every timestep, not just at point resolution. Shaped rewards are role-specific and hierarchically structured (contact quality rewards are higher frequency than rally outcome rewards).

### Reward hierarchy (approximate, to be tuned)

| Signal | Frequency | Example |
|---|---|---|
| Locomotion reward | Every step | Reward for moving toward ball when ball is in agent's zone |
| Contact quality reward | On ball contact | Reward proportional to: outgoing ball vector alignment with target zone, arm velocity at contact moment |
| Phase completion reward | On phase transition | Serve → dig → set → attack chain completed legally |
| Rally outcome reward | On point resolution | Win rally: +1, Lose rally: -1 |

### Role-specific contact reward targets

- **Libero / Back-row**: Outgoing ball vector must point toward the **set zone** (±N meters from center of court, height H). Reward = cosine similarity of actual outgoing vector vs. ideal vector to set zone.
- **Setter**: Outgoing ball must arrive at an **attack window** (height range, proximity to net, lateral position). Reward = distance from ball apex to ideal attack position.
- **Outside Hitter / Middle Blocker / Opposite**: Outgoing ball must land **in-bounds** on the opponent's side. Reward = negative distance from ball landing point to nearest out-of-bounds line (inverted), bonus for landing away from opponent positions.

## Consequences

**Good:**
- Dense signals give agents a learning gradient even before they can execute a full rally
- Role-specific shaping naturally produces specialization without requiring a centralized coordinator
- Reward hierarchy mirrors real volleyball skill acquisition

**Bad:**
- Reward shaping can cause **reward hacking**: agents optimize the shaped reward without achieving the intended behavior (e.g., tapping the ball gently toward the set zone every time instead of developing a real dig)
- Shaped rewards must be carefully scaled relative to each other — wrong scales produce degenerate policies
- Requires iterative tuning; first designed values will almost certainly be wrong

## Alternatives Rejected

- **Curriculum learning**: Valid and complementary, but adds structural complexity. Can be introduced later if reward shaping alone proves insufficient.
- **Imitation pre-training**: Requires motion capture data or scripted reference policies — not in scope for initial implementation.
- **Sparse reward only**: Will not converge in reasonable time given the action space and agent count.
