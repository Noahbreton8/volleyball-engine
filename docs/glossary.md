# Domain Glossary — Volleyball Engine

## Volleyball Domain

**Attack window** — The spatial region near the net where a set ball should arrive to enable an effective attack: within ~1m of the net, at a height between 2.4m and 3.5m, within the lateral bounds of the court antenna.

**Back-row player** — One of the three players in positions 1, 6, or 5 (behind the 3m attack line). Back-row players are the primary receivers of serve and attack. Their primary reward signal is pass accuracy.

**Dig** — A defensive contact, typically with the forearms, used to receive an attack or hard-driven ball. A good dig directs the ball to the set zone.

**Front-row player** — One of the three players in positions 2, 3, or 4 (between the net and the 3m attack line). Front-row players execute sets and attacks.

**Libero** — A specialized defensive player who wears a contrasting jersey and cannot attack above net height. Replaces the middle blocker in back-row rotations. Primary reward: pass accuracy to set zone.

**Magnus effect** — The aerodynamic force on a spinning ball that causes its trajectory to curve. In volleyball, topspin drives the ball downward faster; float serves have minimal spin and move unpredictably. The physics engine models this as a force proportional to spin × velocity.

**Middle blocker (MB)** — A front-row player who specializes in blocking opposing attacks at the net and executing quick attacks (quick sets near the setter). Positioned at position 3.

**Net** — A horizontal barrier at the center of the court at height 2.43m (men's). The ball must cross the net above this height. Contact with the net by a player is a fault.

**Opposite (OPP)** — The player in position 2, opposite the setter in the rotation. Typically a powerful attacker who also receives some back-row passes.

**Outside hitter (OH)** — The primary attacker, positioned at position 4 (left front) or position 5 (left back). Receives the majority of sets and attacks from the left antenna.

**Pass** — See *dig*. In context of serve receive, a pass is the first contact that redirects the ball from the server toward the set zone.

**Phase** — The discrete state of a rally: `SERVE`, `RECEIVE`, `SET`, `ATTACK`, `BLOCKED`, `RALLY_IN_PLAY`, `POINT`. The phase determines which reward signals are active.

**Rally** — The sequence of contacts between the serve and the point. A rally ends when the ball hits the floor, goes out of bounds, or a team commits a fault.

**Rotation** — After winning a sideout, the receiving team rotates clockwise one position. This decouples a player's court position from their specialist role over the course of a set.

**Set** — The second contact by the receiving team, intended to deliver the ball to an attacker at the attack window. A good set is measured by proximity of ball apex to the attack window.

**Set zone** — The spatial target for a back-row pass: approximately 1-3m from the net, near position 3 (center-front), at a height that allows the setter to make a controlled set. Defined in the engine as a configurable region.

**Setter** — The player at position 3 (or the designated setter in rotation). Their role is to receive the pass and deliver an accurate set to an attacker. Primary reward: set placement within the attack window.

**Sideout** — Winning a rally as the receiving team, earning the right to serve and rotating the team.

**Spin (angular velocity)** — Rotational velocity of the ball, expressed as a 3D vector (rad/s). Combined with linear velocity, spin determines the Magnus force and contact outcome.

---

## RL / Simulation Domain

**Action space** — The set of actions available to an agent each timestep. In this engine: continuous joint torques (or target velocities) for ~20 joints of a full articulated humanoid. Dimensionality: ~60 per agent.

**Centralized critic** — In CTDE, a value function that observes global state (all 12 agents + ball) during training only. Not available at execution time. Reduces non-stationarity in multi-agent training.

**CTDE (Centralized Training, Decentralized Execution)** — A multi-agent RL paradigm where agents train with access to global state (via a centralized critic) but execute using only their own local observations. Enables coordination without a runtime communication oracle.

**Decentralized execution** — Each agent acts on its own local observation at runtime, without access to other agents' internal states or a global coordinator.

**Episode** — One complete game unit used for training. TBD: whether an episode is one rally, one set, or one match. Shorter episodes = more resets = faster exploration; longer episodes = more context for coordination to emerge.

**Fixed timestep** — The simulation advances in uniform time increments (e.g., 1/120s). All physics integration, contact resolution, and agent actions occur at this frequency. Ensures deterministic, reproducible playback.

**Headless** — Running the physics simulation without a display or renderer attached. Used during RL training to maximize throughput.

**IPPO (Independent PPO)** — Each agent runs its own PPO training loop, treating all other agents as part of the environment. Simpler than MAPPO but prone to coordination failure.

**Local observation** — The subset of global state visible to a single agent at execution time: own joint state, ball state, teammate positions, opponent positions, phase, role.

**MAPPO (Multi-Agent PPO)** — PPO extended to multi-agent settings with a shared centralized critic. The algorithm used in this project (see ADR-002).

**Non-stationarity** — In multi-agent RL, the environment appears non-stationary from each agent's perspective because other agents' policies are changing simultaneously. The centralized critic in CTDE mitigates this.

**Observation space** — The inputs available to an agent's policy network each timestep. Includes own joint state, ball state, teammate positions, opponent positions, phase, and role indicator.

**Policy network** — The neural network that maps an agent's observation to a distribution over actions. Each of the 6 roles has an independent policy network.

**pybind11** — A C++ library that exposes C++ classes and functions to Python with minimal overhead. Used to wrap the C++ physics engine in a Gym-compatible Python interface for RL training.

**Reward hacking** — When an agent finds a degenerate strategy that maximizes the shaped reward without achieving the intended behavior. A known risk with dense reward shaping.

**Reward shaping** — Adding intermediate, dense reward signals to guide learning beyond the sparse terminal reward. See ADR-004 for the full hierarchy.

**Self-play** — Training agents by having them compete against a frozen or lagged copy of their own policy. Produces a natural curriculum: as the agent improves, so does its opponent.

**Timestep** — One discrete advance of the simulation. At 120Hz, one timestep = ~8.3ms of simulated time.

**MuJoCo** — MuJoCo Physics (Multi-Joint dynamics with Contact). The rigid-body physics solver used by this engine, called via its C API from the C++ simulation layer. Industry standard for RL on articulated humanoids. See ADR-008.

**MJCF** — MuJoCo XML format for defining body geometry, joint structure, actuators, and contact parameters. Used to define the 12 humanoid agents and the volleyball.

**Magnus force** — Aerodynamic force on a spinning ball: `F = k * (ω × v)`. Applied as a custom external force via MuJoCo's `xfrc_applied` each timestep. Causes topspin serves to drop fast and float serves to move unpredictably.

**xfrc_applied** — MuJoCo API field for applying external forces and torques to bodies each timestep. Used to inject the Magnus force on the ball without modifying MuJoCo's internal contact solver.

**OH_front / OH_back** — The two policies of an Outside Hitter, selected by current court position after rotation. `OH_front` handles attacking and blocking; `OH_back` handles passing and court coverage.

**MB_front / MB_back** — The two policies of a Middle Blocker. `MB_back` is rarely active in practice because the Libero replaces the MB in the back row.

**OPP_front / OPP_back** — The two policies of the Opposite hitter. `OPP_back` includes pipe attack (back-row attack from position 6).

**Rally episode** — One training episode: begins on serve, ends on point resolution. See ADR-007. Score and rotation state do not persist across episodes during early training.
