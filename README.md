# Tabular Q-learning on CliffWalking-v1

A from-scratch tabular Q-learning agent for Gymnasium's `CliffWalking-v1`, plus hyperparameter
sweeps over learning rate (α), discount factor (γ), and exploration rate (ε). One script
produces every plot and an MP4 of the trained agent.

## Usage

```bash
pip install -r requirements.txt
python cliffwalking_qlearning.py
```

## The environment

| Item | Value |
|---|---|
| Grid / states | 4x12, `Discrete(48)`, `state = row * 12 + col` |
| Start / goal | state 36 (bottom-left) / state 47 (bottom-right) |
| Actions | `Discrete(4)`: 0 up, 1 right, 2 down, 3 left |
| Rewards | -1 per step; -100 for stepping on the cliff (agent is sent back to start) |
| Termination | Only on reaching the goal. Falling off the cliff does **not** end the episode |
| Time limit | None by default in Gymnasium 1.4 (`env.spec.max_episode_steps` is `None`), so the script sets `max_episode_steps=200` |
| Reference returns | Optimal path = **-13**; safe top-row path = **-17** |

## The algorithm

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

- `QLearningAgent` holds a 48x4 Q-table initialized to `q_init`.
- Actions are ε-greedy, with random tie-breaking among equal Q-values.
- **Terminal handling:** the bootstrap term is dropped only when `terminated` is true (goal reached).
  Time-limit `truncated` episodes still bootstrap, since the state isn't truly terminal. Cliff falls
  also bootstrap, because the episode continues from the start state.
- ε follows `max(epsilon_end, epsilon_start * epsilon_decay ** episode)`. A constant ε is just
  `epsilon_start == epsilon_end`.

**Baseline:** α=0.5, γ=0.99, ε=0.1 (constant), Q₀=0, 500 episodes, 20 seeds.
Each sweep changes one setting and keeps the rest at baseline.

| Sweep | Values |
|---|---|
| α | 0.05, 0.1, 0.25, 0.5, 0.75, 1.0 |
| γ | 0.5, 0.8, 0.9, 0.95, 0.99, 1.0 |
| ε | 0.01, 0.05, 0.1, 0.2, 0.5, and decay 1.0→0.01 (x0.98 per episode) |

## What it measures

Every plot has three panels, with mean ± 95% CI across seeds:

1. **Training return** (ε-greedy, 10-episode moving average): what the agent earns while exploring.
2. **Greedy return** (ε=0, evaluated every 5 episodes): the quality of the policy it has learned.
3. **Cliff falls per episode** during training.

The script also prints a summary line per setting to the console: `final_train_return` (mean of last
50 episodes), `final_greedy_return`, `frac_seeds_optimal` (share of seeds whose final greedy policy
hits -13), `mean_episodes_to_optimal` (first evaluation at -13, among seeds that got there),
`total_cliff_falls`, and `start_state_value` (max_a Q(start, a)).

## Outputs (`results/`)

| File | Contents |
|---|---|
| `baseline_learning_curves.png` | Baseline training / greedy return and cliff falls |
| `baseline_policy.png` | Learned arrows, max-Q per cell, greedy path |
| `sweep_alpha.png`, `sweep_gamma.png`, `sweep_epsilon.png` | Sweep comparisons |
| `trained_agent.mp4` | Greedy rollout of the seed-0 baseline agent (13 steps, held on the final frame) |
| `q_table_baseline.npy` | Learned 48x4 Q-table |