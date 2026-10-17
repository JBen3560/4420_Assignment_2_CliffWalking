#!/usr/bin/env python3
"""
Tabular Q-learning on Gymnasium's CliffWalking-v1.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame rendering

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENV_ID = "CliffWalking-v1"
MAX_STEPS = 200          # limit to avoid infinite loops
N_ROWS, N_COLS = 4, 12
START, GOAL = 36, 47
CLIFF = set(range(37, 47))
OPTIMAL_RETURN = -13     # up, 11x right, down
SAFE_RETURN = -17        # path along the top row
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


# Configuration
@dataclass(frozen=True)
class Config:
    alpha: float = 0.5            # learning rate
    gamma: float = 0.99           # discount factor
    epsilon_start: float = 0.1    # epsilon-greedy exploration rate at episode 0
    epsilon_end: float = 0.1      # floor for the schedule
    epsilon_decay: float = 0.98   # per-episode multiplicative decay
    q_init: float = 0.0           # initial value of every Q(s, a)
    episodes: int = 500
    eval_every: int = 5           # greedy evaluation every N training episodes

    def epsilon(self, episode: int) -> float:
        return max(self.epsilon_end, self.epsilon_start * self.epsilon_decay**episode)


def make_env(cfg: Config, render_mode: str | None = None) -> gym.Env:
    return gym.make(ENV_ID, max_episode_steps=MAX_STEPS, render_mode=render_mode)


# Agent
class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy action selection."""

    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float,
                 q_init: float, rng: np.random.Generator):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng
        self.Q = np.full((n_states, n_actions), q_init, dtype=np.float64)

    def act(self, state: int, epsilon: float) -> int:
        """Epsilon-greedy: explore with prob. epsilon, else exploit (random tie-break)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)].

        Only true termination (reaching the goal) zeroes the bootstrap term.
        Time-limit truncation is NOT terminal: the state still has future value.
        """
        target = r if terminated else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# Rollouts / evaluation
def greedy_rollout(env: gym.Env, Q: np.ndarray, seed: int | None = None,
                   collect_frames: bool = False):
    """Follow argmax Q (deterministic tie-break). Returns (return, states, frames)."""
    s, _ = env.reset(seed=seed)
    states, frames, total = [s], [], 0.0
    if collect_frames:
        frames.append(env.render())
    for _ in range(MAX_STEPS):
        s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
        total += r
        states.append(s)
        if collect_frames:
            frames.append(env.render())
        if terminated or truncated:
            break
    return total, states, frames


# Training
def train(cfg: Config, seed: int) -> dict:
    """Run one training run. Returns per-episode logs and the final Q-table."""
    env, eval_env = make_env(cfg), make_env(cfg)
    rng = np.random.default_rng(seed)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n,
                           cfg.alpha, cfg.gamma, cfg.q_init, rng)

    train_returns = np.zeros(cfg.episodes)
    cliff_falls = np.zeros(cfg.episodes)
    eval_returns = []

    for ep in range(cfg.episodes):
        eps = cfg.epsilon(ep)
        s, _ = env.reset(seed=seed if ep == 0 else None)
        total, falls = 0.0, 0
        while True:
            a = agent.act(s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, s_next, terminated)
            total += r
            falls += r == -100
            s = s_next
            if terminated or truncated:
                break
        train_returns[ep], cliff_falls[ep] = total, falls
        if (ep + 1) % cfg.eval_every == 0:
            eval_returns.append(greedy_rollout(eval_env, agent.Q)[0])

    env.close()
    eval_env.close()
    return {"train": train_returns, "falls": cliff_falls,
            "eval": np.array(eval_returns), "Q": agent.Q}


def _train_job(args):
    cfg, seed = args
    return train(cfg, seed)


def run_seeds(cfg: Config, seeds: list[int], workers: int) -> list[dict]:
    jobs = [(cfg, s) for s in seeds]
    if workers <= 1:
        return [_train_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_train_job, jobs))


# Metrics and aggregation
def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average along the last axis (shorter window at the start)."""
    c = np.cumsum(np.insert(x, 0, 0.0, axis=-1), axis=-1)
    idx = np.arange(1, x.shape[-1] + 1)
    lo = np.maximum(idx - window, 0)
    return (c[..., idx] - c[..., lo]) / (idx - lo)


def mean_ci(arr: np.ndarray):
    """Mean and 95% CI half-width across axis 0 (seeds)."""
    m = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0 * m
    return m, ci


def summarize(cfg: Config, runs: list[dict]) -> dict:
    train = np.stack([r["train"] for r in runs])
    falls = np.stack([r["falls"] for r in runs])
    ev = np.stack([r["eval"] for r in runs])
    first_opt = []
    for row in ev:
        hit = np.flatnonzero(row >= OPTIMAL_RETURN)
        if hit.size:
            first_opt.append((hit[0] + 1) * cfg.eval_every)
    return {
        "final_train_return": train[:, -50:].mean(),
        "final_greedy_return": ev[:, -1].mean(),
        "frac_seeds_optimal": float((ev[:, -1] >= OPTIMAL_RETURN).mean()),
        "mean_episodes_to_optimal": float(np.mean(first_opt)) if first_opt else float("nan"),
        "total_cliff_falls": falls.sum(axis=1).mean(),
        # max_a Q(start, a); for the optimal policy this approaches -(1 - g^13)/(1 - g)
        "start_state_value": float(np.mean([r["Q"][START].max() for r in runs])),
    }


# Plotting
def _style_return_axis(ax, title, ylabel="Return per episode"):
    ax.axhline(OPTIMAL_RETURN, color="green", ls="--", lw=1, alpha=0.7)
    ax.axhline(SAFE_RETURN, color="orange", ls=":", lw=1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_curves(ax_train, ax_eval, ax_falls, results: dict[str, list[dict]], cfg: Config,
                smooth: int = 10, color_cycle=None):
    """Draw mean +/- 95% CI curves for each labelled group of runs."""
    colors = color_cycle or plt.cm.viridis(np.linspace(0, 0.95, len(results)))
    for (label, runs), color in zip(results.items(), colors):
        tr = moving_average(np.stack([r["train"] for r in runs]), smooth)
        fl = moving_average(np.stack([r["falls"] for r in runs]), smooth)
        ev = np.stack([r["eval"] for r in runs])
        x_tr = np.arange(1, tr.shape[1] + 1)
        x_ev = np.arange(1, ev.shape[1] + 1) * cfg.eval_every
        for ax, y, x in ((ax_train, tr, x_tr), (ax_eval, ev, x_ev), (ax_falls, fl, x_tr)):
            m, ci = mean_ci(y)
            ax.plot(x, m, color=color, label=label, lw=1.6)
            ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.15)


def style_three_panel(axes, smooth):
    _style_return_axis(axes[0], f"Training return (ε-greedy, {smooth}-ep. moving avg.)")
    _style_return_axis(axes[1], "Greedy-policy return (evaluation, ε = 0)")
    axes[2].set_title(f"Cliff falls per episode (training, {smooth}-ep. avg.)")
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("Falls")
    axes[2].grid(alpha=0.3)
    axes[0].set_ylim(-250, 0)
    axes[1].set_ylim(-210, 0)


def plot_baseline(runs, cfg, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    plot_curves(*axes, {"baseline": runs}, cfg, color_cycle=["tab:blue"])
    style_three_panel(axes, 10)
    axes[0].text(cfg.episodes * 0.98, OPTIMAL_RETURN + 4, "optimal (-13)", color="green",
                 ha="right", va="bottom", fontsize=8)
    axes[0].text(cfg.episodes * 0.98, SAFE_RETURN - 4, "safe path (-17)", color="darkorange",
                 ha="right", va="top", fontsize=8)
    fig.suptitle(f"Baseline: α={cfg.alpha}, γ={cfg.gamma}, ε={cfg.epsilon_start}, "
                 f"Q₀={cfg.q_init}  (mean ± 95% CI over {len(runs)} seeds)")
    fig.tight_layout()
    fig.savefig(out / "baseline_learning_curves.png", dpi=150)
    plt.close(fig)


def plot_policy(Q: np.ndarray, cfg: Config, out: Path, seed: int):
    env = make_env(cfg)
    ret, path, _ = greedy_rollout(env, Q, seed=seed)
    env.close()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, N_COLS)
    ax.set_ylim(N_ROWS, 0)
    ax.set_aspect("equal")
    for s in range(N_ROWS * N_COLS):
        r, c = divmod(s, N_COLS)
        face = "white"
        if s in CLIFF:
            face = "#f4a6a6"
        elif s == START:
            face = "#cfe8cf"
        elif s == GOAL:
            face = "#fff2a8"
        elif s in path:
            face = "#d6e6f7"
        ax.add_patch(plt.Rectangle((c, r), 1, 1, fc=face, ec="gray", lw=0.8))
        if s in CLIFF:
            ax.text(c + 0.5, r + 0.5, "cliff", ha="center", va="center", fontsize=8, color="darkred")
        elif s == GOAL:
            ax.text(c + 0.5, r + 0.5, "G", ha="center", va="center", fontsize=16, weight="bold")
        else:
            q = Q[s]
            if np.ptp(q) > 0:  # skip never-updated states (all-equal Q)
                ax.text(c + 0.5, r + 0.42, ARROWS[int(np.argmax(q))], ha="center", va="center",
                        fontsize=18)
                ax.text(c + 0.5, r + 0.85, f"{q.max():.1f}", ha="center", va="center",
                        fontsize=7, color="dimgray")
            if s == START:
                ax.text(c + 0.08, r + 0.2, "S", fontsize=10, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Learned greedy policy (arrow = argmax Q, number = max Q); blue = greedy path; "
                 f"return = {ret:.0f}")
    fig.tight_layout()
    fig.savefig(out / "baseline_policy.png", dpi=150)
    plt.close(fig)


def plot_sweep(name: str, results: dict[str, list[dict]], cfg: Config, out: Path, n_seeds: int):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    plot_curves(*axes, results, cfg)
    style_three_panel(axes, 10)
    axes[1].legend(title=name, fontsize=8, loc="lower right")
    fig.suptitle(f"Sweep over {name}  (other settings at baseline; mean ± 95% CI over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(out / f"sweep_{name}.png", dpi=150)
    plt.close(fig)


# Video
def record_video(Q: np.ndarray, cfg: Config, path: Path, seed: int, fps: int = 3,
                 hold_seconds: float = 2.0) -> float:
    import imageio.v2 as imageio

    env = make_env(cfg, render_mode="rgb_array")
    ret, _, frames = greedy_rollout(env, Q, seed=seed, collect_frames=True)
    env.close()
    frames = frames + [frames[-1]] * int(hold_seconds * fps)
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)
    return ret


# Sweeps
def build_sweeps() -> dict[str, dict[str, dict]]:
    """name -> {label: config overrides}. Everything not listed stays at baseline."""
    const_eps = lambda e: {"epsilon_start": e, "epsilon_end": e}
    return {
        "alpha": {f"α={a}": {"alpha": a} for a in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)},
        "gamma": {f"γ={g}": {"gamma": g} for g in (0.5, 0.8, 0.9, 0.95, 0.99, 1.0)},
        "epsilon": {
            **{f"ε={e}": const_eps(e) for e in (0.01, 0.05, 0.1, 0.2, 0.5)},
            "ε decay 1.0→0.01": {"epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.98},
        },
    }


# Main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seeds", type=int, default=20, help="independent runs per setting")
    p.add_argument("--sweeps", nargs="*", default=["alpha", "gamma", "epsilon"],
                   choices=["alpha", "gamma", "epsilon"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--quick", action="store_true", help="150 episodes, 5 seeds")
    args = p.parse_args()

    if args.quick:
        args.episodes, args.seeds = 150, 5

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Config(episodes=args.episodes)
    seeds = list(range(args.seeds))
    t0 = time.time()

    # baseline
    print(f"Baseline: {base}")
    runs = run_seeds(base, seeds, args.workers)
    plot_baseline(runs, base, out)
    Q0 = runs[0]["Q"]
    np.save(out / "q_table_baseline.npy", Q0)
    plot_policy(Q0, base, out, seed=0)
    base_summary = summarize(base, runs)
    print("  " + ", ".join(f"{k}={v:.2f}" for k, v in base_summary.items()))

    if not args.no_video:
        ret = record_video(Q0, base, out / "trained_agent.mp4", seed=0)
        print(f"  Video written (greedy return {ret:.0f})")

    # sweeps
    for name in args.sweeps:
        print(f"Sweep: {name}")
        results = {}
        for label, overrides in build_sweeps()[name].items():
            cfg = replace(base, **overrides)
            rs = run_seeds(cfg, seeds, args.workers)
            results[label] = rs
            print("  " + f"{label:<18}" + "  ".join(f"{k}={v:.2f}" for k, v in summarize(cfg, rs).items()))
        plot_sweep(name, results, base, out, args.seeds)

    print(f"Done in {time.time() - t0:.0f}s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
