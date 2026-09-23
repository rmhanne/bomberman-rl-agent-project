"""
module to run matches and get their stats

matches are independent, so they are dispatched across processes
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results'


@dataclass
class Match:
    """One invocation of main.py play."""

    agents: list
    scenario: str = 'classic'
    n_rounds: int = 100
    seed: int = None
    train: int = 0
    label: str = ''
    stats_path: Path = None
    extra_args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)

    def environment(self):
        return {**os.environ, **{k: str(v) for k, v in self.env.items()}}

    def command(self):
        stats = self.stats_path or (RESULTS / f'{self.label or "match"}.json')
        stats.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(REPO / 'main.py'),
            'play',
            '--no-gui',
            '--n-rounds', str(self.n_rounds),
            '--scenario', self.scenario,
            '--agents', *self.agents,
            '--save-stats', str(stats),
            '--match-name', self.label or 'match']
        if self.seed is not None:
            cmd += ['--seed', str(self.seed)]
        if self.train:
            cmd += ['--train', str(self.train)]
        cmd += self.extra_args
        return cmd, stats


def run(match, timeout=None, verbose=False, retries=0, on_retry=None):
    """Run one match; return its parsed stats dict (or None on failure).

    retries re-runs the match after a *transient* failure only; a genuine
    bug in the agent still fails loudly on the first attempt.  on_retry(attempt)
    may return a replacement :class:Match, which the training driver uses to
    resume from the last checkpoint instead of starting the run over.
    """
    attempt = 0
    while True:
        cmd, stats = match.command()
        if verbose:
            print('  $', ' '.join(cmd[1:]), flush=True)
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=timeout, env=match.environment())
        output = (proc.stderr or '') + (proc.stdout or '')
        print("Output: ", output)

        if proc.returncode == 0 and stats.is_file():
            with open(stats) as fh:
                return json.load(fh)

        if proc.returncode != 0:
            print(f'match {match.label!r} failed (exit {proc.returncode})')
        else:
            print(f'match {match.label!r} produced no stats file')
        return None


def run_all(matches, workers=4, timeout=None, verbose=True, retries=2):
    """Run matches concurrently. order is preserved in the results."""
    if verbose:
        print(f'running {len(matches)} match(es) on {workers} worker(s)')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, m, timeout, False, retries) for m in matches]
        results = []
        for i, fut in enumerate(futures):
            results.append(fut.result())
            if verbose:
                print(f'  [{i + 1}/{len(matches)}] {matches[i].label}', flush=True)
    return results


def get_agent_metrics(stats, agent_name):
    """per-round metrics for one agent, derived from a --save-stats file."""
    by_agent = stats['by_agent'].get(agent_name)
    if by_agent is None:
        return None

    rounds = max(1, by_agent.get('rounds', 1))
    world_steps = sum(r['steps'] for r in stats['by_round'].values())
    agent_steps = by_agent.get('steps', 0)
    bombs = by_agent.get('bombs', 0)

    return {
        'rounds': rounds,
        'score': by_agent.get('score', 0) / rounds,
        'coins': by_agent.get('coins', 0) / rounds,
        'kills': by_agent.get('kills', 0) / rounds,
        'suicides': by_agent.get('suicides', 0) / rounds,
        'crates': by_agent.get('crates', 0) / rounds,
        'bombs': bombs / rounds,
        'steps': agent_steps / rounds,
        'invalid_rate': by_agent.get('invalid', 0) / max(1, agent_steps),
        'crates_per_bomb': by_agent.get('crates', 0) / max(1, bombs),
        'alive_fraction': agent_steps / max(1, world_steps),
        'think_ms': 1000 * by_agent.get('time', 0.0) / max(1, agent_steps),
    }


def world_total_records(stats):
    """
    steps, coins, kills, suicides world totals, so they are only attributable to a single agent in
    solo scenarios.
    """
    rows = []
    for i, (name, rec) in enumerate(sorted(stats['by_round'].items())):
        rows.append({'round': i + 1, 'round_id': name, **rec})
    return rows
