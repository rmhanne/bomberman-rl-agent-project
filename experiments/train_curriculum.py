"""Training curriculum

python -m experiments.train_curriculum --stage task1 --rounds 3000
python -m experiments.train_curriculum --stage task2 --rounds 10000 --fresh
python -m experiments.train_curriculum --all

Each stage is one main.py play --train 1 run with the scenario and opponents that define the task.
It's kinda just to test the actual workings of the subtasks.
The stage table below is the curriculum, so it doubles as documentation of what each agent was trained against.

--fresh deletes the stage's checkpoint and training log first.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import runner  # noqa: E402

MODEL_DIR = REPO / 'agent_code' / 'q_agent' / 'models'
TRAIN_LOG_DIR = REPO / 'results' / 'train'

AGENT = 'q_agent'
SPARRING = 'q_sparring'

# stage -> (scenario, opponents, default rounds)
# An opponent given as (SPARRING, 'config') is one of our own variants.
STAGES = {
    # stage 1 navigation
    'task1':            ('coin-heaven', [], 3_000),

    # stage 2: crates + bombs but alone (small saddest violin in the world)
    'task2':            ('loot-crate', [], 10_000),

    # stage 3: hunt opponents
    # peaceful_agent is an easy target and allegedly coin_collector_agent a hard one (according to the pdf)

    # note: I train against both, because previous versions simply just otherwise overfitted to one.
    'task3':            ('classic', ['peaceful_agent', 'coin_collector_agent'], 8_000),

    # stage 4: actual agent
    'task4':            ('classic', ['rule_based_agent', 'rule_based_agent', 'rule_based_agent'], 15_000),
}

#: the default order for --all
ORDER = ['task1', 'task2', 'task3', 'task4']


def _split_opponents(opponents):
    """Turn the STAGES opponent list into folder names plus their env overrides."""
    folders, env = [], {}
    for entry in opponents:
        if isinstance(entry, tuple):
            folder, config = entry
            folders.append(folder)
            env['Q_SPARRING_CONFIG'] = config
        else:
            folders.append(entry)
    return folders, env


def stage_files(stage):
    from agent_code.q_agent import config as cfg_module
    cfg = cfg_module.get(stage)
    return MODEL_DIR / cfg['checkpoint'], TRAIN_LOG_DIR / cfg['log_csv']


def rounds_logged(log_csv):
    """How many rounds a (possibly interrupted) run already recorded."""
    if not log_csv.is_file():
        return 0
    with open(log_csv) as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def _make_match(stage, rounds, seed):
    scenario, opponents, _default = STAGES[stage]
    folders, env = _split_opponents(opponents)
    env['Q_AGENT_CONFIG'] = stage
    return runner.Match(
        agents=[AGENT] + folders, scenario=scenario, n_rounds=rounds,
        seed=seed, train=1, label=f'train_{stage}', env=env,
        stats_path=REPO / 'results' / 'train' / f'{stage}_match.json')


def _resume_hook(stage, target, seed, log_csv):
    """
    Build a on_retry callback so that we can continue from a previous checkpoint.

    The reason I added this is cuz i didn't realise my filesystem was transient due to having the code on a nextcloud
    synced folder initially XD
    So I added this and then just kept it even after i found out why writes someitmes failed (when the files were transient)

    checkpoints are written every save_every rounds and the per-round CSV is appended as training proceeds,
    so a run killed by FUCKING NEXTCLOUD SYNCING XD can carry on rather than start over.
    """
    def hook(attempt):
        done = rounds_logged(log_csv)
        remaining = target - done
        print(f'{stage}: {done}/{target} rounds already logged,' f'resuming with {remaining}')
        if remaining <= 0:
            return None
        return _make_match(stage, remaining, seed)

    return hook


def train(stage, rounds=None, fresh=False, seed=None, verbose=True, retries=5):
    if stage not in STAGES:
        raise SystemExit(f'unknown stage {stage!r}; known: {sorted(STAGES)}')
    scenario, opponents, default_rounds = STAGES[stage]
    rounds = rounds or default_rounds

    checkpoint, log_csv = stage_files(stage)
    if fresh:
        for path in (checkpoint, log_csv):
            if path.exists():
                path.unlink()
                print(f'  removed {path.relative_to(REPO)}')

    match = _make_match(stage, rounds, seed)

    print(f'training {stage}: {rounds} rounds of {scenario} '
          f'vs {opponents or "nobody"}')
    started = time.time()
    stats = runner.run(match, verbose=verbose, retries=retries,
                       on_retry=_resume_hook(stage, rounds, seed, log_csv))
    elapsed = time.time() - started

    if stats is None:
        print(f'FAILED after {elapsed:.0f}s')
        return None

    metrics = runner.get_agent_metrics(stats, AGENT)
    print(f'done in {elapsed / 60:.1f} min '
          f'({rounds / max(elapsed, 1e-9):.1f} rounds/s)')
    if metrics:
        print(f'training-time averages: score {metrics["score"]:.2f}, '
              f'coins {metrics["coins"]:.2f}, suicides {metrics["suicides"]:.3f}, '
              f'alive {metrics["alive_fraction"]:.2f}')
    print(f'checkpoint: {checkpoint.relative_to(REPO)}')
    return metrics


def train_parallel(stages, rounds=None, fresh=False, seed=None, workers=4,
                   retries=5):
    """Train several arms at once."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for stage in stages:
        if stage not in STAGES:
            raise SystemExit(f'unknown stage {stage!r}')
        _scenario, _opponents, default_rounds = STAGES[stage]
        target = rounds or default_rounds
        checkpoint, log_csv = stage_files(stage)
        if fresh:
            for path in (checkpoint, log_csv):
                if path.exists():
                    path.unlink()
        jobs.append((stage, _make_match(stage, target, seed),
                     _resume_hook(stage, target, seed, log_csv)))

    print(f'training {len(jobs)} arm(s) on {workers} worker(s): 'f'{", ".join(stages)}')
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(runner.run, match, None, False, retries, hook): stage
                   for stage, match, hook in jobs}
        for future, stage in futures.items():
            stats = future.result()
            if stats is None:
                print(f'{stage}: FAILED')
                continue
            metrics = runner.get_agent_metrics(stats, AGENT)
            print(f'{stage}: score {metrics["score"]:.2f}  '
                  f'coins {metrics["coins"]:.2f}  '
                  f'suicides {metrics["suicides"]:.3f}  '
                  f'(training-time averages, includes the exploration phase)'
                  )
    print(f'all parallel stuff done in {(time.time() - started) / 60:.1f} min')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', help='stage name, see STAGES')
    parser.add_argument('--stages', nargs='+',
                        help='train multiple in parallel')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--all', action='store_true',
                        help=f'run all training curriculum: {" -> ".join(ORDER)}')

    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--fresh', action='store_true',
                        help='delete the checkpoint and log before training')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)

    if args.list:
        print(f'{"stage":22s} {"scenario":12s} {"rounds":>7s}  opponents')
        for name, (scenario, opponents, rounds) in STAGES.items():
            shown = [o[0] + ':' + o[1] if isinstance(o, tuple) else o
                     for o in opponents]
            print(f'{name:22s} {scenario:12s} {rounds:7d}  '
                  f'{", ".join(shown) or "-"}')
        return 0

    if args.stages:
        train_parallel(args.stages, rounds=args.rounds, fresh=args.fresh,
                       seed=args.seed, workers=args.workers)
        return 0

    stages = ORDER if args.all else ([args.stage] if args.stage else [])
    if not stages:
        parser.error('give --stage NAME, --stages A B C, --all, or --list')

    for stage in stages:
        train(stage, rounds=args.rounds, fresh=args.fresh, seed=args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
