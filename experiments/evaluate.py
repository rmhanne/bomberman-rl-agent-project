"""
Evaluation to run matchups over several seeds and gather metrics.

e.g.
python -m experiments.evaluate --suite stage1
python -m experiments.evaluate --suite stage4 --rounds 200 --seeds 5

Every arm is played for --rounds (or default rounds) rounds under each of --seeds different world seeds.

files are in results/eval/<suite>_raw.csv
one row per (arm, agent, seed) with every metric
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import runner

EVAL_DIR = REPO / 'results' / 'eval'

AGENT = 'q_agent'
SPARRING = 'q_sparring'


# Ours => For when our own agent plays
def ours(config, *opponents, label=None):
    """
    Helper to create own agent configs
    An arm in which our agent plays config.

    opponents are folder names, or (SPARRING, config) for one of our own
    variants.
    """
    # config goes in through env vars
    folders, env = [], {'Q_AGENT_CONFIG': config}
    for entry in opponents:
        if isinstance(entry, tuple):
            folders.append(entry[0])
            env['Q_SPARRING_CONFIG'] = entry[1]
        else:
            folders.append(entry)
    return (label or config, [AGENT] + folders, env)

# Baseline for when a provided agent plays
def baseline(name, *opponents, label=None):
    return (label or name, [name] + list(opponents), {})


# Headline columns for the console table, per suite kind.
#
# alive_fraction is not in solo suits, because the only way to die is to suicide there
# Also agent steps = world steps and the metric is pinned at 1.0 no matter how often the agent blew itself up.

# For solo play suicides and steps carry that information instead; alive_fraction only becomes meaningful once other
# agents keep the round running after a death.

SOLO_COLUMNS = ('score', 'coins', 'suicides', 'steps')
VERSUS_COLUMNS = ('score', 'kills', 'suicides', 'alive_fraction')

RULE_BASED_3 = ['rule_based_agent'] * 3

SUITES = {
    # stage 1 navigation
    'stage1': dict(
        scenario='coin-heaven',
        rounds=100,
        columns=SOLO_COLUMNS,
        arms=[ours('task1'),
              baseline('coin_collector_agent'),
              baseline('rule_based_agent'),  # Reference baselines
              baseline('random_agent'),
              baseline('safe_random_agent')
              ],
    ),

    # stage 2 crates and bombs, alone
    'stage2': dict(
        scenario='loot-crate', rounds=100,
        columns=SOLO_COLUMNS,
        arms=[ours('task2'),
              baseline('coin_collector_agent'),
              baseline('rule_based_agent'),
              baseline('safe_random_agent')
              ],
    ),

    # stage 3 - hunting or killing
    'stage3': dict(
        scenario='classic',
        rounds=100,
        columns=VERSUS_COLUMNS,
        arms=[
            ours('task3', 'peaceful_agent', label='task3 vs peaceful'),
            ours('task3', 'coin_collector_agent', label='task3 vs coin_collector'),
            baseline('rule_based_agent', 'peaceful_agent', label='rule_based vs peaceful'),
            baseline('rule_based_agent', 'coin_collector_agent', label='rule_based vs coin_collector'),
        ],
    ),

    # stage 4 - actual agent
    'stage4': dict(
        scenario='classic', rounds=100, columns=VERSUS_COLUMNS,
        arms=[ours('task4',
                   *RULE_BASED_3,
                   label='task4 vs 3 rule based agents')
              ],
    )
}

METRICS = ['score', 'coins', 'kills', 'suicides', 'crates', 'bombs', 'steps',
           'invalid_rate', 'crates_per_bomb', 'alive_fraction', 'think_ms']


def unique_names(agents):
    """Reproduce BombeRLeWorld.setup_agents naming for duplicated folders."""
    # three rule_based_agents become _0/_1/_2, a single one keeps its plain name.
    # has to match the game exactly or we look up the wrong agent in the stats.
    counts = defaultdict(int)
    names = []
    for folder in agents:
        if agents.count(folder) > 1:
            names.append(f'{folder}_{counts[folder]}')
        else:
            names.append(folder)
        counts[folder] += 1
    return names


def build_games(suite_name, suite, rounds, seeds):
    # one match per (arm, seed) + parallel list of keys
    matches = []
    keys = []
    for arm_label, agents, env in suite['arms']:
        for seed in seeds:
            tx = ''.join(c if c.isalnum() else '_' for c in arm_label).strip('_')

            label = f'{suite_name}__{tx}__seed{seed}'
            matches.append(
                # start match
                runner.Match(
                    agents=list(agents), scenario=suite['scenario'],
                    n_rounds=rounds,
                    seed=seed,
                    label=label, env=dict(env),
                    stats_path=EVAL_DIR / 'raw' / f'{label}.json'
                ))
            keys.append((arm_label, tuple(agents), seed))
    return matches, keys


def collect(keys, results):
    # flatten everything into one long table
    # pos 0 = agent we actually care about.
    rows = []
    for (arm_label, agents, seed), stats in zip(keys, results):
        if stats is None:  # match crashed or timed out, just skip it
            print(f"Skipping match for arm {arm_label}, agents {agents}, seed {seed}.... Game crashed or timed out? Valve, please fix.")
            continue
        names = unique_names(list(agents))
        for position, (folder, name) in enumerate(zip(agents, names)):
            metrics = runner.get_agent_metrics(stats, name)
            if metrics is None:
                continue
            rows.append({
                'arm': arm_label,
                'agent': name,
                'folder': folder,
                'position': position, 'under_test': position == 0,
                'seed': seed,
                **{m: metrics[m] for m in METRICS},
            })
    return rows

def write_csv(path, rows, fields):
    # nothing fancy, we just want something to put in the report or somethign idk XD
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path.relative_to(REPO)} ({len(rows)} rows)')


def main(argv=None):
    # run all the matches of a suite, dump the csvs, print the table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=sorted(SUITES))
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=None)
    args = parser.parse_args(argv)

    suite = SUITES[args.suite]
    rounds = args.rounds or suite['rounds']
    seeds = list(range(1, args.seeds + 1))

    matches, keys = build_games(args.suite, suite, rounds, seeds)
    results = runner.run_all(matches, workers=args.workers, timeout=args.timeout)

    rows = collect(keys, results)
    if not rows:
        print('no results collected')
        return 1

    raw_fields = ['arm', 'agent', 'folder', 'position', 'under_test', 'seed'] + METRICS
    write_csv(EVAL_DIR / f'{args.suite}_raw.csv', rows, raw_fields)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
