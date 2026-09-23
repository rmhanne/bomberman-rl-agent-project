"""Named agent configurations

Every agent variant in this project is the same code with a different config really.
"""

import copy

from ..q_agent.features import ALL_BLOCKS

# Full feature set, used from stage 2 onwards so checkpoints stay compatible
FULL_BLOCKS = tuple(name for name, _w, _d in ALL_BLOCKS)

BASE = dict(
    # representation
    feature_blocks=FULL_BLOCKS,

    # learning
    gamma=0.95,
    n_step=3,
    lr=0.02,
    lr_decay=0.0,                # tabular: alpha = lr / (1 + lr_decay * visits)
    batch_size=128,
    updates_per_step=1,
    grad_clip=5.0,
    memory=100_000,
    rare_fraction=0.25,          # share of each batch drawn from the rare pool
    target_sync=1_000,           # gradient steps between target-network syncs
    symmetry_augment=True,       # 8x D4 data augmentation

    # action space
    # In stage 1 we disable bombs here cuz otherwise we quickly kill ourselves XD
    # we don't have all features enabled on the first stage
    enable_bomb=True,

    # exploration
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_rounds=1_500,      # time constant of the exponential decay
    eps_greedy_bias=True,        # explore over legal actions only
    safe_explore=0.9,            # share of exploration restricted to safe actions

    # reward shaping
    shaping=True,
    shaping_weight=0.15,         # weight of the potential Phi = -w * distance
    shaping_cap=15,              # distances are clipped here
    step_penalty=-0.05,
    rewards={},                  # merged into DEFAULT_REWARDS

    # loop breaking
    loop_penalty=0.34,           # fraction of the Q-spread charged per recent visit
    loop_memory=8,               # how many past positions count as "recent"
    safety_mask=False,           # veto provably fatal actions at decision time
    eval_eps=0.0,                # exploration outside training mode
    greedy_tie_break='random',


    save_every=200,              # rounds between checkpoint writes

    checkpoint='task1.pt',
    init_from=None,              # previous checkpoint to carry over
    train_scenario='coin-heaven',
    log_csv=None,                # defaults to '<checkpoint stem>_train.csv'
)


# Real game rewards:
# Coin: 1
# Kill: 5
DEFAULT_REWARDS = {
    # genuine game events
    'COIN_COLLECTED': 17.0,
    'KILLED_OPPONENT': 25.0,
    'KILLED_SELF': -150.0,
    'GOT_KILLED': -100.0,
    'CRATE_DESTROYED': 1.3,
    'COIN_FOUND': 0.7,
    'SURVIVED_ROUND': 3.0,
    'OPPONENT_ELIMINATED': 15.0,
    'INVALID_ACTION': -1.0,

    'WAITED': -0.2,
    'IN_LEAD': 0.1,
    'IN_CORNER': -1,

    # aux events raised by rewards.py, not present in the actual game tho
    'SUICIDAL_BOMB': -30.0,       # dropped a bomb with no escape route
    'UNSAFE_MOVE': -1.5,         # stepped somewhere with no escape route
    'USELESS_BOMB': -0.6,        # bomb that can hit neither crate nor opponent
    'GOOD_BOMB': 1.5,            # bomb next to crates, escape available
    'ATTACK_BOMB': 2.0,          # bomb that can catch an opponent
    'ESCAPED_DANGER': 0.8,
    'LINGERED_IN_DANGER': -0.5,
    'ENTERED_DANGER': -0.4,
}


def _cfg(**overrides):
    cfg = copy.deepcopy(BASE)
    rewards = copy.deepcopy(DEFAULT_REWARDS)
    rewards.update(overrides.pop('rewards', {}))
    cfg.update(overrides)
    cfg['rewards'] = rewards
    return cfg

CONFIGS = {
    # same task, richer representation -- seeds the stage-2 warm start
    'task1': _cfg(
        feature_blocks=FULL_BLOCKS,
        lr=0.02,
        eps_decay_rounds=250,
        eps_end=0.02,
        checkpoint='task1.pt',
        train_scenario='coin-heaven',
        rewards={'COIN_COLLECTED': 3.0, 'INVALID_ACTION': -1.0, 'WAITED': -0.5},
    ),
    # stage 2 -- crates, bombs, no opponents.
    'task2': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        lr=0.02,
        eps_decay_rounds=1_000,
        eps_end=0.05,
        checkpoint='task2.pt',
        init_from='task1.pt',
        train_scenario='loot-crate',
    ),
    # stage 3 hunt weak opponents.
    'task3': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        lr=0.015,
        eps_decay_rounds=2_500,
        eps_end=0.05,
        checkpoint='task3.pt',
        init_from=['task2.pt', 'task1.pt'],
        train_scenario='classic',
    ),

    # stage 4 full game against rule_based_agent
    'task4': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        lr=0.01,
        eps_decay_rounds=3_000,
        eps_end=0.03,
        checkpoint='task4.pt',
        init_from=['task3.pt', 'task2.pt', 'task1.pt'],
        train_scenario='classic',
    ),

    'tournament': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        safety_mask=True,
        eps_end=0.0,
        checkpoint='task4.pt',
        train_scenario='classic',
    )
}


def get(name):
    """Return a deep copy of the named configuration."""
    if name not in CONFIGS:
        raise KeyError(f'unknown config {name!r}; known: {sorted(CONFIGS)}')
    cfg = copy.deepcopy(CONFIGS[name])
    cfg['name'] = name
    if cfg.get('log_csv') is None:
        cfg['log_csv'] = cfg['checkpoint'].replace('.pt', '') + '_train.csv'
    return cfg
