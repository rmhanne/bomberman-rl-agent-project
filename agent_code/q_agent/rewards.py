
from . import gamestate as gs

# Custom events I added that I think are good to note. Note, that they do not exist in the framework and aren't emitted by it.
AUX_EVENTS = (
    'SUICIDAL_BOMB', 'UNSAFE_MOVE', 'USELESS_BOMB', 'GOOD_BOMB',
    'ATTACK_BOMB', 'ESCAPED_DANGER', 'LINGERED_IN_DANGER', 'ENTERED_DANGER',
)

# Transitions containing any of these are copied into the rare replay pool, see replay.py. The idea is that these events are rare and should be sampled more often.
IMPORTANT_EVENTS = frozenset({
    'COIN_COLLECTED', 'KILLED_OPPONENT', 'KILLED_SELF', 'GOT_KILLED',
    'CRATE_DESTROYED', 'COIN_FOUND', 'INVALID_ACTION', 'BOMB_DROPPED',
    'SUICIDAL_BOMB', 'UNSAFE_MOVE', 'ATTACK_BOMB',
})

# Gather auxiliary events
def auxiliary_events(action, old_info, new_info, events):
    """Derive our own events from the before/after feature summaries."""
    out = []
    if old_info is None:
        return out

    dropped_bomb = 'BOMB_DROPPED' in events

    if dropped_bomb:
        if not old_info['bomb_safe']:
            out.append('SUICIDAL_BOMB')
        if old_info['crate_gain'] == 0 and old_info['enemy_gain'] == 0:
            out.append('USELESS_BOMB')
        else:
            if old_info['crate_gain'] > 0 and old_info['bomb_safe']:
                out.append('GOOD_BOMB')
            if old_info['enemy_gain'] > 0:
                out.append('ATTACK_BOMB')
    else:
        a = gs.ACTION_INDEX.get(action, 4)
        if a < 5 and not (old_info['safe_mask'] >> a) & 1:
            # the escape search proved this branch fatal
            out.append('UNSAFE_MOVE')

    if new_info is not None:
        if old_info['in_danger'] and not new_info['in_danger']:
            out.append('ESCAPED_DANGER')
        elif old_info['in_danger'] and new_info['in_danger']:
            out.append('LINGERED_IN_DANGER')
        elif not old_info['in_danger'] and new_info['in_danger'] and not dropped_bomb:
            # dropping a bomb necessarily puts us in danger; that is not a fault
            out.append('ENTERED_DANGER')

    return out

def potential(cfg, info):

    if info is None or not cfg['shaping']:
        return 0.0

    if info['n_coins'] > 0 and info['coin_dist'] < gs.NEVER:
        dist = info['coin_dist']
    elif info['crate_dist'] < gs.NEVER:
        dist = info['crate_dist']
    elif info['enemy_dist'] < gs.NEVER:
        dist = info['enemy_dist']
    else:
        return 0.0

    return -cfg['shaping_weight'] * min(dist, cfg['shaping_cap'])


def reward_from_events(cfg, events):
    table = cfg['rewards']
    return float(sum(table.get(ev, 0.0) for ev in events))

def total_reward(cfg, events, old_info, new_info):
    """Event reward + step penalty + potential-based shaping."""
    reward = reward_from_events(cfg, events) + cfg['step_penalty']
    if cfg['shaping']:
        reward += cfg['gamma'] * potential(cfg, new_info) - potential(cfg, old_info)
    return reward


def is_important(events):
    return any(ev in IMPORTANT_EVENTS for ev in events)
