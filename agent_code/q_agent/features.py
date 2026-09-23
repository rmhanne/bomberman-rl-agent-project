"""
Game state -> feature vector.
"""

import numpy as np

from . import gamestate as gs
from . import symmetry

#: (name, width, is_direction_block) in canonical order.
ALL_BLOCKS = (
    # direction blocks
    ('dir_free', 4, True),      
    ('dir_lethal', 4, True),    
    ('dir_safe', 4, True),      
    ('dir_coin', 4, True),      
    ('dir_crate', 4, True),     # first step towards a worthwhile bombing spot
    ('dir_enemy', 4, True),     # first step towards the nearest opponent
    ('dir_escape', 4, True),    # first step towards the nearest lastingly safe tile
    # scalar
    ('in_danger', 1, False),    # current tile is in some bomb's future blast
    ('danger_k', 5, False),     # moves left before the current tile becomes lethal
    ('wait_safe', 1, False),    # standing still keeps a proven escape route
    ('bomb_left', 1, False),    # a bomb is available
    ('bomb_safe', 1, False),    # dropping a bomb here still leaves an escape
    ('crate_gain', 4, False),   # crates a bomb here would destroy
    ('enemy_gain', 3, False),   # opponents a bomb here would catch
    ('coin_dist', 5, False),
    ('crate_dist', 5, False),
    ('enemy_dist', 5, False),
    ('n_coins', 4, False),      # collectable coins currently visible
    ('n_crates', 4, False),     # crates left on the board
    ('dead_end', 1, False),     # current tile has a single free neighbour
    ('bias', 1, False),
)

_BLOCK_BY_NAME = {name: (width, is_dir) for name, width, is_dir in ALL_BLOCKS}

class FeatureSpec:
    """An ordered, named subset of `ALL_BLOCKS` plus its symmetry maps."""

    def __init__(self, blocks):
        unknown = [b for b in blocks if b not in _BLOCK_BY_NAME]
        if unknown:
            raise ValueError(f'unknown feature blocks: {unknown}')
        # keep canonical order regardless of how the config listed them
        chosen = [b for b, _w, _d in ALL_BLOCKS if b in set(blocks)]

        self.names = tuple(chosen)
        self.layout = []
        base = 0
        self.offset = {}
        for name in chosen:
            width, is_dir = _BLOCK_BY_NAME[name]
            self.layout.append((name, base, width, is_dir))
            self.offset[name] = (base, width)
            base += width
        self.size = base

        self.index_perms = symmetry.index_permutations(self.layout, self.size)
        self.action_perms = symmetry.action_permutations(gs.N_ACTIONS)

    def has(self, name):
        return name in self.offset

    def __repr__(self):
        return f'FeatureSpec(size={self.size}, blocks={len(self.names)})'


# The crate layout only changes when a bomb goes off, but the gain map is needed on every step
_GAIN_CACHE = {}


def _crate_gain_cached(field):
    key = field.tobytes()
    cached = _GAIN_CACHE.get(key)
    if cached is None:
        cached = gs.crate_gain_map(field)
        _GAIN_CACHE.clear()   # the layout only ever loses crates; one entry is enough, crates dont actually respawn XD
        _GAIN_CACHE[key] = cached
    return cached


def _bucket(value, edges):
    """Index of the first edge that value does not exceed, else len(edges)."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def extract(spec, game_state, parsed=None):
    """Return (phi, info).

    phi is an int8 vector of length spec.size.  info carries the
    intermediate quantities that reward shaping and the safety mask need, so
    they never have to recompute a BFS.
    """
    p = parsed if parsed is not None else gs.parse(game_state)
    field = p['field']
    x, y = p['pos']
    lethal = p['lethal']
    blocked = p['blocked']
    others = p['others']
    coins = p['coins']

    # REACHABILTIY
    # Tiles we may walk on
    #  - free, not occupied by a bomb or an opponent, and
    #  - not about to be hit by an explosion this very step.
    passable = ~blocked[0] & ~lethal[0]
    passable[x, y] = True
    dist, fmask = gs.bfs(passable, (x, y))

    # TARGETS
    coin_mask = np.zeros(field.shape, dtype=bool)
    for cx, cy in coins:
        coin_mask[cx, cy] = True
    coin_dist, coin_dirs = gs.nearest(dist, fmask, coin_mask)

    gain_map = _crate_gain_cached(field)
    crate_spots = gain_map > 0
    crate_dist, crate_dirs = gs.nearest(dist, fmask, crate_spots)

    # Opponents stand on tiles we cannot enter, so we aim at their neighbors.
    enemy_mask = np.zeros(field.shape, dtype=bool)
    for ox, oy in others:
        for dx, dy in gs.DELTAS:
            nx, ny = ox + dx, oy + dy
            if field[nx, ny] == gs.FREE:
                enemy_mask[nx, ny] = True
    enemy_dist, enemy_dirs = gs.nearest(dist, fmask, enemy_mask)

    # Tiles that stay safe over the whole horizon: where we want to be.
    safe_forever = ~lethal.any(axis=0)
    escape_dist, escape_dirs = gs.nearest(dist, fmask, safe_forever)

    # DANGER LOGIC
    my_danger = lethal[:, x, y] # are we on the lethal mask?
    in_danger = bool(my_danger.any())
    danger_k = int(np.argmax(my_danger)) if in_danger else gs.NEVER

    safe_mask = gs.survivable_moves(field, lethal, (x, y), blocked)

    bombs_left = p['bombs_left']
    if bombs_left:
        lethal_b = gs.with_own_bomb(lethal, field, (x, y))
        blocked_b = blocked.copy()
        blocked_b[:, x, y] = True  # our own bomb: we may leave but not return
        bomb_safe = gs.survivable_moves(field, lethal_b, (x, y), blocked_b) != 0
    else:
        bomb_safe = False

    crate_gain_here = int(gain_map[x, y])
    enemy_gain_here = gs.count_in_blast(
        field, (x, y), _coord_mask(field.shape, others)) if others else 0

    free_neighbours = sum(1 for dx, dy in gs.DELTAS if field[x + dx, y + dy] == gs.FREE)

    # LEGALITY OF ACTIONS
    legal = np.zeros(gs.N_ACTIONS, dtype=bool)
    for a, (dx, dy) in enumerate(gs.DELTAS):
        legal[a] = not blocked[0, x + dx, y + dy]
    legal[4] = True
    legal[5] = bombs_left

    # Assemble options
    phi = np.zeros(spec.size, dtype=np.int8)

    def put_dir(name, mask_or_list):
        if name not in spec.offset:
            return
        base, _ = spec.offset[name]
        if isinstance(mask_or_list, int):
            for a in range(4):
                if mask_or_list & (1 << a):
                    phi[base + a] = 1
        else:
            for a in range(4):
                if mask_or_list[a]:
                    phi[base + a] = 1

    def put_onehot(name, index):
        if name not in spec.offset:
            return
        base, width = spec.offset[name]
        phi[base + min(max(index, 0), width - 1)] = 1

    def put_flag(name, value):
        if name not in spec.offset:
            return
        base, _ = spec.offset[name]
        phi[base] = 1 if value else 0

    put_dir('dir_free', [legal[a] for a in range(4)])
    put_dir('dir_lethal', [lethal[0, x + dx, y + dy] for dx, dy in gs.DELTAS])
    put_dir('dir_safe', safe_mask & 0b1111)
    put_dir('dir_coin', coin_dirs)
    put_dir('dir_crate', crate_dirs)
    put_dir('dir_enemy', enemy_dirs)
    put_dir('dir_escape', escape_dirs)

    put_flag('in_danger', in_danger)
    put_onehot('danger_k', min(danger_k, 4) if in_danger else 4)
    put_flag('wait_safe', safe_mask & (1 << 4))
    put_flag('bomb_left', bombs_left)
    put_flag('bomb_safe', bomb_safe)
    put_onehot('crate_gain', _bucket(crate_gain_here, (0, 1, 2)))
    put_onehot('enemy_gain', _bucket(enemy_gain_here, (0, 1)))
    put_onehot('coin_dist', _bucket(coin_dist, (0, 1, 3, 7)))
    put_onehot('crate_dist', _bucket(crate_dist, (0, 1, 3, 7)))
    put_onehot('enemy_dist', _bucket(enemy_dist, (1, 3, 7, 12)))
    put_onehot('n_coins', _bucket(len(coins), (0, 2, 5)))
    put_onehot('n_crates', _bucket(int((field == gs.CRATE).sum()), (0, 10, 40)))
    put_flag('dead_end', free_neighbours <= 1)
    put_flag('bias', True)

    info = dict(
        pos=(x, y), legal=legal, safe_mask=safe_mask, bomb_safe=bomb_safe,
        in_danger=in_danger, danger_k=danger_k,
        coin_dist=coin_dist, crate_dist=crate_dist, enemy_dist=enemy_dist,
        escape_dist=escape_dist,
        crate_gain=crate_gain_here, enemy_gain=enemy_gain_here,
        n_coins=len(coins), n_crates=int((field == gs.CRATE).sum()),
        bombs_left=bombs_left, dead_end=free_neighbours <= 1,
    )
    return phi, info


def _coord_mask(shape, coords):
    m = np.zeros(shape, dtype=bool)
    for cx, cy in coords:
        m[cx, cy] = True
    return m


def state_key(phi):
    """Hashable tabular key for a feature vector."""
    return phi.tobytes()
