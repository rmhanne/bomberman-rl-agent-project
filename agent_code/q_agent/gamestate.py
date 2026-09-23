"""Board geometry, exact blast/danger timing and BFS utilities.

Time is indexed by the number of moves the agent has already made counting
the move it is about to choose right now as move k = 0.

* A bomb whose state shows timer = t explodes at the end of move k = t

  (environment.update_bombs decrements after the action was performed).

  The resulting Explosion stays in stage 0 for two evaluate_explosions passes, so its blast kills an
  agent standing on it after move t AND after move t + 1.

* A bomb dropped at k = 0 is lethal after moves BOMB_TIMER and BOMB_TIMER + 1.
  Exactly four moves to leave a blast of radius three, so it can just be outrun.

* game_state['explosion_map'][x, y] >= 1 means the tile is lethal after
  move k = 0.
  A value of 0 is already harmless smoke. This matches the explosion_map[d] < 1 test used by the
  provided rule_based_agent.

* A bomb blocks movement onto its tile while it exists, i.e. for moves k <= t.
  Staying on a tile one already occupies is always legal, which is why standing on one's own bomb works.
  """

import numpy as np

try:
    import settings as s

    BOMB_POWER = s.BOMB_POWER
    BOMB_TIMER = s.BOMB_TIMER

except Exception:
    BOMB_POWER = 3
    BOMB_TIMER = 4

WALL, FREE, CRATE = -1, 0, 1

ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
#Movement deltas - DO NOT CHANGE. HAS TO BE THE SAME ORDER as the first four entries of ACTIONS.
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# Horizon that covers every bomb currently on the board plus one we might drop.
KMAX = BOMB_TIMER + 2

# "no danger" / "unreachable".
NEVER = 99


def blast_coords(field, x, y, power=BOMB_POWER):
    """
    Tiles hit by a bomb at (x, y).

    Based off the logic in items.Bomb.get_blast_coords: the blast is stopped by walls
    (-1) only, it passes straight through crates.
    """
    coords = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == WALL:
                break
            coords.append((nx, ny))
    return coords


def blast_mask(field, x, y, power=BOMB_POWER):
    """Boolean-array version of `blast_coords`."""
    m = np.zeros(field.shape, dtype=bool)
    for cx, cy in blast_coords(field, x, y, power):
        m[cx, cy] = True
    return m


# danger over time for ticking bombs and shit
def lethal_schedule(field, bombs, explosion_map, power=BOMB_POWER, kmax=KMAX):
    """lethal[k, x, y]: is standing on (x, y) after move k fatal?"""
    cols, rows = field.shape
    lethal = np.zeros((kmax + 1, cols, rows), dtype=bool)

    if explosion_map is not None:
        lethal[0] |= np.asarray(explosion_map) >= 1

    for (bx, by), t in bombs:
        mask = blast_mask(field, bx, by, power)
        for k in (t, t + 1):
            if 0 <= k <= kmax:
                lethal[k] |= mask

    return lethal


def with_own_bomb(lethal, field, pos, power=BOMB_POWER):
    """Copy of lethal extended by a bomb the agent drops on move k = 0."""
    out = lethal.copy()
    mask = blast_mask(field, pos[0], pos[1], power)
    kmax = out.shape[0] - 1
    for k in (BOMB_TIMER, BOMB_TIMER + 1):
        if k <= kmax:
            out[k] |= mask
    return out


def blocked_schedule(field, bombs, others, kmax=KMAX, block_others_k=1):
    """
    blocked[k, x, y]: may the agent NOT move onto that tile on move k?

    Bombs block while they exist (moves k <= t).

    Technically, opponents are obstacles too, but only for the first block_others_k moves.
    Thing is, though, they move too, so blocking them for the whole horizon makes the escape search needlessly pessimistic.
    """
    cols, rows = field.shape
    blocked = np.zeros((kmax + 1, cols, rows), dtype=bool)
    blocked |= (field != FREE)[None, :, :]

    for (bx, by), t in bombs:
        blocked[: min(t, kmax) + 1, bx, by] = True

    for ox, oy in others:
        blocked[: min(block_others_k, kmax + 1), ox, oy] = True

    return blocked


# vectorised grid search because SIMD go BRRR
def _spread(m):
    """OR of the four one-step shifts of m (no self-term, no wrap-around)."""
    out = np.zeros_like(m)
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def survivable_moves(field, lethal, start, blocked, kmax=KMAX):
    """Which immediate moves keep a guaranteed escape route open?

    Breadth-first search (BFS) over (position, time).
    It returns a bitmask over the five non-bomb actions (bit a <-> ACTIONS[a], bit 4 = WAIT).
    Bit a is set if some sequence of moves starting with a never puts the agent on a lethal tile within the horizon.

    Since the horizon covers every bomb on the board, a set bit ensures the survivability under the
    assumption that no new bombs appear. An unset bit means death along that branch is unavoidable.
    """
    x0, y0 = start
    cols, rows = field.shape

    if not lethal.any():
        # Nothing on the board can kill us, so every legal move survives.
        # Early termination so it saves extra speed on the training time
        mask = 1 << 4
        for a, (dx, dy) in enumerate(DELTAS):
            if not blocked[0, x0 + dx, y0 + dy]:
                mask |= 1 << a
        return mask

    # Move k = 0: Seed each of the five options with its own bit
    level = np.zeros((cols, rows), dtype=np.uint8)
    for a, (dx, dy) in enumerate(DELTAS):
        nx, ny = x0 + dx, y0 + dy
        if not blocked[0, nx, ny]:
            level[nx, ny] |= np.uint8(1 << a)
    level[x0, y0] |= np.uint8(1 << 4)  # WAIT: staying put is always legal
    level[lethal[0]] = 0

    for k in range(1, kmax + 1):
        if not level.any():
            return 0
        moved = _spread(level)
        moved[blocked[k]] = 0
        level = moved | level  # | level: the option of standing still
        level[lethal[k]] = 0

    return int(np.bitwise_or.reduce(level, axis=None))


def bfs(passable, start):
    """BFS from start over passable tiles.

    Returns (dist, first_dir_mask).  dist is -1 where unreachable.
    first_dir_mask[x, y] is a bitmask of the directions that begin some shortest path from start to (x, y).

    Implemented over flat Python lists rather than array ops. As this runs on
    every training step (millions of calls), and at 289 tiles the per-element numpy overhead actually dominated, I think (at least on my machine and tests).
    The list version measured ~2.5x faster. Genuinely think that NumPy has an extra overhead here
    """
    cols, rows = passable.shape
    flat = passable.reshape(-1).tolist()
    n = cols * rows
    dist = [-1] * n
    fdir = [0] * n

    src = start[0] * rows + start[1]
    dist[src] = 0
    queue = [src]
    head = 0
    # index offsets matching DELTAS = UP, RIGHT, DOWN, LEFT
    offsets = (-1, rows, 1, -rows)

    while head < len(queue):
        i = queue[head]
        head += 1
        nd = dist[i] + 1
        parent_mask = fdir[i]
        for a in range(4):
            j = i + offsets[a]
            if not flat[j]:
                continue
            dj = dist[j]
            # neighbours of the source get their own direction bit, everything
            # further away inherits the bits of its predecessor
            bits = parent_mask if nd > 1 else (1 << a)
            if dj < 0:
                dist[j] = nd
                fdir[j] = bits
                queue.append(j)
            elif dj == nd:
                fdir[j] |= bits

    return (np.array(dist, dtype=np.int16).reshape(cols, rows),
            np.array(fdir, dtype=np.uint8).reshape(cols, rows))


def nearest(dist, fmask, targets):
    """Closest reachable target (distance, first_direction_mask).

    targets is a boolean array or a list of coordinates.

    Returns (NEVER, 0) when nothing is reachable.
    """
    if isinstance(targets, np.ndarray) and targets.dtype == bool:
        sel = targets & (dist >= 0)
        if not sel.any():
            return NEVER, 0
        best = int(dist[sel].min())
        hit = sel & (dist == best)
        return best, int(np.bitwise_or.reduce(fmask[hit], axis=None))

    best, mask = NEVER, 0
    for tx, ty in targets:
        d = int(dist[tx, ty])
        if d < 0:
            continue
        if d < best:
            best, mask = d, int(fmask[tx, ty])
        elif d == best:
            mask |= int(fmask[tx, ty])
    return best, mask


def count_in_blast(field, pos, targets_mask, power=BOMB_POWER):
    """How many tiles flagged in targets_mask a bomb at pos would hit."""
    total = 0
    for cx, cy in blast_coords(field, pos[0], pos[1], power):
        if targets_mask[cx, cy]:
            total += 1
    return total


def _shift(a, ox, oy, fill):
    """out[x, y] = a[x + ox, y + oy], padding out-of-board reads with fill."""
    cols, rows = a.shape
    out = np.full_like(a, fill)
    dst_x = slice(max(0, -ox), cols - max(0, ox))
    src_x = slice(max(0, ox), cols - max(0, -ox))
    dst_y = slice(max(0, -oy), rows - max(0, oy))
    src_y = slice(max(0, oy), rows - max(0, -oy))
    out[dst_x, dst_y] = a[src_x, src_y]
    return out


def crate_gain_map(field, power=BOMB_POWER):
    """For every free tile, the number of crates a bomb there would destroy.

    Defines interesting bombing spots as BFS targets. A bomb reaches three
    tiles, so this is deliberately more permissive than "stand next to a crate".

    Each of the four rays is walked once for the whole board,
    carrying an alive mask that switches off as soon as the ray meets a wall.
    """
    crates = field == CRATE
    gain = np.zeros(field.shape, dtype=np.int16)
    if not crates.any():
        return gain

    walls = field == WALL
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        alive = np.ones(field.shape, dtype=bool)
        for i in range(1, power + 1):
            alive &= ~_shift(walls, i * dx, i * dy, True)
            gain += alive & _shift(crates, i * dx, i * dy, False)

    gain[field != FREE] = 0
    return gain


def parse(game_state, block_others_k=1, kmax=KMAX):
    """Bundle everything one decision needs into a plain dict."""
    field = game_state['field']
    _, score, bombs_left, pos = game_state['self']
    bombs = game_state['bombs']
    others = [xy for (_n, _s, _b, xy) in game_state['others']]
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    lethal = lethal_schedule(field, bombs, explosion_map, kmax=kmax)
    blocked = blocked_schedule(field, bombs, others, kmax=kmax,
                               block_others_k=block_others_k)

    return dict(
        field=field, pos=(int(pos[0]), int(pos[1])), score=score,
        bombs_left=bool(bombs_left), bombs=bombs, others=others, coins=coins,
        explosion_map=explosion_map, lethal=lethal, blocked=blocked,
    )
