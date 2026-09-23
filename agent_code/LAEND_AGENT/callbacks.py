import os
import pickle
from collections import deque
import numpy as np
import settings as s


ACTIONS   = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRS      = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
DIR_ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT']

FEATURES = (
    [f'can_move_{d}'        for d in DIR_ORDER] +   # tile is walkable
    [f'escape_ok_{d}'       for d in DIR_ORDER] +   # escape route in dir d
    ['in_danger'] +
    ['danger_urgency'] +                           
    ['danger_steps'] +                              # how many consecutive steps we're in danger 
    [f'coin_dir_{d}'        for d in DIR_ORDER] +   # BFS dir to nearest coin
    ['coin_dist'] +                                   
    ['coin_visible'] +                                
    [f'crate_dir_{d}'       for d in DIR_ORDER] +   # BFS dir to bombable crate spot
    ['adj_crates'] +                                  
    [f'safe_dir_{d}'        for d in DIR_ORDER] +   # BFS dir to nearest safe tile
    ['can_bomb'] +                                    
    ['bomb_safe'] +                                  
    ['bomb_hits_crate'] +                             
    ['bomb_hits_opp'] +                               
    [f'opp_dir_{d}'         for d in DIR_ORDER] + 
    ['opp_dist'] +                                    
    ['num_opps'] +                                   
    ['hunt_mode'] +                                 # scenario: no coins and no crates remain (only kills score now)
    ['collection_mode'] +                           # scenario: no opponents remain (focus purely on collection)
    ['bias']                                         
)
FEATURE_SIZE = len(FEATURES)

MODEL_FILE         = "my-saved-model.pt"
# Epsilon-greedy exploration
EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY_ROUNDS   = 3000
OSCILLATION_WINDOW = 8   # we remember the N last visited tiles to avoid oscillating behaviour (e.g. only moving back and forth or running in a loop)
STAGNATION_WINDOW  = 12  # number of steps with no objective-distance improvement before forcing a new direction


def blast_tiles(bomb_pos, arena):
    x, y  = bomb_pos
    tiles = [(x, y)]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if arena[nx, ny] == -1:
                break
            tiles.append((nx, ny))
    return tiles

# track all bombs and their timers 
def danger_map(arena, bombs):
    d = np.full(arena.shape, np.inf)
    for (bx, by), t in bombs:
        for (x, y) in blast_tiles((bx, by), arena):
            d[x, y] = min(d[x, y], t)
    return d

# check when pos will be hit by a bomb (if any)
def tile_danger_timer(arena, bombs, pos):
    timer = np.inf
    for (bx, by), t in bombs:
        if pos in blast_tiles((bx, by), arena):
            timer = min(timer, t)
    return timer


def walkable(arena, explosion_map, bombs_xy, others_xy, x, y):
    return (arena[x, y] == 0
            and explosion_map[x, y] == 0
            and (x, y) not in bombs_xy
            and (x, y) not in others_xy)


def safe_tiles(arena, dmap, explosion_map=None):
    result = []
    for x in range(arena.shape[0]):
        for y in range(arena.shape[1]):
            if arena[x, y] != 0:
                continue
            if dmap[x, y] != np.inf:
                continue
            if explosion_map is not None and explosion_map[x, y] > 0:
                continue
            result.append((x, y))
    return result

def trace_dist(parent, node, start):
    d = 0
    while node != start:
        node = parent[node]
        d += 1
    return d


def bfs_dir_dist(free_space, start, targets):
    targets = set(targets)
    if not targets:
        return None, None
    if start in targets:
        return 'HERE', 0
    frontier = deque([start])
    parent   = {start: None}
    while frontier:
        cur = frontier.popleft()
        if cur in targets:
            dist = trace_dist(parent, cur, start)
            node = cur
            while parent[node] != start:
                node = parent[node]
            dx, dy = node[0] - start[0], node[1] - start[1]
            for d, (ddx, ddy) in DIRS.items():
                if (ddx, ddy) == (dx, dy):
                    return d, dist
        x, y = cur
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if (0 <= nx < free_space.shape[0]
                    and 0 <= ny < free_space.shape[1]
                    and free_space[nx, ny]
                    and (nx, ny) not in parent):
                parent[(nx, ny)] = cur
                frontier.append((nx, ny))
    return None, None


def dir_onehot(direction):
    vec = [0.0] * 4
    if direction in DIR_ORDER:
        vec[DIR_ORDER.index(direction)] = 1.0
    return vec

# map showing which tiles are free -> block tiles occupied by opponents and tiles currently exploding
def walkable_map(arena, others_xy, explosion_map=None):
    free = (arena == 0)
    for ox, oy in others_xy:
        free[ox, oy] = False
    if explosion_map is not None:
        for x in range(arena.shape[0]):
            for y in range(arena.shape[1]):
                if explosion_map[x, y] > 0:
                    free[x, y] = False
    return free

# check if we can reach a safe tile from first_step within budget steps, taking into account all bombs and currently-exploding tiles
def escape_reachable(arena, bombs, others_xy, origin, first_step, budget, explosion_map=None):
    dmap = danger_map(arena, bombs)
    free = walkable_map(arena, others_xy, explosion_map)
    frontier = deque([(first_step, 1)])
    visited  = {origin, first_step}
    while frontier:
        (x, y), dist = frontier.popleft()
        if dmap[x, y] == np.inf and (explosion_map is None or explosion_map[x, y] == 0):
            return True
        if dist >= budget:
            continue
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if ((nx, ny) not in visited
                    and 0 <= nx < arena.shape[0]
                    and 0 <= ny < arena.shape[1]
                    and free[nx, ny]):
                visited.add((nx, ny))
                frontier.append(((nx, ny), dist + 1))
    return False


def bomb_kills_crate(arena, pos):
    for (x, y) in blast_tiles(pos, arena)[1:]:
        if arena[x, y] == 1:
            return True
    return False
 
 
def bomb_kills_opponent(arena, others_xy, pos):
    for (x, y) in blast_tiles(pos, arena)[1:]:
        if (x, y) in others_xy:
            return True
    return False
 
 
def bomb_is_useful(arena, others_xy, pos):
    return bomb_kills_crate(arena, pos) or bomb_kills_opponent(arena, others_xy, pos)


def bomb_has_escape(arena, bombs, others_xy, pos, explosion_map=None):
    hypo = list(bombs) + [(pos, s.BOMB_TIMER)]
    dmap = danger_map(arena, hypo)
    free = walkable_map(arena, others_xy, explosion_map)
    frontier = deque([(pos, 0)])
    visited  = {pos}
    while frontier:
        (x, y), dist = frontier.popleft()
        if dist > 0 and dmap[x, y] == np.inf and (explosion_map is None or explosion_map[x, y] == 0):
            return True
        if dist >= s.BOMB_TIMER - 1:
            continue
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if ((nx, ny) not in visited
                    and 0 <= nx < arena.shape[0]
                    and 0 <= ny < arena.shape[1]
                    and free[nx, ny]):
                visited.add((nx, ny))
                frontier.append(((nx, ny), dist + 1))
    return False


# find all crates and their adjacent free tiles (-> bombing positions). If no adjacent free tiles exist, return the crate positions themselves.
def crate_targets(arena):
    cols   = range(1, arena.shape[0] - 1)
    rows   = range(1, arena.shape[1] - 1)
    crates = [(cx, cy) for cx in cols for cy in rows if arena[cx, cy] == 1]
    adj    = set()
    for cx, cy in crates:
        for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
            if (0 <= nx < arena.shape[0]
                    and 0 <= ny < arena.shape[1]
                    and arena[nx, ny] == 0):
                adj.add((nx, ny))
    return list(adj) if adj else crates, crates


def extract_features(game_state):
    if game_state is None:
        return None

    arena         = game_state['field']
    _, _, has_bomb, (x, y) = game_state['self']
    bombs         = game_state['bombs']
    bombs_xy      = [xy for xy, t in bombs]
    others        = game_state['others']
    others_xy     = [xy for (_, _, _, xy) in others]
    coins         = game_state['coins']
    explosion_map = game_state['explosion_map']
    dmap          = danger_map(arena, bombs)

    feats = {}

    # where can we move legally
    for d, (dx, dy) in DIRS.items():
        feats[f'can_move_{d}'] = float(
            walkable(arena, explosion_map, bombs_xy, others_xy, x + dx, y + dy)
        )

    # escape route per direction
    timer = tile_danger_timer(arena, bombs, (x, y))
    budget    = int(timer) if timer != np.inf else s.BOMB_TIMER
    in_danger = timer != np.inf
    for d, (dx, dy) in DIRS.items():
        nx, ny = x + dx, y + dy
        if not feats[f'can_move_{d}']:
            feats[f'escape_ok_{d}'] = 0.0
        elif not in_danger:
            feats[f'escape_ok_{d}'] = 1.0
        else:
            feats[f'escape_ok_{d}'] = float(
                escape_reachable(arena, bombs, others_xy, (x, y), (nx, ny),
                                 budget, explosion_map)
            )

    feats['in_danger']      = float(in_danger)
    feats['danger_urgency'] = 0.0 if not in_danger else (1.0 - min(timer, s.BOMB_TIMER) / s.BOMB_TIMER)
    # danger_steps: consecutive steps in danger -> tells the model how urgent escape is
    danger_step_count = game_state.get('_danger_step_count', 0)
    feats['danger_steps'] = min(danger_step_count, s.BOMB_TIMER) / s.BOMB_TIMER

    free = (arena == 0)
    for bx, by in bombs_xy:
        free[bx, by] = False
    for ox, oy in others_xy:
        free[ox, oy] = False

    # coins
    coin_dir, coin_dist = bfs_dir_dist(free, (x, y), coins)
    for d, v in zip(DIR_ORDER, dir_onehot(coin_dir)):
        feats[f'coin_dir_{d}'] = v
    feats['coin_dist']    = 0.0 if coin_dist is None else 1.0 / (1.0 + coin_dist)
    feats['coin_visible'] = float(bool(coins))

    # crates 
    crate_tgts, crate_list = crate_targets(arena)
    crate_dir, _           = bfs_dir_dist(free, (x, y), crate_tgts)
    for d, v in zip(DIR_ORDER, dir_onehot(crate_dir)):
        feats[f'crate_dir_{d}'] = v
    feats['adj_crates'] = sum(
        1 for dx, dy in DIRS.values() if arena[x + dx, y + dy] == 1
    ) / 4.0

    # safe direction 
    if in_danger:
        stiles    = safe_tiles(arena, dmap, explosion_map)
        safe_dir, _ = bfs_dir_dist(free, (x, y), stiles)
    else:
        safe_dir = None
    for d, v in zip(DIR_ORDER, dir_onehot(safe_dir)):
        feats[f'safe_dir_{d}'] = v

    # bombing 
    feats['can_bomb']        = float(bool(has_bomb))
    feats['bomb_safe']       = float(bool(has_bomb) and bomb_has_escape(
        arena, bombs, others_xy, (x, y), explosion_map))
    feats['bomb_hits_crate'] = float(bool(has_bomb) and bomb_kills_crate(arena, (x, y)))
    feats['bomb_hits_opp']   = float(bool(has_bomb) and bomb_kills_opponent(arena, others_xy, (x, y)))

    # opponents 
    opp_dir, opp_dist = bfs_dir_dist(free, (x, y), others_xy)
    for d, v in zip(DIR_ORDER, dir_onehot(opp_dir)):
        feats[f'opp_dir_{d}'] = v
    feats['opp_dist']  = 0.0 if opp_dist is None else 1.0 / (1.0 + opp_dist)
    feats['num_opps']  = len(others_xy) / 3.0

    # mode flags 
    feats['hunt_mode']    = float(not coins and not crate_list)
    feats['collection_mode'] = float(not others_xy)

    feats['bias'] = 1.0

    return np.array([feats[name] for name in FEATURES], dtype=np.float32)


def any_objective_left(game_state):
    if game_state['coins'] or game_state['others']:
        return True
    return bool(np.any(game_state['field'] == 1))


def can_move(game_state):
    arena         = game_state['field']
    _, _, _, (x, y) = game_state['self']
    bombs_xy      = [xy for xy, t in game_state['bombs']]
    others_xy     = [xy for (_, _, _, xy) in game_state['others']]
    explosion_map = game_state['explosion_map']
    return any(
        walkable(arena, explosion_map, bombs_xy, others_xy, x + dx, y + dy)
        for dx, dy in DIRS.values()
    )


# block unsafe or wasteful actions regardless of Q-values
def compute_mask(game_state, recent_positions=None, stagnating=False):
    """Hard-block unsafe or wasteful actions regardless of Q-values.
    stagnating=True also blocks the last-visited tile to force a new direction."""
    arena         = game_state['field']
    _, _, has_bomb, (x, y) = game_state['self']
    bombs         = game_state['bombs']
    bombs_xy      = [xy for xy, t in bombs]
    others_xy     = [xy for (_, _, _, xy) in game_state['others']]
    explosion_map = game_state['explosion_map']
    dmap          = danger_map(arena, bombs)
    in_danger     = dmap[x, y] != np.inf
    timer     = tile_danger_timer(arena, bombs, (x, y))
    budget        = int(timer) if timer != np.inf else s.BOMB_TIMER
 
    mask = np.zeros(len(ACTIONS), dtype=bool)
 
    for idx, action in enumerate(ACTIONS):
        if action in DIRS:
            dx, dy = DIRS[action]
            nx, ny = x + dx, y + dy
            ok = walkable(arena, explosion_map, bombs_xy, others_xy, nx, ny)
            if ok and dmap[nx, ny] <= 1:
                ok = dmap[x, y] <= dmap[nx, ny]
            if ok and in_danger:
                ok = escape_reachable(arena, bombs, others_xy, (x, y), (nx, ny),
                                      budget, explosion_map)
            mask[idx] = ok
 
        elif action == 'WAIT':
            mask[idx] = not in_danger
 
        elif action == 'BOMB':
            mask[idx] = (
                bool(has_bomb)
                and bomb_is_useful(arena, others_xy, (x, y))
                and bomb_has_escape(arena, bombs, others_xy, (x, y), explosion_map)
            )
 
    if not in_danger and any_objective_left(game_state):
        movement_ok = mask[:4].copy()
        if np.any(movement_ok):
            mask[ACTIONS.index('WAIT')] = False
 
            blocked = set(recent_positions) if recent_positions else set()
            if stagnating and recent_positions:
                blocked.add(recent_positions[-1])
 
            if blocked:
                fresh = movement_ok.copy()
                for d, (dx, dy) in DIRS.items():
                    action_idx = ACTIONS.index(d)
                    if fresh[action_idx] and (x + dx, y + dy) in blocked:
                        fresh[action_idx] = False
                if np.any(fresh):
                    mask[:4] = fresh
 
    if not np.any(mask):
        mask[ACTIONS.index('WAIT')] = True
    return mask


def setup(self):
    self.round_count        = 0
    self.recent_positions   = deque(maxlen=OSCILLATION_WINDOW)
    self.dist_history       = deque(maxlen=STAGNATION_WINDOW)
    self.danger_step_count  = 0 

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
        if self.model.shape != (len(ACTIONS), FEATURE_SIZE):
            self.logger.warning("Model shape mismatch -- reinitialising.")
            self.model = np.zeros((len(ACTIONS), FEATURE_SIZE), dtype=np.float32)
    else:
        self.logger.info("No saved model found, starting from scratch.")
        self.model = np.zeros((len(ACTIONS), FEATURE_SIZE), dtype=np.float32)
        self.model[:4, -1] = 0.1


def current_epsilon(self):
    frac = min(self.round_count / EPS_DECAY_ROUNDS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


# BFS distance to the nearest coin, opponent, or crate-bombing spot
def _nearest_objective_dist(game_state):
    """BFS distance to the nearest coin, opponent, or crate-bombing spot.
    Used to detect stagnation (no progress for STAGNATION_WINDOW steps)."""
    arena     = game_state['field']
    _, _, _, (x, y) = game_state['self']
    bombs_xy  = [xy for xy, t in game_state['bombs']]
    others_xy = [xy for (_, _, _, xy) in game_state['others']]
    coins     = game_state['coins']

    free = (arena == 0)
    for bx, by in bombs_xy:
        free[bx, by] = False
    for ox, oy in others_xy:
        free[ox, oy] = False

    if coins:
        _, d = bfs_dir_dist(free, (x, y), coins)
        return d
    ctgts, crate_list = crate_targets(arena)
    if crate_list:
        _, d = bfs_dir_dist(free, (x, y), ctgts)
        return d
    if others_xy:
        free_hunt = (arena == 0)
        for bx, by in bombs_xy:
            free_hunt[bx, by] = False
        _, d = bfs_dir_dist(free_hunt, (x, y), others_xy)
        return d
    return None


def act(self, game_state):
    if game_state['step'] == 1:
        self.recent_positions  = deque(maxlen=OSCILLATION_WINDOW)
        self.dist_history      = deque(maxlen=STAGNATION_WINDOW)
        self.danger_step_count = 0

    arena    = game_state['field']
    bombs    = game_state['bombs']
    _, _, _, pos = game_state['self']
    if tile_danger_timer(arena, bombs, pos) != np.inf:
        self.danger_step_count += 1
    else:
        self.danger_step_count = 0

    gs = dict(game_state)
    gs['_danger_step_count'] = self.danger_step_count

    d = _nearest_objective_dist(game_state)
    self.dist_history.append(d)
    stagnating = (
        len(self.dist_history) == STAGNATION_WINDOW
        and all(x is not None for x in self.dist_history)
        and min(self.dist_history) >= self.dist_history[0]
    )

    features = extract_features(gs)
    mask     = compute_mask(game_state, self.recent_positions, stagnating)
    q_vals   = np.where(mask, self.model @ features, -np.inf)

    if self.train and np.random.rand() < current_epsilon(self):
        choices = np.flatnonzero(mask)
        idx     = np.random.choice(choices) if len(choices) else ACTIONS.index('WAIT')
        chosen  = ACTIONS[idx]
        self.logger.debug(f"Exploring: {chosen}  eps={current_epsilon(self):.3f}")
    elif np.any(np.isfinite(q_vals)):
        idx    = int(np.argmax(q_vals))
        chosen = ACTIONS[idx]
        self.logger.debug(f"Exploiting: {chosen}  Q={q_vals[idx]:.3f}")
    else:
        chosen = 'WAIT'

    self.recent_positions.append(game_state['self'][3])
    return chosen