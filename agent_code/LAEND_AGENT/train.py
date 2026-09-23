import csv
import pathlib
import pickle
import random
from collections import deque, namedtuple
from typing import List

import numpy as np

import events as e
import settings as s
from .callbacks import (ACTIONS, MODEL_FILE,
                        bomb_kills_crate,
                        bomb_kills_opponent,
                        bfs_dir_dist,
                        can_move,
                        crate_targets,
                        current_epsilon,
                        danger_map,
                        extract_features,
                        safe_tiles,
                        tile_danger_timer)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Hyperparameters
GAMMA                      = 0.9
LEARNING_RATE              = 0.001
L2_REG                     = 1e-5
REPLAY_BUFFER_SIZE         = 40_000
BATCH_SIZE                 = 128
UPDATES_PER_STEP           = 1
SAVE_EVERY_N_ROUNDS        = 50
TD_ERROR_CLIP              = 15.0
TARGET_SYNC_EVERY_N_ROUNDS = 200
LOOP_WINDOW                = 8
LOOP_THRESHOLD             = 2

# Custom shaping events
CLOSER_TO_COIN    = "CLOSER_TO_COIN"
FARTHER_FROM_COIN = "FARTHER_FROM_COIN"
CLOSER_TO_CRATE   = "CLOSER_TO_CRATE"
FARTHER_FROM_CRATE= "FARTHER_FROM_CRATE"
CLOSER_TO_OPP     = "CLOSER_TO_OPP"
FARTHER_FROM_OPP  = "FARTHER_FROM_OPP"
CLOSER_TO_SAFE    = "CLOSER_TO_SAFE"
FARTHER_FROM_SAFE = "FARTHER_FROM_SAFE"
ENTERED_BLAST     = "ENTERED_BLAST"
BOMB_ON_CRATE     = "BOMB_ON_CRATE"
BOMB_ON_OPP       = "BOMB_ON_OPP"
LOOPING           = "LOOPING"
IDLE_WASTE        = "IDLE_WASTE"

GAME_REWARDS = {
    # coins 
    e.COIN_COLLECTED:       25,
    CLOSER_TO_COIN:          1.5,
    FARTHER_FROM_COIN:      -1.5,

    # crates 
    e.CRATE_DESTROYED:       3,
    e.COIN_FOUND:            4,
    CLOSER_TO_CRATE:         0.5,
    FARTHER_FROM_CRATE:     -0.5,

    # opponents 
    e.KILLED_OPPONENT:      60,
    e.OPPONENT_ELIMINATED:   0,
    CLOSER_TO_OPP:           1.5,
    FARTHER_FROM_OPP:       -1.5,
    BOMB_ON_OPP:             4.0,
    BOMB_ON_CRATE:           2.0,

    # survival/escape 
    e.KILLED_SELF:          -80,
    e.GOT_KILLED:           -40,
    e.SURVIVED_ROUND:         3,
    CLOSER_TO_SAFE:           6.0,
    FARTHER_FROM_SAFE:       -6.0,
    ENTERED_BLAST:           -5.0,

    # general 
    e.INVALID_ACTION:        -2,
    e.WAITED:                -0.2,
    e.BOMB_DROPPED:          -0.3,
    LOOPING:                 -2.0,
    IDLE_WASTE:              -1.5,
}

STATS_CSV = pathlib.Path('training_stats.csv')

def setup_training(self):
    self.replay_buffer  = deque(maxlen=REPLAY_BUFFER_SIZE)
    self.episode_reward = 0.0
    self.reward_history = deque(maxlen=200)
    self.score_history  = deque(maxlen=200)
    self.round_count    = getattr(self, 'round_count', 0)
    self.target_model   = self.model.copy()
    self.pos_history    = deque(maxlen=LOOP_WINDOW)
    # per-round stat counters
    self.round_coins    = 0
    self.round_kills    = 0
    self.round_suicides = 0
    self.round_crates   = 0
    self.round_score    = 0


def steps_to_safety(arena, bombs, pos, explosion_map=None):
    dmap = danger_map(arena, bombs)
    if dmap[pos] == np.inf and (explosion_map is None or explosion_map[pos] == 0):
        return 0
    stiles = safe_tiles(arena, dmap, explosion_map)
    if not stiles:
        return np.inf
    free     = (arena == 0)
    frontier = deque([(pos, 0)])
    visited  = {pos}
    while frontier:
        (x, y), dist = frontier.popleft()
        if (x, y) in stiles:
            return dist
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if ((nx, ny) not in visited
                    and 0 <= nx < arena.shape[0]
                    and 0 <= ny < arena.shape[1]
                    and free[nx, ny]):
                visited.add((nx, ny))
                frontier.append(((nx, ny), dist + 1))
    return np.inf


def derive_shaping_events(self, old_state, new_state, action):
    custom = []
    if old_state is None or new_state is None:
        return custom

    arena        = old_state['field']
    _, _, _, old_pos = old_state['self']
    _, _, _, new_pos = new_state['self']
    old_coins    = old_state['coins']
    old_bombs    = old_state['bombs']
    old_bombs_xy = [xy for xy, t in old_bombs]
    old_opps_xy  = [xy for (_, _, _, xy) in old_state['others']]
    old_expmap   = old_state['explosion_map']
    new_expmap   = new_state['explosion_map']

    old_timer     = tile_danger_timer(arena, old_bombs, old_pos)
    was_in_danger = old_timer != np.inf

    new_dmap      = danger_map(new_state['field'], new_state['bombs'])
    now_in_danger = (new_dmap[new_pos] != np.inf
                     or new_expmap[new_pos] > 0)

    free = (arena == 0)
    for bx, by in old_bombs_xy:
        free[bx, by] = False
    for ox, oy in old_opps_xy:
        free[ox, oy] = False

    _, crate_list = crate_targets(arena)
    no_coins      = not old_coins
    no_crates     = not crate_list
    no_opps       = not old_opps_xy
    hunt_mode     = no_coins and no_crates
    collection_mode  = no_opps

    # Escape quality
    if was_in_danger:
        od = steps_to_safety(arena, old_bombs, old_pos, old_expmap)
        nd = steps_to_safety(new_state['field'], new_state['bombs'], new_pos, new_expmap)
        if nd < od:
            custom.append(CLOSER_TO_SAFE)
        elif nd > od:
            custom.append(FARTHER_FROM_SAFE)
        if action == 'BOMB':
            if bomb_kills_opponent(arena, old_opps_xy, old_pos):
                custom.append(BOMB_ON_OPP)
            elif bomb_kills_crate(arena, old_pos):
                custom.append(BOMB_ON_CRATE)
        self.pos_history.append(new_pos)
        return custom
    elif now_in_danger and action != 'BOMB':
        custom.append(ENTERED_BLAST)

    # Progress toward current objective
    if not was_in_danger:
        free_hunt = (arena == 0)
        for bx, by in old_bombs_xy:
            free_hunt[bx, by] = False
        new_bombs_xy = [xy for xy, t in new_state['bombs']]
        free_new_hunt = (new_state['field'] == 0)
        for bx, by in new_bombs_xy:
            free_new_hunt[bx, by] = False

        if collection_mode:
            # No opponents -> focus on collection only
            if old_coins:
                _, od = bfs_dir_dist(free, old_pos, old_coins)
                _, nd = bfs_dir_dist(free, new_pos, old_coins)
                if od is not None and nd is not None:
                    if nd < od:
                        custom.append(CLOSER_TO_COIN)
                    elif nd > od:
                        custom.append(FARTHER_FROM_COIN)
            elif not no_crates:
                ctgts, _ = crate_targets(arena)
                _, od = bfs_dir_dist(free, old_pos, ctgts)
                _, nd = bfs_dir_dist(free, new_pos, ctgts)
                if od is not None and nd is not None:
                    if nd < od:
                        custom.append(CLOSER_TO_CRATE)
                    elif nd > od:
                        custom.append(FARTHER_FROM_CRATE)

        elif hunt_mode and old_opps_xy:
            # No coins and no crates -> chase opponents exclusively
            _, od = bfs_dir_dist(free_hunt, old_pos, old_opps_xy)
            _, nd = bfs_dir_dist(free_new_hunt, new_pos, old_opps_xy)
            if od is not None and nd is not None:
                if nd < od:
                    custom.append(CLOSER_TO_OPP)
                elif nd > od:
                    custom.append(FARTHER_FROM_OPP)

        else:
            # Normal mode: opponents and coins/crates both exist
            opp_in_range = any(
                abs(ox - old_pos[0]) + abs(oy - old_pos[1]) <= s.BOMB_POWER
                for ox, oy in old_opps_xy
            )

            if old_coins:
                _, od = bfs_dir_dist(free, old_pos, old_coins)
                _, nd = bfs_dir_dist(free, new_pos, old_coins)
                if od is not None and nd is not None:
                    if nd < od:
                        custom.append(CLOSER_TO_COIN)
                    elif nd > od:
                        custom.append(FARTHER_FROM_COIN)
                if opp_in_range:
                    _, od = bfs_dir_dist(free_hunt, old_pos, old_opps_xy)
                    _, nd = bfs_dir_dist(free_new_hunt, new_pos, old_opps_xy)
                    if od is not None and nd is not None:
                        if nd < od:
                            custom.append(CLOSER_TO_OPP)
                        elif nd > od:
                            custom.append(FARTHER_FROM_OPP)

            elif not no_crates:
                ctgts, _ = crate_targets(arena)
                _, od = bfs_dir_dist(free, old_pos, ctgts)
                _, nd = bfs_dir_dist(free, new_pos, ctgts)
                if od is not None and nd is not None:
                    if nd < od:
                        custom.append(CLOSER_TO_CRATE)
                    elif nd > od:
                        custom.append(FARTHER_FROM_CRATE)
                if opp_in_range:
                    _, od = bfs_dir_dist(free_hunt, old_pos, old_opps_xy)
                    _, nd = bfs_dir_dist(free_new_hunt, new_pos, old_opps_xy)
                    if od is not None and nd is not None:
                        if nd < od:
                            custom.append(CLOSER_TO_OPP)
                        elif nd > od:
                            custom.append(FARTHER_FROM_OPP)

    # Bomb quality
    if action == 'BOMB':
        if bomb_kills_opponent(arena, old_opps_xy, old_pos):
            custom.append(BOMB_ON_OPP)
        elif bomb_kills_crate(arena, old_pos):
            custom.append(BOMB_ON_CRATE)

    # Anti-stagnation
    objectives_remain = bool(old_coins) or bool(old_opps_xy) or not no_crates
    if objectives_remain and not was_in_danger and can_move(old_state):
        if self.pos_history.count(new_pos) >= LOOP_THRESHOLD:
            custom.append(LOOPING)
        if action == 'WAIT':
            custom.append(IDLE_WASTE)
    self.pos_history.append(new_pos)

    return custom


def store_and_learn(self, old_state, action, new_state, events, done):
    feat_old   = extract_features(old_state)
    feat_new   = extract_features(new_state) if not done else None
    action_idx = ACTIONS.index(action) if action in ACTIONS else ACTIONS.index('WAIT')
    reward     = reward_from_events(self, events)
    self.episode_reward += reward

    if feat_old is not None:
        for feat_s, feat_ns, act_s in augment(feat_old, feat_new, action_idx):
            self.replay_buffer.append(
                Transition(feat_s, act_s, feat_ns, reward, done)
            )

    for _ in range(UPDATES_PER_STEP):
        learn_from_batch(self)



def build_dir_groups():
    from .callbacks import FEATURES, DIR_ORDER
    groups = []
    i = 0
    while i < len(FEATURES) - 3:
        names = FEATURES[i:i+4]
        suffixes = [n.split('_')[-1] for n in names]
        if suffixes == DIR_ORDER:
            groups.append(i)
            i += 4
        else:
            i += 1
    return groups

_DIR_GROUPS = build_dir_groups()

_A_UP    = ACTIONS.index('UP')
_A_RIGHT = ACTIONS.index('RIGHT')
_A_DOWN  = ACTIONS.index('DOWN')
_A_LEFT  = ACTIONS.index('LEFT')
_A_WAIT  = ACTIONS.index('WAIT')
_A_BOMB  = ACTIONS.index('BOMB')

_SYMMETRIES = [
    # 0: identity
    ([0, 1, 2, 3], [_A_UP, _A_RIGHT, _A_DOWN, _A_LEFT, _A_WAIT, _A_BOMB]),
    # 1: flip horizontal (mirror left<->right): RIGHT<->LEFT, UP/DOWN unchanged
    ([0, 3, 2, 1], [_A_UP, _A_LEFT, _A_DOWN, _A_RIGHT, _A_WAIT, _A_BOMB]),
    # 2: flip vertical (mirror up<->down): UP<->DOWN, LEFT/RIGHT unchanged
    ([2, 1, 0, 3], [_A_DOWN, _A_RIGHT, _A_UP, _A_LEFT, _A_WAIT, _A_BOMB]),
    # 3: rotate 180°: UP<->DOWN AND LEFT<->RIGHT
    ([2, 3, 0, 1], [_A_DOWN, _A_LEFT, _A_UP, _A_RIGHT, _A_WAIT, _A_BOMB]),
    # 4: rotate 90° clockwise: UP->RIGHT->DOWN->LEFT->UP
    ([3, 0, 1, 2], [_A_RIGHT, _A_DOWN, _A_LEFT, _A_UP, _A_WAIT, _A_BOMB]),
    # 5: rotate 90° counter-clockwise: UP->LEFT->DOWN->RIGHT->UP
    ([1, 2, 3, 0], [_A_LEFT, _A_UP, _A_RIGHT, _A_DOWN, _A_WAIT, _A_BOMB]),
    # 6: flip along main diagonal (transpose): UP<->LEFT, RIGHT<->DOWN
    ([3, 2, 1, 0], [_A_LEFT, _A_DOWN, _A_RIGHT, _A_UP, _A_WAIT, _A_BOMB]),
    # 7: flip along anti-diagonal: UP<->RIGHT, DOWN<->LEFT
    ([1, 0, 3, 2], [_A_RIGHT, _A_UP, _A_LEFT, _A_DOWN, _A_WAIT, _A_BOMB]),
]


def apply_symmetry(feat, dir_perm):
    f = feat.copy()
    for start in _DIR_GROUPS:
        block = feat[start:start+4].copy()
        for new_i, old_i in enumerate(dir_perm):
            f[start + new_i] = block[old_i]
    return f


def augment(feat_old, feat_new, action_idx):
    for dir_perm, action_perm in _SYMMETRIES:
        f_s  = apply_symmetry(feat_old, dir_perm)
        f_ns = apply_symmetry(feat_new, dir_perm) if feat_new is not None else None
        a_s  = action_perm[action_idx]
        yield f_s, f_ns, a_s


def learn_from_batch(self):
    if len(self.replay_buffer) < max(BATCH_SIZE, 32):
        return
    batch = random.sample(self.replay_buffer, BATCH_SIZE)

    states      = np.stack([t.state  for t in batch])
    actions     = np.array([t.action for t in batch])
    rewards     = np.array([t.reward for t in batch], dtype=np.float32)
    dones       = np.array([t.done   for t in batch])
    next_states = np.stack([
        t.next_state if t.next_state is not None else np.zeros_like(t.state)
        for t in batch
    ])

    q_next     = next_states @ self.target_model.T
    max_q_next = np.max(q_next, axis=1)
    max_q_next[dones] = 0.0
    targets    = rewards + GAMMA * max_q_next
    q_current  = np.sum(
        (states @ self.model.T) * np.eye(len(ACTIONS))[actions], axis=1
    )
    td_error   = np.clip(targets - q_current, -TD_ERROR_CLIP, TD_ERROR_CLIP)

    for a in range(len(ACTIONS)):
        m = actions == a
        if not np.any(m):
            continue
        grad = (td_error[m, None] * states[m]).mean(axis=0)
        self.model[a] += LEARNING_RATE * grad - LEARNING_RATE * L2_REG * self.model[a]


def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events: List[str]):
    self.round_coins    += events.count(e.COIN_COLLECTED)
    self.round_kills    += events.count(e.KILLED_OPPONENT)
    self.round_suicides += events.count(e.KILLED_SELF)
    self.round_crates   += events.count(e.CRATE_DESTROYED)
    _, self.round_score, _, _ = new_game_state['self']

    events = list(events) + derive_shaping_events(
        self, old_game_state, new_game_state, self_action
    )
    self.logger.debug(f"Step {new_game_state['step']}: {events}")
    store_and_learn(self, old_game_state, self_action, new_game_state, events, done=False)


def end_of_round(self, last_game_state, last_action, events: List[str]):
    self.round_coins    += events.count(e.COIN_COLLECTED)
    self.round_kills    += events.count(e.KILLED_OPPONENT)
    self.round_suicides += events.count(e.KILLED_SELF)
    self.round_crates   += events.count(e.CRATE_DESTROYED)
    survived = int(e.SURVIVED_ROUND in events)
    _, final_score, _, _ = last_game_state['self']
    self.round_score = final_score

    events = list(events) + derive_shaping_events(
        self, last_game_state, last_game_state, last_action
    )
    self.logger.debug(f"End of round: {events}")
    store_and_learn(self, last_game_state, last_action, None, events, done=True)

    self.round_count += 1
    self.pos_history.clear()
    self.reward_history.append(self.episode_reward)
    self.score_history.append(self.round_score)
    avg_reward = float(np.mean(self.reward_history))
    avg_score  = float(np.mean(self.score_history))

    self.logger.info(
        f"Round {self.round_count} done.  "
        f"Reward: {self.episode_reward:.1f}  "
        f"(avg {len(self.reward_history)}: {avg_reward:.1f})  "
        f"eps={current_epsilon(self):.3f}"
    )

    write_header = not STATS_CSV.exists()
    with open(STATS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'round', 'score', 'avg_score_200', 'reward', 'avg_reward_200',
                'coins', 'kills', 'suicides', 'crates_destroyed', 'alive_fraction',
            ])
        writer.writerow([
            self.round_count,
            self.round_score,
            round(avg_score, 2),
            round(self.episode_reward, 2),
            round(avg_reward, 2),
            self.round_coins,
            self.round_kills,
            self.round_suicides,
            self.round_crates,
            survived,
        ])

    self.episode_reward = 0.0
    self.round_coins    = 0
    self.round_kills    = 0
    self.round_suicides = 0
    self.round_crates   = 0
    self.round_score    = 0

    if self.round_count % TARGET_SYNC_EVERY_N_ROUNDS == 0:
        self.target_model = self.model.copy()
        self.logger.info(f"Target network synced at round {self.round_count}.")

    if self.round_count % SAVE_EVERY_N_ROUNDS == 0:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"Model saved at round {self.round_count}.")


def reward_from_events(self, events: List[str]) -> float:
    total = sum(GAME_REWARDS.get(ev, 0) for ev in events)
    self.logger.debug(f"  reward {total:.2f} <- {events}")
    return total