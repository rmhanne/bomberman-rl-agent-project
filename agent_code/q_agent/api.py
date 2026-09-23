"""
The heart of the agent really.
Decision-making and learning loop.

Callback timing (see environment.BombeRLeWorld)

act(s_t) runs before the world advances
game_events_occurred(s_t, a_t, s_{t+1}, events) runs at the end of the same step.

When the agent dies,game_events_occurred is skipped and end_of_round delivers the terminal transition instead.

When the agent survives to the end of the round, the last step arrives through BOTH callbacks
we detect that by step number and only add the late SURVIVED_ROUND bonus rather than duplicating the transition.
"""

import atexit
import csv
import logging
import math
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np

from . import config as cfg_module
from . import features as feat
from . import gamestate as gs
from . import models as model_module
from . import replay as replay_module
from . import rewards as rewards_module

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent / 'models'
TRAIN_LOG_DIR = REPO_ROOT / 'results' / 'train'

# Per-step decision tracing is off by default because I made my training take quite a bit longer.
# Set BOMBERMAN_VERBOSE=1 to get the trace back when debugging a policy.
VERBOSE = os.environ.get('BOMBERMAN_VERBOSE', '') not in ('', '0', 'false')


def resolve_config(env_var, default):

    name = os.environ.get(env_var, '').strip()
    return name or default
# callbacks.py entry points
def setup(self, config_name):

    self.cfg = cfg_module.get(config_name)
    self.logger.info(f"Using config: {self.cfg}")

    self.spec = feat.FeatureSpec(self.cfg['feature_blocks'])
    self.model = model_module.build(self.cfg, self.spec, gs.N_ACTIONS)
    self.rng = np.random.default_rng()

    self.checkpoint_path = MODEL_DIR / self.cfg['checkpoint']
    loaded = model_module.load_checkpoint(self.model, self.checkpoint_path,
                                          self.logger)

    # Check if we pull over something from a previous stage learned
    # list is tried in order, so a stage can fall back to an earlier
    # checkpoint when its immediate predecessor has not been trained yet.
    if not loaded and self.cfg['init_from']:
        self.logger.info(f"Checkpoint not found at {self.checkpoint_path}, trying curriculum warm start")
        candidates = self.cfg['init_from']
        if isinstance(candidates, str):
            candidates = [candidates]
        for candidate in candidates:
            if model_module.load_checkpoint(self.model, MODEL_DIR / candidate,
                                            self.logger):
                break

    if not VERBOSE:
        # settings.py enables DEBUG for agent code, which writes a line per step
        # into agent_code/<name>/logs/.  Besides costing time, that constant
        # churn inside the agent folder is what makes a sync client grab the
        # directory handle that the framework's os.chdir then trips over.
        self.logger.setLevel(logging.WARNING)

    self.enabled_actions = [a for a in range(gs.N_ACTIONS)
                            if self.cfg['enable_bomb'] or a != gs.ACTION_INDEX['BOMB']]
    self.round_index = 0
    self._feat_cache = (None, None, None, None)  # round, step, phi, info
    self.recent_positions = deque(maxlen=max(1, self.cfg['loop_memory']))
    self._seen_round = None
    self.logger.info(f'q_agent ready: config={config_name} spec={self.spec} '
                     f'model={self.model.kind}')

def act(self, game_state):
    if game_state['round'] != self._seen_round:
        # end_of_round only fires while training seemingly, so we reset per-round memory here
        self._seen_round = game_state['round']
        self.recent_positions.clear()

    phi, info = _extract_features(self, game_state)
    q = self.model.q_values(phi)

    allowed = _allowed_actions(self, info)
    epsilon = _epsilon(self)


    # Explore vs. greedy
    if epsilon > 0 and self.rng.random() < epsilon:
        action_idx = _explore(self, info, allowed)
    else:
        action_idx = _pick_greedy(self, _break_loops(self, info, q), allowed)

    # Add position to recent_positions
    self.recent_positions.append(info['pos'])

    action = gs.ACTIONS[action_idx]

    if VERBOSE:
        self.logger.debug('step %d pos=%s eps=%.3f phi=%s q=%s -> %s',
                          game_state['step'], info['pos'], epsilon,
                          ''.join(map(str, phi.tolist())), q.round(2), action)
    return action

# train.py entry points
def setup_training(self):
    cfg = self.cfg
    self.memory = replay_module.ReplayMemory(
        cfg['memory'], self.spec.size, rare_fraction=cfg['rare_fraction'])
    self.nstep = replay_module.NStepAccumulator(cfg['n_step'], cfg['gamma'])

    self.gradient_steps = 0
    self._processed_step = None
    self._round_stats = _reset_round_stats()
    self._td_errors = []

    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    self.train_log_path = TRAIN_LOG_DIR / cfg['log_csv']
    self._log_fields = ['round', 'steps', 'score', 'coins', 'crates', 'kills',
                        'suicides', 'deaths', 'invalid', 'bombs', 'survived',
                        'reward', 'epsilon', 'td_abs', 'memory', 'model_size']
    if not self.train_log_path.exists():
        with open(self.train_log_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(self._log_fields)
    else:
        # Runs that were interrupted resume from their checkpoint, so the exploration schedule has to resume too
        self.round_index = _rounds_already_logged(self.train_log_path)
        if self.round_index:
            self.logger.warning('resuming training log at round %d',
                                self.round_index)

    # save on exit
    atexit.register(_save_quietly, self)

    self.logger.info(f'training on; log -> {self.train_log_path}')

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return

    old_phi, old_info = _extract_features(self, old_game_state)
    new_phi, new_info = _extract_features(self, new_game_state)

    _learn_from_step(self, old_phi, old_info, self_action, new_phi, new_info,
                     events, terminal=False)
    self._processed_step = old_game_state['step']

def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is not None and last_action is not None:
        already = self._processed_step == last_game_state['step']

        if already:
            # only SURVIVED_ROUND is new, fold it into the pending window
            bonus = rewards_module.reward_from_events(
                self.cfg, [ev for ev in events if ev == 'SURVIVED_ROUND'])
            self.nstep.add_reward_to_last(bonus)
            self._round_stats['reward'] += bonus
        else:
            old_phi, old_info = _extract_features(self, last_game_state)
            _learn_from_step(self, old_phi, old_info, last_action, None, None,
                             events, terminal=True)

    for transition in self.nstep.flush():
        self.memory.add(transition)
    self.nstep.reset()

    _train_batches(self, self.cfg['updates_per_step'] * 4)

    epsilon = _epsilon(self)  # the value actually used during this round
    self.round_index += 1
    _write_round_log(self, last_game_state, events, epsilon)

    if self.round_index % self.cfg['save_every'] == 0:
        self.model.save(self.checkpoint_path)
        self.logger.info(f'checkpoint written after round {self.round_index}')

    self._processed_step = None
    self._round_stats = _reset_round_stats()
    self._td_errors = []
    self._feat_cache = (None, None, None, None)

def _rounds_already_logged(path):
    # Number of rounds a previous, interrupted run recorded in the CSV
    try:
        with open(path, newline='') as fh:
            return max(0, sum(1 for _ in csv.reader(fh)) - 1)
    except OSError:
        return 0

def _save_quietly(self):
    # Best-effort checkpoint write, which calls when interpreter shutdown
    try:
        self.model.save(self.checkpoint_path)
    except Exception:
        pass

# region Helpers

def _extract_features(self, game_state):
    """Feature extraction with a one-step cache.

    act and game_events_occurred see the same states, so without the cache every state would be encoded twice
    a ~40% waste of training time.
    """
    if game_state is None:
        return None, None
    key_round, key_step = game_state['round'], game_state['step']
    c_round, c_step, c_phi, c_info = self._feat_cache
    if c_round == key_round and c_step == key_step:
        return c_phi, c_info
    phi, info = feat.extract(self.spec, game_state)
    self._feat_cache = (key_round, key_step, phi, info)
    return phi, info

def _epsilon(self):
    cfg = self.cfg
    if not self.train:
        # Q_EVAL_EPS: sweep vary inference-time exploration without editing configs
        override = os.environ.get('Q_EVAL_EPS', '').strip()
        return float(override) if override else cfg['eval_eps']
    decay = max(1, cfg['eps_decay_rounds'])
    return cfg['eps_end'] + (cfg['eps_start'] - cfg['eps_end']) * math.exp(
        -self.round_index / decay)


# Actions for greedy / exploration
def _explore(self, info, allowed):
    """Exploration distribution over the allowed actions.

    Two (ablatable) biases, both aimed at the same problem: in Bomberman a
    single random action can end the episode, so naive uniform exploration
    collects almost no experience.

    safe_explore
        With this probability the random choice is restricted to actions the
        escape search PROVES survivable.
        Added after stage-2 rounds last ~10 steps and nothing is ever learned.
        It is left below 1 on purpose as the remaining fraction still lets the agent experience death >:3
    eps_greedy_bias
        Avoids spending the exploration budget walking into walls.
    """
    p_safe = self.cfg['safe_explore']
    if p_safe > 0 and self.rng.random() < p_safe:
        safe = _get_proven_safe_actions(self, info, allowed)
        if safe:
            return int(self.rng.choice(safe))

    if self.cfg['eps_greedy_bias']:
        legal = [a for a in allowed if info['legal'][a]]
        if legal:
            return int(self.rng.choice(legal))
    return int(self.rng.choice(allowed))
def _pick_greedy(self, q, allowed):
    if len(allowed) < gs.N_ACTIONS:
        masked = np.full_like(q, -np.inf)
        masked[allowed] = q[allowed]
        q = masked
    best = np.flatnonzero(q == q.max())
    if best.size == 1 or self.cfg['greedy_tie_break'] != 'random':
        return int(best[0])
    return int(self.rng.choice(best))

def _get_proven_safe_actions(self, info, allowed):
    """
    Takes in all allowed actions the agent can take and returns the subset of actions that our escape search actually
    proves are survivable. Needed to quality safe exploration. After all, we kinda should know what is safe or if we bomb
    ourselves XD
    Empty only when the agent is already doomed.
    """
    safe = [a for a in allowed if a < 5 and (info['safe_mask'] >> a) & 1]
    if 5 in allowed and info['bombs_left'] and info['bomb_safe']:
        safe.append(5)
    return [a for a in safe if info['legal'][a]]

def _break_loops(self, info, q):
    """
    Charge recently visited tiles against the Q-values used for the choice.

    The idea is to basically prevent the agent just running in circles.

    Only the behaviour is changed - learning targets keep using the raw q, though makign it a decision rule,
    not a reward hack.

    To avoid the agent just standing still half the time, WAIT is charged for the agent's own tile.
    Ngl, it kinda doesn't seem to work all that well tho XD
    """
    penalty = self.cfg['loop_penalty']
    if penalty <= 0 or not self.recent_positions:
        return q

    counts = Counter(self.recent_positions)
    spread = float(q.max() - q.min())
    if spread <= 1e-9:
        spread = 1.0
    unit = penalty * spread

    adjusted = np.asarray(q, dtype=np.float64).copy()
    x, y = info['pos']
    for a, (dx, dy) in enumerate(gs.DELTAS):
        adjusted[a] -= unit * counts.get((x + dx, y + dy), 0)
    adjusted[4] -= unit * counts.get((x, y), 0)
    return adjusted

def _allowed_actions(self, info):
    """Action indices the agent may pick this step (never empty)."""
    allowed = self.enabled_actions
    if not self.cfg['safety_mask']:
        return allowed
    # if nothing is provably safe we are doomed anyway; fall back to the policy
    return _get_proven_safe_actions(self, info, allowed) or allowed

def _learn_from_step(self, old_phi, old_info, action, new_phi, new_info, events,
                     terminal):
    events = list(events)
    events += rewards_module.auxiliary_events(action, old_info, new_info, events)
    reward = rewards_module.total_reward(self.cfg, events, old_info,
                                         new_info if not terminal else None)

    _accumulate_stats(self._round_stats, events, reward)

    action_idx = gs.ACTION_INDEX.get(action)
    if action_idx is None:
        return

    important = rewards_module.is_important(events)
    transition = self.nstep.push(old_phi, action_idx, reward,
                                 None if terminal else new_phi, important)
    if transition is not None:
        self.memory.add(transition)

    if terminal:
        for pending in self.nstep.flush():
            self.memory.add(pending)
        self.nstep.reset()

    _train_batches(self, self.cfg['updates_per_step'])

def _train_batches(self, n_batches):
    cfg = self.cfg
    batch_size = cfg['batch_size']
    if len(self.memory) < batch_size:
        return

    for _ in range(n_batches):
        sample = self.memory.sample(batch_size, self.rng)
        if sample is None:
            return
        phi, action, ret, next_phi, length, done = sample


        q_next = self.model.q_batch_target(next_phi)
        bootstrap = q_next.max(axis=1)
        bootstrap[done] = 0.0
        targets = ret + (cfg['gamma'] ** length) * bootstrap

        if cfg['symmetry_augment']:
            phi, action, targets = _augment(self.spec, phi, action, targets,
                                            self.rng)

        errors = self.model.update(phi, action, targets)
        self._td_errors.append(float(np.abs(errors).mean()))

        self.gradient_steps += 1
        if cfg['target_sync'] and self.gradient_steps % cfg['target_sync'] == 0:
            self.model.sync_target()

def _augment(spec, phi, action, targets, rng):
    """
    Present each sampled transition under random board symmetry.

    Q is equivariant under D4, so a transition observed once is evidence
    about all eight symmetric situations.
    """
    n_sym = spec.index_perms.shape[0]
    which = rng.integers(0, n_sym, size=len(action))
    phi_g = np.take_along_axis(phi, spec.index_perms[which], axis=1)
    act_g = spec.action_perms[which, action]
    return phi_g, act_g, targets

def _reset_round_stats():
    return dict(coins=0, crates=0, kills=0, suicides=0, deaths=0, invalid=0,
                bombs=0, survived=0, reward=0.0)

def _accumulate_stats(stats, events, reward):
    stats['reward'] += reward
    for ev in events:
        if ev == 'COIN_COLLECTED':
            stats['coins'] += 1
        elif ev == 'CRATE_DESTROYED':
            stats['crates'] += 1
        elif ev == 'KILLED_OPPONENT':
            stats['kills'] += 1
        elif ev == 'KILLED_SELF':
            stats['suicides'] += 1
        elif ev == 'GOT_KILLED':
            stats['deaths'] += 1
        elif ev == 'INVALID_ACTION':
            stats['invalid'] += 1
        elif ev == 'BOMB_DROPPED':
            stats['bombs'] += 1
        elif ev == 'SURVIVED_ROUND':
            stats['survived'] = 1
def _write_round_log(self, last_game_state, events, epsilon):
    stats = self._round_stats
    for ev in events:
        if ev == 'SURVIVED_ROUND':
            stats['survived'] = 1

    steps = last_game_state['step'] if last_game_state else 0
    score = last_game_state['self'][1] if last_game_state else 0
    td = float(np.mean(self._td_errors)) if self._td_errors else 0.0
    model_stats = self.model.stats()
    model_size = model_stats.get('states', model_stats.get('w_norm', 0.0))

    row = [self.round_index, steps, score, stats['coins'], stats['crates'],
           stats['kills'], stats['suicides'], stats['deaths'], stats['invalid'],
           stats['bombs'], stats['survived'], round(stats['reward'], 3),
           round(epsilon, 4), round(td, 4), len(self.memory),
           round(float(model_size), 3)]
    with open(self.train_log_path, 'a', newline='') as fh:
        csv.writer(fh).writerow(row)

#endregion Helper