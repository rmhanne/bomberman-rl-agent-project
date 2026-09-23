"""
n-step return acc + experience replay.

Two ideas here matter for the experiments:

n-step returns:
    Bomberman rewards are sparse and delayed as a bomb dropped now pays off four
    steps later (or not and we die XD). However, propagating credit one step at a time needs many more episodes
    than propagating it n steps at a time.

Two-pool replay:
    Deaths, kills, and coins are rare compared to plain movement. Uniform replay
    over a 100k buffer would show a death to the learner roughly once per batch.
    We therefore keep a second, smaller pool of "rare" transitions and draw a
    fixed fraction of every batch from it.
"""

from collections import deque

import numpy as np


class NStepAccumulator:
    """Turns a stream of one-step transitions into n-step transitions."""

    def __init__(self, n, gamma):
        self.n = max(1, int(n))
        self.gamma = float(gamma)
        self.buf = deque()

    def reset(self):
        self.buf.clear()

    def push(self, phi, action, reward, next_phi, important=False):
        """Add one environment step; return a finished n-step transition or None."""
        self.buf.append([phi, action, float(reward), next_phi, bool(important)])
        if len(self.buf) >= self.n:
            out = self._emit(self.n, done=False)
            self.buf.popleft()
            return out
        return None

    def add_reward_to_last(self, bonus):
        """Attribute a late bonus (e.g. SURVIVED_ROUND) to the newest step.

        The framework only appends SURVIVED_ROUND AFTER the last
        game_events_occurred call, so the bonus arrives once the step has
        already been pushed.
        Windows that were emitted before the bonus arrived keep the un-bonused return
        every window still pending in the buffer picks it up.
        The residual bias is at most one n-step window per round.
        """
        if self.buf and bonus:
            self.buf[-1][2] += float(bonus)
            self.buf[-1][4] = True

    def flush(self):
        """Close the episode: every pending prefix becomes a terminal transition."""
        out = []
        while self.buf:
            out.append(self._emit(len(self.buf), done=True))
            self.buf.popleft()
        return out

    def _emit(self, length, done):
        head_phi, head_action = self.buf[0][0], self.buf[0][1]
        ret, discount, important = 0.0, 1.0, False
        for i in range(length):
            ret += discount * self.buf[i][2]
            discount *= self.gamma
            important = important or self.buf[i][4]
        next_phi = self.buf[length - 1][3]
        return (head_phi, head_action, ret, next_phi, length, done, important)


class _Pool:
    """Fixed-capacity ring buffer of transitions, stored column-wise."""

    def __init__(self, capacity, n_features):
        self.capacity = int(capacity)
        self.phi = np.zeros((self.capacity, n_features), dtype=np.int8)
        self.next_phi = np.zeros((self.capacity, n_features), dtype=np.int8)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.ret = np.zeros(self.capacity, dtype=np.float64)
        self.length = np.zeros(self.capacity, dtype=np.int64)
        self.done = np.zeros(self.capacity, dtype=bool)
        self.size = 0
        self.cursor = 0

    def add(self, phi, action, ret, next_phi, length, done):
        i = self.cursor
        self.phi[i] = phi
        self.next_phi[i] = 0 if next_phi is None else next_phi
        self.action[i] = action
        self.ret[i] = ret
        self.length[i] = length
        self.done[i] = done or next_phi is None
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, k, rng):
        if self.size == 0 or k <= 0:
            return np.empty(0, dtype=np.int64)
        return rng.integers(0, self.size, size=k)

class ReplayMemory:
    """Main pool plus a smaller pool of rare, high-signal transitions."""

    def __init__(self, capacity, n_features, rare_fraction=0.25,
                 rare_capacity=None):
        self.rare_fraction = float(rare_fraction)
        self.main = _Pool(capacity, n_features)
        self.rare = _Pool(rare_capacity or max(1024, capacity // 10), n_features)

    def __len__(self):
        return self.main.size + self.rare.size

    def add(self, transition):
        phi, action, ret, next_phi, length, done, important = transition
        self.main.add(phi, action, ret, next_phi, length, done)
        if important:
            self.rare.add(phi, action, ret, next_phi, length, done)

    def sample(self, batch_size, rng):
        """Return (phi, action, ret, next_phi, length, done) arrays."""
        n_rare = 0
        if self.rare.size > 0 and self.rare_fraction > 0:
            n_rare = min(int(batch_size * self.rare_fraction), self.rare.size)
        n_main = batch_size - n_rare
        if self.main.size == 0:
            n_rare, n_main = batch_size, 0

        parts = []
        for pool, k in ((self.main, n_main), (self.rare, n_rare)):
            idx = pool.sample_indices(k, rng)
            if idx.size:
                parts.append((pool, idx))

        if not parts:
            return None

        phi = np.concatenate([p.phi[i] for p, i in parts])
        next_phi = np.concatenate([p.next_phi[i] for p, i in parts])
        action = np.concatenate([p.action[i] for p, i in parts])
        ret = np.concatenate([p.ret[i] for p, i in parts])
        length = np.concatenate([p.length[i] for p, i in parts])
        done = np.concatenate([p.done[i] for p, i in parts])
        return phi, action, ret, next_phi, length, done

    def stats(self):
        return {'memory': self.main.size, 'memory_rare': self.rare.size}
