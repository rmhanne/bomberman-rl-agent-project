"""Q-function approximators.

Two models behind one interface so that train.py is model-agnostic and the
tabular/linear comparison is a one-word config change:

LinearQ
    Q(s, a) = w_a . phi(s) with per-action weight vectors, trained by
    semi-gradient Q-learning with RMSProp and a periodically synced target copy.
    Generalises across states that share features, which is what makes the
    ~10^5-state full game tractable.

Note 14.9.2026: fully removed the tabular implementation
"""

import pickle
from pathlib import Path

import numpy as np

# WHY THE FUCK CAN PYTHON NOT HAVE AN INTERFACE MAN >:(
class QModel:
    """Common interface."""

    kind = 'abstract'

    def q_values(self, phi):
        raise NotImplementedError

    def q_batch(self, phis):
        raise NotImplementedError

    def q_batch_target(self, phis):
        return self.q_batch(phis)

    def update(self, phis, actions, targets):
        raise NotImplementedError

    def sync_target(self):
        pass

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(self.state_dict(), fh)

    def state_dict(self):
        raise NotImplementedError

    def stats(self):
        return {}


# Implementations
class LinearQ(QModel):
    kind = 'linear'

    def __init__(self, n_actions, layout, lr=0.02, grad_clip=5.0,
                 rmsprop_decay=0.95, eps=1e-6, init_scale=0.0):
        self.n_actions = n_actions
        self.layout = _layout_tuple(layout)
        self.n_features = sum(w for _n, _b, w in self.layout)
        self.lr = lr
        self.grad_clip = grad_clip
        self.rmsprop_decay = rmsprop_decay
        self.eps = eps

        rng = np.random.default_rng(0)
        self.W = (init_scale * rng.standard_normal((n_actions, self.n_features))
                  if init_scale else np.zeros((n_actions, self.n_features)))
        self.W_target = self.W.copy()
        self.ms = np.zeros_like(self.W)  # RMSProp running mean square

    def q_values(self, phi):
        return self.W @ np.asarray(phi, dtype=np.float64)

    def q_batch(self, phis):
        return np.asarray(phis, dtype=np.float64) @ self.W.T

    def q_batch_target(self, phis):
        return np.asarray(phis, dtype=np.float64) @ self.W_target.T

    def update(self, phis, actions, targets):
        phis = np.asarray(phis, dtype=np.float64)
        n_samples = len(actions)

        pred = (phis @ self.W.T)[np.arange(n_samples), actions]
        errors = np.clip(targets - pred, -self.grad_clip, self.grad_clip)

        scatter = np.zeros((n_samples, self.n_actions))
        scatter[np.arange(n_samples), actions] = errors
        grad = scatter.T @ phis                      # (n_actions, n_features)

        counts = np.bincount(actions, minlength=self.n_actions)
        touched = counts > 0
        grad[touched] /= counts[touched, None]

        self.ms[touched] = (self.rmsprop_decay * self.ms[touched]
                            + (1.0 - self.rmsprop_decay) * grad[touched] ** 2)
        self.W[touched] += (self.lr * grad[touched]
                            / (np.sqrt(self.ms[touched]) + self.eps))

        return errors

    def sync_target(self):
        self.W_target = self.W.copy()

    def state_dict(self):
        return dict(kind=self.kind, n_actions=self.n_actions,
                    layout=self.layout, W=self.W, lr=self.lr,
                    grad_clip=self.grad_clip)

    def load_state(self, sd):
        """
        Load weights, remapping feature blocks by name.
        """
        if sd.get('kind') != self.kind:
            return False
        src_layout = _layout_tuple(sd['layout'])
        src_W = sd['W']
        src_offsets = {name: (base, width) for name, base, width in src_layout}

        copied = 0
        for name, base, width in self.layout:
            if name in src_offsets:
                sbase, swidth = src_offsets[name]
                if swidth == width:
                    self.W[:, base:base + width] = src_W[:, sbase:sbase + swidth]
                    copied += width
        self.sync_target()
        return copied > 0

    def stats(self):
        return {'w_absmax': float(np.abs(self.W).max()),
                'w_norm': float(np.linalg.norm(self.W))}

def _layout_tuple(layout):
    """Normalise a FeatureSpec layout to ((name, base, width), ...)."""
    out = []
    for entry in layout:
        name, base, width = entry[0], entry[1], entry[2]
        out.append((name, int(base), int(width)))
    return tuple(out)

def build(cfg, spec, n_actions):
    """Instantiate the model named by cfg['model']."""
    return LinearQ(n_actions, spec.layout, lr=cfg['lr'],
                       grad_clip=cfg['grad_clip'])

def load_checkpoint(model, path, logger=None):
    """Try to restore model from path; return True on success."""
    path = Path(path)
    if not path.is_file():
        if logger:
            logger.info(f'no checkpoint at {path}, starting from scratch')
        return False
    try:
        with open(path, 'rb') as fh:
            sd = pickle.load(fh)
            logger.info(f"Loaded checkpoint from {path}")
    except Exception as exc:  # pragma: no cover - corrupt file
        if logger:
            logger.warning(f'could not read {path}: {exc}')
        return False

    ok = model.load_state(sd)
    if logger:
        logger.info(f'checkpoint {path} -> {"loaded" if ok else "incompatible, starting fresh"}')
    return ok
