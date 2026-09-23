import numpy as np

# Quarter turn: UP -> RIGHT -> DOWN -> LEFT -> UP.
ROT = (1, 2, 3, 0)
# Reflection about the vertical axis: swaps RIGHT and LEFT.
MIRROR = (0, 3, 2, 1)


def _compose(g, h):
    """
    Function composition (g o h)(d) = g[h[d]].
    """
    return tuple(g[h[d]] for d in range(4))


def _generate():
    identity = (0, 1, 2, 3)
    rotations = [identity]
    for _ in range(3):
        rotations.append(_compose(ROT, rotations[-1]))
    group = []
    for r in rotations:
        group.append(r)
        group.append(_compose(r, MIRROR))
    return tuple(group)


DIR_MAPS = _generate()
N_SYMMETRIES = len(DIR_MAPS)


def action_permutations(n_actions=6):
    perms = np.zeros((N_SYMMETRIES, n_actions), dtype=np.int64)
    for gi, gmap in enumerate(DIR_MAPS):
        for a in range(n_actions):
            perms[gi, a] = gmap[a] if a < 4 else a
    return perms


def index_permutations(layout, size):
    perms = np.zeros((N_SYMMETRIES, size), dtype=np.int64)
    for gi, gmap in enumerate(DIR_MAPS):
        ginv = [0] * 4
        for d, image in enumerate(gmap):
            ginv[image] = d
        idx = np.arange(size)
        for _name, base, width, is_dir in layout:
            if is_dir:
                for j in range(4):
                    idx[base + j] = base + ginv[j]
        perms[gi] = idx
    return perms
