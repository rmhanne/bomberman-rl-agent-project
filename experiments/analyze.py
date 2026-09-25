"""
Little module to generate figures for the report using matplotlib.

    python -m experiments.analyze curves --stage 1
    python -m experiments.analyze curves all

Just gotta make sure the csv files exist. Usually done via training in results/train/*.csv

All figures are written next to their .csv holding exactly the numbers plotted,
so the figures never have to be trusted on their own and the report can quote
the table instead of the picture.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRAIN_DIR = REPO / 'results' / 'train'
EVAL_DIR = REPO / 'results' / 'eval'
FIG_DIR = REPO / 'figures'

#color config
SURFACE = '#ffffff' # To fit with report color
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
        'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'grid.color': GRID, 'grid.linewidth': 0.8,
        'legend.frameon': False, 'figure.dpi': 140,
    })


def tidy(ax, title=None, xlabel=None, ylabel=None):
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc='left', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def color_for(index):
    """Colour by series identity. Past eight slots we would fold to 'Other'."""
    if index >= len(SERIES):
        raise ValueError('more than 8 series: fold into "Other" or facet')
    return SERIES[index]


def write_table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_train_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            try:
                out[key].append(float(value))
            except (TypeError, ValueError):
                pass
    return out


def rolling(values, window):
    if window <= 1 or len(values) < 2:
        return list(values)
    out, total = [], 0.0
    from collections import deque
    buf = deque()
    for v in values:
        buf.append(v)
        total += v
        if len(buf) > window:
            total -= buf.popleft()
        out.append(total / len(buf))
    return out


# stage -> (variant order, metrics to plot)
CURVE_SETS = {
    '1': (['task1'],
              [('coins', 'coins collected per round'),
               ('suicides', 'suicides per round')]),
    '2': (['task2'],
              [('coins', 'coins collected per round'),
               ('crates', 'crates destroyed per round'),
               ('suicides', 'suicides per round'),
               ('steps', 'round length (steps)')]),
    '3': (['task3'], [('score', 'score per round'),
                      ('kills', 'kills per round'),
                      ('coins', 'coins per round'),
                      ('suicides', 'suicides per round')]),
    '4': (['task4'], [('score', 'score per round'),
                      ('kills', 'kills per round'),
                      ('coins', 'coins per round'),
                      ('survived', 'survival rate')]),
}


def plot_curves(stage, window=100, out_name=None):
    if stage not in CURVE_SETS:
        raise SystemExit(f'unknown curve set {stage!r}; known: {sorted(CURVE_SETS)}')
    variants, metrics = CURVE_SETS[stage]

    available = [(v, TRAIN_DIR / f'{v}_train.csv') for v in variants]
    available = [(v, p) for v, p in available if p.is_file()]
    if not available:
        print(f'  (no training logs for stage {stage}, skipping)')
        return None

    style()
    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.1 * nrows),
                             squeeze=False)

    table_rows = []
    for panel, (metric, label) in enumerate(metrics):
        ax = axes[panel // ncols][panel % ncols]
        for i, (variant, path) in enumerate(available):
            data = read_train_csv(path)
            if metric not in data:
                continue
            y = rolling(data[metric], window)
            x = data.get('round', list(range(1, len(y) + 1)))
            ax.plot(x, y, color=color_for(i), linewidth=2.0, label=variant,
                    solid_capstyle='round')
            if panel == 0:
                for r, v in zip(x, y):
                    table_rows.append({'variant': variant, 'round': r,
                                       'metric': metric, 'value': round(v, 4)})
        tidy(ax, title=label, xlabel='training round', ylabel=None)

    for panel in range(len(metrics), nrows * ncols):
        axes[panel // ncols][panel % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Stage {stage}: training progress '
                 f'(rolling mean over {window} rounds)',
                 fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out_name = out_name or f'stage{stage}_learning'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{out_name}.png', bbox_inches='tight')
    plt.close(fig)
    write_table(FIG_DIR / f'{out_name}.csv', table_rows,
                ['variant', 'round', 'metric', 'value'])
    print(f'  wrote figures/{out_name}.png (+ .csv)')
    return FIG_DIR / f'{out_name}.png'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    c = sub.add_parser('curves')
    c.add_argument('--stage', default='1')
    c.add_argument('--window', type=int, default=100)

    sub.add_parser('all')

    args = parser.parse_args(argv)

    if args.command == 'curves':
        plot_curves(args.stage, args.window)
    else:
        for stage in CURVE_SETS:
            plot_curves(stage)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
