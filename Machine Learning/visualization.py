import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

COLOR_FEATURES = plt.cm.coolwarm(0.0)
COLOR_TARGET = plt.cm.coolwarm(1.0)
CMAP_2D = ListedColormap([COLOR_FEATURES, COLOR_TARGET])
CMAP_2D_CONT = plt.cm.coolwarm


def plot_2d_data(X, y, labels):
    fig, ax = plt.subplots(figsize=[10,10])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_2D, s=50, edgecolors='black')
    ax.set_xlabel(labels[0], fontsize=16)
    ax.set_ylabel(labels[1], fontsize=16)
    return fig


def plot_2d_boundary(classifier, X, y, labels, smooth=True):
    cmap = CMAP_2D_CONT if smooth else CMAP_2D
    fig, ax = plt.subplots(figsize=[10,10])

    xx, yy = _make_meshgrid(X[:, 0], X[:, 1])
    out = _plot_contours(ax, classifier, xx, yy, alpha=0.5, cmap=cmap,
            antialiased=True)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=50, edgecolors='black')
    ax.set_xlabel(labels[0], fontsize=16)
    ax.set_ylabel(labels[1], fontsize=16)
    ax.set_title('Decision Boundary', fontsize=22)
    return fig


def visualize_statistics(X, y, labels):
    num_cols = len(X.columns) + 1
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=[15, 5])

    for i, (c_name, c_vals) in enumerate(X.iteritems()):
        axes[i].hist(c_vals, bins=10, color=COLOR_FEATURES)
        axes[i].set_title(c_name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    counts = {label: count for label, count in zip(labels, y.value_counts())}
    axes[-1].bar(counts.keys(), counts.values(), color=COLOR_TARGET)
    axes[-1].set_xlabel('Label')
    axes[-1].set_xlabel('Label')

    fig.suptitle(
        'Information About Features and Target Variable',
        fontsize=22,
    )

    fig.tight_layout()

    return fig


def _make_meshgrid(x, y, h=.02):
    bounds = 1
    x_min, x_max = x.min() - bounds, x.max() + bounds
    y_min, y_max = y.min() - bounds, y.max() + bounds
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    return xx, yy


def _plot_contours(ax, clf, xx, yy, **params):
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    if hasattr(clf, 'decision_function'):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
