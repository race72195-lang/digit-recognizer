"""
Handwritten digit recognizer: training, evaluation and export of the model for the web app.

Data: the digits dataset bundled with scikit-learn (1,797 images of 8x8 pixels, no download).

The model must work on digits DRAWN BY A PERSON in a browser, which look different from the
tidy 8x8 scans. So training images are first turned into large "drawings" and randomly
rotated, sheared and thickened, then converted back to 8x8 with exactly the same preprocessing
that runs in the browser (see preprocess.py).

Usage:  python train.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, zoom
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from preprocess import preprocess
from strokes import render as render_strokes

BASE = Path(__file__).resolve().parent
SEED = 42
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
AUG_PER_IMAGE = 8
STROKES_TRAIN_PER_DIGIT = 700
STROKES_TEST_PER_DIGIT = 150


def to_drawing(img8):
    """8x8 scan (0..16) -> 96x96 smooth 'drawing' with margins (ink 0..1)."""
    big = zoom(img8 / 16.0, 8, order=1)
    big = np.pad(big, 16)
    return np.clip(big, 0, 1)


def random_drawing(img8, rng):
    """A randomly rotated, sheared and thickened version of a digit, like a human drawing."""
    ink = to_drawing(img8)
    ang = np.deg2rad(rng.uniform(-12, 12))
    shear = rng.uniform(-0.2, 0.2)
    c, s = np.cos(ang), np.sin(ang)
    m = np.array([[c, -s], [s, c]]) @ np.array([[1, shear], [0, 1]])
    centre = np.array(ink.shape) / 2
    ink = affine_transform(ink, m, offset=centre - m @ centre, order=1)
    ink = gaussian_filter(ink, rng.uniform(0.3, 2.5))
    return ink / ink.max() if ink.max() > 0 else ink


def make_set(images, labels, rng, n_aug):
    X, y = [], []
    for img, lab in zip(images, labels):
        for k in range(n_aug):
            f = preprocess(random_drawing(img, rng))
            if f is not None:
                X.append(f / 16.0); y.append(lab)
    return np.array(X), np.array(y)


def make_clean(images, labels):
    X, y = [], []
    for img, lab in zip(images, labels):
        f = preprocess(to_drawing(img))
        if f is not None:
            X.append(f / 16.0); y.append(lab)
    return np.array(X), np.array(y)


def make_strokes(n_per_digit, seed, hard=False):
    """Digits drawn as thick or thin pen lines (see strokes.py)."""
    r = np.random.default_rng(seed)
    X, y = [], []
    for lab in range(10):
        for _ in range(n_per_digit):
            f = preprocess(render_strokes(lab, r, hard=hard))
            X.append(f / 16.0); y.append(lab)
    return np.array(X), np.array(y)


def main():
    (BASE / "results").mkdir(exist_ok=True)
    (BASE / "images").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = load_digits()
    idx_tr, idx_te = train_test_split(np.arange(len(d.target)), test_size=0.2, stratify=d.target, random_state=SEED)
    print(f"Images: {len(d.target)} (8x8 pixels, 10 classes) | train {len(idx_tr)} | test {len(idx_te)}")

    X_tr, y_tr = make_set(d.images[idx_tr], d.target[idx_tr], rng, AUG_PER_IMAGE)
    X_te_clean, y_te_clean = make_clean(d.images[idx_te], d.target[idx_te])
    X_te_draw, y_te_draw = make_set(d.images[idx_te], d.target[idx_te], np.random.default_rng(SEED + 1), 4)
    X_st, y_st = make_strokes(STROKES_TRAIN_PER_DIGIT, SEED + 10)
    X_tr = np.vstack([X_tr, X_st]); y_tr = np.concatenate([y_tr, y_st])
    X_te_st, y_te_st = make_strokes(STROKES_TEST_PER_DIGIT, SEED + 99)
    X_te_hard, y_te_hard = make_strokes(STROKES_TEST_PER_DIGIT, SEED + 77, hard=True)
    print(f"Training samples: {len(y_tr)} (distorted scans + simulated pen strokes)")

    models = {
        "Logistic regression": LogisticRegression(max_iter=3000),
        "Neural network (64-64-10)": MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=600, random_state=SEED),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        a_clean = accuracy_score(y_te_clean, m.predict(X_te_clean))
        a_draw = accuracy_score(y_te_draw, m.predict(X_te_draw))
        a_st = accuracy_score(y_te_st, m.predict(X_te_st))
        a_hard = accuracy_score(y_te_hard, m.predict(X_te_hard))
        rows.append({"model": name, "accuracy_clean_scans": a_clean, "accuracy_distorted_scans": a_draw,
                     "accuracy_pen_strokes": a_st, "accuracy_pen_strokes_hard": a_hard})
        print(f"{name}: clean scans {a_clean:.3f} | distorted scans {a_draw:.3f} | pen strokes {a_st:.3f} | hard pen strokes {a_hard:.3f}")
    mlp = models["Neural network (64-64-10)"]
    (BASE / "results" / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    per_digit = [float(np.mean(mlp.predict(X_te_hard[y_te_hard == k]) == k)) for k in range(10)]
    print("Hard pen strokes, accuracy per digit:", {k: round(v, 2) for k, v in enumerate(per_digit)})
    pred = mlp.predict(X_te_hard)
    cm = confusion_matrix(y_te_hard, pred)
    worst = sorted([(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j], reverse=True)[:3]
    print("Most common mistakes (true -> predicted, count):", [(i, j, int(n)) for n, i, j in worst])

    # ---- export weights for the browser
    weights = {"W1": np.round(mlp.coefs_[0], 4).tolist(), "b1": np.round(mlp.intercepts_[0], 4).tolist(),
               "W2": np.round(mlp.coefs_[1], 4).tolist(), "b2": np.round(mlp.intercepts_[1], 4).tolist()}
    acc_draw = rows[1]["accuracy_pen_strokes_hard"]
    tpl = (BASE / "app_template.html").read_text(encoding="utf-8")
    html = (tpl.replace("/*MODEL_JSON*/null", json.dumps(weights, separators=(",", ":")))
               .replace("__ACC__", f"{acc_draw * 100:.0f}")
               .replace("__ACC2__", f"{rows[1]['accuracy_clean_scans'] * 100:.0f}"))
    (BASE / "docs" / "index.html").write_text(html, encoding="utf-8")
    print(f"Web app written: docs/index.html ({len(html) / 1024:.0f} KB)")

    # ---- charts
    fig, ax = plt.subplots(figsize=(5.6, 5), facecolor=SURFACE)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        if v:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="white" if v > cm.max() * 0.6 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit", color=MUTED); ax.set_ylabel("True digit", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix, hard pen-stroke test (accuracy {acc_draw:.1%})", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(BASE / "images" / "01_confusion_matrix.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    r2 = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 8, figsize=(10, 4.3), facecolor=SURFACE)
    for k in range(8):
        i = idx_te[k * 11]
        draw = random_drawing(d.images[i], r2)
        feat = preprocess(draw).reshape(8, 8)
        for r, (im, ttl) in enumerate(((d.images[i], "original 8x8"), (draw, "simulated drawing"), (feat, "model input"))):
            axes[r, k].imshow(im, cmap="Greys", interpolation="nearest")
            axes[r, k].axis("off")
            if k == 0:
                axes[r, k].set_title(ttl, loc="left", fontsize=9, color=MUTED)
    fig.suptitle("From a drawing to what the model sees", x=0.02, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(BASE / "images" / "02_what_the_model_sees.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
