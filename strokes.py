"""Synthetic pen-stroke digits: outlines of the digits 0-9 drawn as lines of random thickness,
with random size, rotation, shear and wobble. They imitate drawing with a mouse or a finger.
The digit shapes are hand-made templates, so results on them are optimistic."""
import numpy as np
from PIL import Image, ImageDraw

def ell(cx, cy, rx, ry, a0=0, a1=2*np.pi, n=36):
    t = np.linspace(a0, a1, n)
    return list(zip(cx + rx*np.cos(t), cy + ry*np.sin(t)))

TEMPLATES = {
 0: [[ell(.5,.5,.22,.38)]],
 1: [[[(.5,.1),(.5,.9)]], [[(.32,.28),(.5,.1),(.5,.9)]], [[(.58,.1),(.44,.9)]],
     [[(.32,.28),(.5,.1),(.5,.9)],[(.33,.9),(.67,.9)]], [[(.5,.1),(.5,.9)],[(.35,.9),(.65,.9)]], [[(.4,.25),(.55,.1),(.55,.9)]]],
 2: [[[(.25,.3),(.4,.12),(.65,.12),(.75,.3),(.6,.55),(.25,.88),(.78,.88)]]],
 3: [[[(.25,.2),(.45,.1),(.68,.2),(.6,.42),(.42,.5),(.68,.6),(.72,.78),(.5,.9),(.25,.8)]]],
 4: [[[(.62,.1),(.22,.62),(.82,.62)],[(.62,.1),(.62,.92)]], [[(.3,.1),(.25,.55),(.75,.55)],[(.65,.1),(.65,.9)]]],
 5: [[[(.7,.12),(.32,.12),(.28,.45),(.55,.4),(.75,.55),(.72,.78),(.5,.9),(.25,.8)]]],
 6: [[[(.65,.1),(.35,.35),(.27,.65),(.38,.88),(.62,.88),(.72,.68),(.6,.5),(.38,.52),(.28,.65)]]],
 7: [[[(.25,.15),(.75,.15),(.45,.9)]], [[(.25,.15),(.75,.15),(.45,.9)],[(.4,.52),(.65,.52)]]],
 8: [[ell(.5,.3,.18,.2), ell(.5,.7,.22,.2)], [[(.5,.5)] + ell(.5,.3,.2,.2)[::-1] + [(.5,.5)] + ell(.5,.7,.24,.2)]],
 9: [[ell(.5,.32,.22,.22), [(.72,.32),(.6,.9)]]],
}

def random_one(rng):
    """A '1' with a random slant, an optional flag at the top and an optional base."""
    dx = rng.uniform(-.12, .12)
    top, bottom = (.5 + dx, .1), (.5 - dx, .9)
    st = [[top, bottom]]
    if rng.random() < .5:
        L = rng.uniform(.1, .25)
        st[0] = [(top[0] - L, top[1] + L * .9), top, bottom]
    if rng.random() < .25:
        st.append([(bottom[0] - .17, .9), (bottom[0] + .17, .9)])
    return st


def render(label, rng, size=280, brush=None, hard=False):
    strokes = random_one(rng) if (label == 1 and rng.random() < .6) else TEMPLATES[label][rng.integers(len(TEMPLATES[label]))]
    amax, smax, noise = (22, .35, .035) if hard else (12, .2, .012)
    ang = np.deg2rad(rng.uniform(-amax, amax)); sh = rng.uniform(-smax, smax)
    sx, sy = rng.uniform(.6, .95), rng.uniform(.6, .95)
    if label == 1 and rng.random() < .5: sx = sy
    off = rng.uniform(-.06, .06, 2)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([[1, sh], [0, 1]])
    w = brush if brush else (rng.uniform(10, 40) if hard else rng.uniform(14, 34))
    im = Image.new("L", (size, size), 0); d = ImageDraw.Draw(im)
    for st in strokes:
        pts = np.array(st, float) + rng.normal(0, noise, (len(st), 2))
        pts = (pts - .5) * np.array([sx, sy]) @ R.T + .5 + off
        pts = [(x * size, y * size) for x, y in pts]
        d.line(pts, fill=255, width=int(w), joint="curve")
        for x, y in pts: d.ellipse([x - w/2, y - w/2, x + w/2, y + w/2], fill=255)
    return np.asarray(im, float) / 255
