"""Generate fig_overview.tex (TikZ, standalone) for Figure 1.

The covariance cartoon is a 2-D analogue of the real setting: two support points per
class in d = 2 (K - 1 < d), so each class covariance is rank-deficient, as with
K <= 10 support images in the 512-d fused space. Ellipses are R-sigma contours computed
exactly from Eq. (2), Eq. (5) and the shrinkage rule with alpha = 0.75.
Usage:  python make_overview.py <out_dir>   then   tectonic <out_dir>/fig_overview.tex
"""
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ALPHA, LAM, R = 0.75, 3e-4, 1.25

# class: (centre, direction in degrees, half-spread, colour name)
classes = [
    ((-0.40, 0.24), 16, 0.15, "cA"),
    ((0.42, 0.26), 108, 0.13, "cB"),
    ((0.02, -0.36), -36, 0.16, "cC"),
]

stats, scat, dof = [], np.zeros((2, 2)), 0
for c, deg, h, col in classes:
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    pts = [np.array(c) + s * (h * u + np.array([0.010, -0.008])) for s in (1, -1)]
    d = np.array(pts) - np.array(c)
    S = d.T @ d
    stats.append((c, pts, S / (len(pts) - 1), col))
    scat += S
    dof += len(pts) - 1
S_pool = scat / dof


def ellipse(S):
    w, v = np.linalg.eigh(S + LAM * np.eye(2))
    ang = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    return R * math.sqrt(w[1]), R * math.sqrt(w[0]), ang


def ell(col_style, ox, oy, sc, c, S):
    a, b, ang = ellipse(S)
    return (f"\\draw[{col_style}] ({ox + sc * c[0]:.3f},{oy + sc * c[1]:.3f}) "
            f"ellipse[x radius={sc * a:.3f}, y radius={sc * b:.3f}, rotate={ang:.1f}];")


def dot(col, x, y):
    return f"\\fill[{col}] ({x:.3f},{y:.3f}) circle (0.05);"


def cross(col, x, y, r=0.06):
    return (f"\\draw[{col}!70!black, line width=0.6pt] ({x - r:.3f},{y - r:.3f}) -- ({x + r:.3f},{y + r:.3f}) "
            f"({x - r:.3f},{y + r:.3f}) -- ({x + r:.3f},{y - r:.3f});")


def panel(kind, ox, oy, sc):
    L = []
    if kind == "pool":   # residuals of every class, re-centred on the origin
        a, b, ang = ellipse(S_pool)
        L.append(f"\\draw[ours, line width=0.8pt, fill=ours!8] ({ox:.3f},{oy:.3f}) "
                 f"ellipse[x radius={sc * a:.3f}, y radius={sc * b:.3f}, rotate={ang:.1f}];")
        L.append(f"\\draw[black!45, line width=0.35pt] ({ox - 0.09:.3f},{oy:.3f}) -- ({ox + 0.09:.3f},{oy:.3f}) "
                 f"({ox:.3f},{oy - 0.09:.3f}) -- ({ox:.3f},{oy + 0.09:.3f});")
        for c, pts, S, col in stats:
            for p in pts:
                r = p - np.array(c)
                L.append(dot(col, ox + sc * r[0], oy + sc * r[1]))
        return L
    for c, pts, S, col in stats:
        Sx = S if kind == "class" else (1 - ALPHA) * S + ALPHA * S_pool
        style = f"{col}, line width=0.75pt" + (f", fill={col}!12" if kind == "shrunk" else "")
        L.append(ell(style, ox, oy, sc, c, Sx))
        for p in pts:
            L.append(dot(col, ox + sc * p[0], oy + sc * p[1]))
        L.append(cross(col, ox + sc * c[0], oy + sc * c[1]))
    return L


H, GAP = 3.30, 0.36
widths = [("A", 2.62), ("B", 2.80), ("C", 2.70), ("D", 5.00), ("E", 3.10)]
boxes, x = {}, 0.0
for k, w in widths:
    boxes[k] = (x, w)
    x += w + GAP
cx = lambda k: boxes[k][0] + boxes[k][1] / 2
right = lambda k: boxes[k][0] + boxes[k][1]
F = lambda size: f"\\fontsize{{{size}}}{{{size + 1.1:.1f}}}\\selectfont"

T = [r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,decorations.pathreplacing}
\definecolor{ink}{HTML}{222222}
\definecolor{muted}{HTML}{767676}
\definecolor{ours}{HTML}{1f5fa8}
\definecolor{oursbg}{HTML}{eef3fa}
\definecolor{boxbg}{HTML}{f7f7f5}
\definecolor{cA}{HTML}{2a9d8f}
\definecolor{cB}{HTML}{c8641c}
\definecolor{cC}{HTML}{7b5ea7}
\begin{document}
\begin{tikzpicture}[font=\fontsize{7.2}{8.4}\selectfont, ink,
  box/.style={rounded corners=3pt, draw=black!28, fill=boxbg, line width=0.5pt},
  oursbox/.style={rounded corners=3pt, draw=ours, fill=oursbg, line width=0.9pt},
  title/.style={font=\fontsize{7.6}{9}\selectfont\bfseries, anchor=north},
  flow/.style={-{Stealth[length=4.2pt,width=3.6pt]}, line width=0.8pt, black!55},
  chip/.style={rounded corners=1.5pt, draw=black!35, fill=white, line width=0.4pt, minimum height=0.42cm, inner sep=1.5pt},
  panelframe/.style={draw=black!15, line width=0.3pt, fill=white},
  tag/.style={anchor=north west, font=\fontsize{5.9}{6.6}\selectfont, text=muted, inner sep=1.6pt}]"""]

for k, (x0, w) in boxes.items():
    T.append(f"\\path[{'oursbox' if k == 'D' else 'box'}] ({x0:.2f},0) rectangle ({x0 + w:.2f},{H});")
titles = {"A": r"Session $t$ support set", "B": "Frozen CLIP + fusion",
          "C": "Per-class statistics", "D": r"Pooled shrinkage (ours)", "E": "Hybrid scoring"}
for k, t in titles.items():
    T.append(f"\\node[title{', text=ours' if k == 'D' else ''}] at ({cx(k):.2f},{H - 0.10}) {{{t}}};")
for a, b in ("AB", "BC", "CD", "DE"):
    T.append(f"\\draw[flow] ({right(a) + 0.03:.2f},{H / 2 - 0.08:.2f}) -- ({boxes[b][0] - 0.03:.2f},{H / 2 - 0.08:.2f});")

# A: support thumbnails (3 classes x 4 shots; middle classes elided)
xa, _ = boxes["A"]
for r, (col, lab) in enumerate([("cA", "class 1"), ("cB", "class 2"), ("cC", r"class $C$")]):
    y = 2.30 - r * 0.60 - (0.24 if r == 2 else 0)
    for s in range(4):
        x0 = xa + 0.14 + s * 0.35
        T.append(f"\\path[fill={col}!{16 + 10 * ((s + r) % 3)}, draw={col}!70, line width=0.45pt, rounded corners=0.6pt] "
                 f"({x0:.2f},{y:.2f}) rectangle ++(0.28,0.28);")
        T.append(f"\\draw[{col}!60, line width=0.35pt] ({x0 + 0.05:.2f},{y + 0.06:.2f}) .. controls "
                 f"({x0 + 0.12:.2f},{y + 0.24:.2f}) .. ({x0 + 0.23:.2f},{y + 0.11:.2f});")
    T.append(f"\\node[anchor=west, font={F(6.3)}, text={col}!80!black, inner sep=1pt] at ({xa + 1.53:.2f},{y + 0.14:.2f}) {{{lab}}};")
for d in range(3):
    T.append(f"\\fill[muted] ({xa + 0.80:.2f},{1.16 + d * 0.09:.2f}) circle (0.021);")
T.append(f"\\draw[decorate, decoration={{brace, amplitude=2.5pt, mirror}}, muted, line width=0.4pt] "
         f"({xa + 0.14:.2f},0.56) -- ({xa + 1.47:.2f},0.56);")
T.append(f"\\node[text=muted, font={F(6.5)}] at ({xa + 0.80:.2f},0.30) {{$K$ shots}};")

# B: encoders and fusion
xb, wb = boxes["B"]
T.append(f"\\node[chip, minimum width=1.25cm] (ie) at ({xb + 0.82:.2f},2.24) {{image enc.}};")
T.append(f"\\node[chip, minimum width=1.25cm] (te) at ({xb + 0.82:.2f},1.42) {{text enc.}};")
T.append(f"\\node[font={F(6)}, text=muted, anchor=south west, inner sep=0.5pt] at ($(ie.north west)+(0,0.02)$) {{\\itshape frozen ViT-B/16}};")
T.append(f"\\node[draw=black!55, circle, inner sep=0.2pt, minimum size=0.36cm, fill=white, line width=0.5pt] (plus) at ({xb + 2.20:.2f},1.83) {{$+$}};")
T.append(r"\draw[-{Stealth[length=3pt]}, black!50, line width=0.5pt] (ie.east) -- node[above, pos=0.45, inner sep=1pt] {$v$} (plus);")
T.append(r"\draw[-{Stealth[length=3pt]}, black!50, line width=0.5pt] (te.east) -- node[below, pos=0.45, inner sep=1pt] {$t_c$} (plus);")
T.append(r"\draw[-{Stealth[length=3pt]}, black!50, line width=0.5pt] (plus.south) -- ++(0,-0.40);")
T.append(f"\\node[font={F(6.8)}] at ({cx('B'):.2f},0.56) {{$e_c=\\dfrac{{v+t_c}}{{\\lVert v+t_c\\rVert}}\\in\\mathbb{{R}}^{{512}}$}};")

# C: per-class covariances (rank-deficient needles)
xc, wc = boxes["C"]
T.append(f"\\path[panelframe] ({xc + 0.18:.2f},0.64) rectangle ({xc + wc - 0.18:.2f},2.76);")
T.append(f"\\node[tag] at ({xc + 0.18:.2f},2.76) {{$S_c+\\lambda I$}};")
T += panel("class", cx("C"), 1.60, 1.60)
T.append(f"\\node[font={F(6.6)}] at ({cx('C'):.2f},0.36) {{$S_c$: \\textcolor{{muted}}{{rank $\\le K\\ll 512$}}}};")

# D: pooled + shrunk
xd, wd = boxes["D"]
pw = 2.05
p1 = xd + 0.18 + pw / 2
p2 = xd + wd - 0.18 - pw / 2
for p in (p1, p2):
    T.append(f"\\path[panelframe] ({p - pw / 2:.2f},0.64) rectangle ({p + pw / 2:.2f},2.76);")
T.append(f"\\node[tag] at ({p1 - pw / 2:.2f},2.76) {{$S_{{\\mathrm{{pool}}}}$: all residuals}};")
T.append(f"\\node[tag] at ({p2 - pw / 2:.2f},2.76) {{shrunk $\\tilde S_c+\\lambda I$}};")
T += panel("pool", p1, 1.62, 3.1)
T += panel("shrunk", p2, 1.58, 1.60)
T.append(f"\\draw[-{{Stealth[length=3.6pt]}}, ours, line width=0.8pt] ({p1 + pw / 2 + 0.05:.2f},1.70) -- "
         f"node[above, inner sep=1.5pt, font={F(6.8)}] {{$\\alpha$}} ({p2 - pw / 2 - 0.05:.2f},1.70);")
T.append(f"\\node[font={F(6.8)}, text=ours!85!black] at ({cx('D'):.2f},0.36) "
         f"{{$\\tilde S_c=(1-\\alpha)\\,S_c+\\alpha\\,S_{{\\mathrm{{pool}}}}$}};")

# E: hybrid scoring
xe, we = boxes["E"]
lines = [
    r"$s^{\cos}_c=\cos(e_c,p_c)$",
    r"$s^{\mathrm{md}}_c$ with $(\textcolor{ours}{\tilde S_c}+\lambda I)^{-1}$",
    r"$z$-norm over classes, $\Phi(\cdot)$",
    r"$(1{-}w_c)\Phi^{\cos}_c{+}w_c\Phi^{\mathrm{md}}_c$",
]
for i, s in enumerate(lines):
    T.append(f"\\node[anchor=west, font={F(6.7)}] at ({xe + 0.12:.2f},{2.56 - i * 0.47:.2f}) {{{s}}};")
T.append(f"\\node[chip, draw=ours, text=ours, font={F(6.8)}, minimum width=2.3cm] at ({cx('E'):.2f},0.38) "
         f"{{$\\hat y=\\arg\\max_c\\ \\mathrm{{score}}_c$}};")

T += [r"\end{tikzpicture}", r"\end{document}"]
(OUT / "fig_overview.tex").write_text("\n".join(T), encoding="utf-8")
cond = lambda M: round(float(np.linalg.cond(M + LAM * np.eye(2))), 1)
print("cond(S_c+lam):", [cond(S) for *_, S, _ in stats],
      "cond(S_tilde+lam):", [cond((1 - ALPHA) * S + ALPHA * S_pool) for *_, S, _ in stats])
