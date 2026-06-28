"""Self-contained HTML reports with inline SVG charts.

All SVG is generated with pure string arithmetic — no plotting libraries.
The output is a single .html file that renders offline in any browser.
"""
import os
from typing import Optional

# ── colour palette ────────────────────────────────────────────────────────────
_C0      = "#4e79a7"   # class 0 (blue)
_C1      = "#f28e2b"   # class 1 (orange)
_ACCENT  = "#58a6ff"   # highlight / axis colour
_MUTED   = "#8b949e"
_BG      = "#0d1117"
_CARD    = "#161b22"
_BORDER  = "#30363d"
_TEXT    = "#c9d1d9"


# ── coordinate helpers ────────────────────────────────────────────────────────

def _scale(val: float, vmin: float, vmax: float, out_lo: float, out_hi: float) -> float:
    span = vmax - vmin or 1.0
    return out_lo + (val - vmin) / span * (out_hi - out_lo)


# ── SVG scatter plot ──────────────────────────────────────────────────────────

def _svg_scatter(
    X: list, y: list,
    X_labeled: Optional[list] = None,
    W: int = 380, H: int = 340, pad: int = 30,
) -> str:
    if not X:
        return f'<svg width="{W}" height="{H}"></svg>'

    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    margin = max(max(xs) - min(xs), max(ys) - min(ys), 0.1) * 0.1
    x_lo, x_hi = min(xs) - margin, max(xs) + margin
    y_lo, y_hi = min(ys) - margin, max(ys) + margin

    def tx(v: float) -> float:
        return _scale(v, x_lo, x_hi, pad, W - pad)

    def ty(v: float) -> float:
        return _scale(v, y_lo, y_hi, H - pad - 18, pad)  # flipped, leave room for legend

    # Build lookup of labeled point coordinates for highlighting
    labeled_coords = set()
    if X_labeled:
        for pt in X_labeled:
            labeled_coords.add((pt[0], pt[1]))

    palette = [_C0, _C1]
    circles = []
    for point, label in zip(X, y):
        is_lab = (point[0], point[1]) in labeled_coords
        cx, cy = tx(point[0]), ty(point[1])
        color  = palette[label % 2]
        if is_lab:
            circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                f'fill="{color}" opacity="0.95" stroke="#ffffff" stroke-width="1.2"/>'
            )
        else:
            circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
                f'fill="{color}" opacity="0.30"/>'
            )

    lx = pad
    ly = H - 10
    legend = (
        f'<circle cx="{lx+6}" cy="{ly}" r="4" fill="{_C0}"/>'
        f'<text x="{lx+14}" y="{ly+4}" fill="{_MUTED}" font-size="10">class 0</text>'
        f'<circle cx="{lx+68}" cy="{ly}" r="4" fill="{_C1}"/>'
        f'<text x="{lx+76}" y="{ly+4}" fill="{_MUTED}" font-size="10">class 1</text>'
        f'<circle cx="{lx+132}" cy="{ly}" r="5" fill="{_C0}" stroke="white" stroke-width="1.2"/>'
        f'<text x="{lx+140}" y="{ly+4}" fill="{_MUTED}" font-size="10">labeled</text>'
    )

    return (
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_CARD};border-radius:4px">\n'
        + "\n".join(circles)
        + "\n" + legend
        + "\n</svg>"
    )


# ── SVG learning curve ────────────────────────────────────────────────────────

def _svg_curve(
    history: list,
    W: int = 380, H: int = 300, pad: int = 44,
) -> str:
    if not history:
        return f'<svg width="{W}" height="{H}"></svg>'

    ns   = [h["n_labeled"] for h in history]
    accs = [h["accuracy"]  for h in history]

    def tx(v: float) -> float:
        return _scale(v, min(ns), max(ns), pad, W - pad)

    def ty(v: float) -> float:
        return _scale(v, 0.0, 1.0, H - pad, pad)

    # Horizontal grid lines
    grid = []
    for a in (0.25, 0.5, 0.75, 1.0):
        gy = ty(a)
        grid.append(
            f'<line x1="{pad}" y1="{gy:.1f}" x2="{W - pad}" y2="{gy:.1f}" '
            f'stroke="{_BORDER}" stroke-dasharray="4"/>'
            f'<text x="{pad - 4}" y="{gy + 4:.1f}" fill="{_MUTED}" '
            f'font-size="10" text-anchor="end">{a:.0%}</text>'
        )

    pts_str = " ".join(f"{tx(n):.1f},{ty(a):.1f}" for n, a in zip(ns, accs))
    polyline = (
        f'<polyline points="{pts_str}" fill="none" stroke="{_ACCENT}" '
        f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    dots = [
        f'<circle cx="{tx(n):.1f}" cy="{ty(a):.1f}" r="3.5" fill="{_ACCENT}"/>'
        for n, a in zip(ns, accs)
    ]

    # Axis labels
    cx_mid = (pad + W - pad) / 2
    cy_mid = (pad + H - pad) / 2
    axes = (
        f'<text x="{cx_mid:.0f}" y="{H - 6}" fill="{_MUTED}" font-size="10" '
        f'text-anchor="middle">labeled samples</text>'
        f'<text x="10" y="{cy_mid:.0f}" fill="{_MUTED}" font-size="10" '
        f'text-anchor="middle" '
        f'transform="rotate(-90,10,{cy_mid:.0f})">accuracy</text>'
    )

    return (
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_CARD};border-radius:4px">\n'
        + "\n".join(grid)
        + "\n" + polyline
        + "\n" + "\n".join(dots)
        + "\n" + axes
        + "\n</svg>"
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = f"""
body {{font-family:'Courier New',monospace;max-width:980px;margin:40px auto;
      background:{_BG};color:{_TEXT};padding:0 24px;line-height:1.6}}
h1   {{color:{_ACCENT};margin-bottom:4px;letter-spacing:-0.5px}}
h2   {{color:{_MUTED};font-size:0.85em;text-transform:uppercase;
      letter-spacing:1px;border-bottom:1px solid {_BORDER};
      padding-bottom:4px;margin-top:32px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}}
.card{{background:{_CARD};border:1px solid {_BORDER};border-radius:8px;padding:16px}}
.tag {{display:inline-block;background:#21262d;border:1px solid {_BORDER};
      border-radius:4px;padding:2px 10px;margin:2px;font-size:0.8em;
      color:{_MUTED}}}
.tag b{{color:{_TEXT}}}
table{{border-collapse:collapse;width:100%;margin-top:8px}}
th,td{{border:1px solid {_BORDER};padding:7px 14px;text-align:left}}
th   {{background:#21262d;color:{_ACCENT};font-size:0.85em}}
.val {{color:{_ACCENT};font-weight:bold}}
footer{{margin-top:32px;font-size:0.75em;color:{_MUTED};text-align:center}}
"""


# ── HTML assembly ─────────────────────────────────────────────────────────────

def generate_html(result: dict) -> str:
    """Generate a self-contained HTML report from a run_experiment() result."""
    dataset  = result.get("dataset",  "?")
    strategy = result.get("strategy", "?")
    history  = result.get("history",  [])
    X_train  = result.get("X_train",  [])
    y_train  = result.get("y_train",  [])
    X_lab    = result.get("X_labeled", [])

    from .metrics import summarise
    stats = summarise(history)

    scatter = _svg_scatter(X_train, y_train, X_labeled=X_lab)
    curve   = _svg_curve(history)

    def fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    rows = "".join(
        f"<tr><td>{k.replace('_', ' ')}</td><td class='val'>{fmt(v)}</td></tr>"
        for k, v in stats.items()
    )
    table = f"<table><tr><th>metric</th><th>value</th></tr>{rows}</table>"

    tags = "".join(
        f"<span class='tag'><b>{k}</b>: {v}</span>"
        for k, v in [
            ("dataset",  dataset),
            ("strategy", strategy),
            ("budget",   result.get("budget", "?")),
            ("seed",     result.get("seed",   "?")),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ActiveCanvas Lab — {dataset} × {strategy}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>ActiveCanvas Lab</h1>
  <p>{tags}</p>

  <h2>Training data &amp; labeled subset</h2>
  <div class="grid">
    <div class="card">{scatter}</div>
    <div class="card">{curve}</div>
  </div>

  <h2>Summary statistics</h2>
  <div class="card">{table}</div>

  <footer>generated by <a href="https://github.com/get2salam/active-canvas-lab"
    style="color:{_MUTED}">ActiveCanvas Lab</a></footer>
</body>
</html>"""


def write_report(result: dict, path: str) -> str:
    """Write HTML report to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    html = generate_html(result)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(path)
