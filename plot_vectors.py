"""Plot motion vectors from vectors.md (angle rad, magnitude px).

By default plots only the cleaned '# hummingbird at feeder' section,
one series per track ID (## headings).
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

VECTORS_PATH = Path(__file__).with_name("vectors.md")
LINE_RE = re.compile(r"^([^<\s]+)\s*<([^,]+),([^>]+)>")
SECTION_START = "hummingbird at feeder"


def parse_track_series(path: Path, section_substr: str = SECTION_START) -> dict[str, np.ndarray]:
    """Parse track IDs under the matching # section (## labels preferred)."""
    series: dict[str, list[tuple[float, float]]] = {}
    in_section = False
    current_track: str | None = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#") and not line.startswith("##"):
            heading = line.lstrip("#").strip().lower()
            in_section = section_substr.lower() in heading
            current_track = None
            continue

        if not in_section:
            continue

        if line.startswith("##"):
            current_track = line.lstrip("#").strip().split()[0]
            series.setdefault(current_track, [])
            continue

        match = LINE_RE.match(line)
        if not match:
            continue
        tid = match.group(1)
        angle = float(match.group(2))
        magnitude = float(match.group(3))
        if np.isnan(angle) or np.isnan(magnitude):
            continue
        key = current_track or tid
        series.setdefault(key, []).append((angle, magnitude))

    return {label: np.array(pts) for label, pts in series.items() if len(pts)}


def main() -> None:
    data = parse_track_series(VECTORS_PATH)
    if not data:
        raise SystemExit(f"No tracks found under '{SECTION_START}' in {VECTORS_PATH}")

    n = len(data)
    cmap = plt.colormaps["tab20"].resampled(max(n, 1))
    colors = [cmap(i) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    for (label, pts), color in zip(data.items(), colors):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            alpha=0.75,
            label=f"{label} (n={len(pts)})",
            color=color,
            edgecolors="none",
        )
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Magnitude (px)")
    ax.set_title("Angle vs magnitude")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    for (label, pts), color in zip(data.items(), colors):
        ax.plot(
            np.arange(len(pts)),
            pts[:, 1],
            marker="o",
            markersize=3,
            linewidth=1.1,
            alpha=0.85,
            label=f"{label} (n={len(pts)})",
            color=color,
        )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Magnitude (px)")
    ax.set_title("Magnitude over time")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"Motion vectors — {SECTION_START} ({n} tracks)", fontsize=13)
    out = Path(__file__).with_name("vectors_plot.png")
    fig.savefig(out, dpi=150)
    print(f"Plotted {n} tracks ({sum(len(v) for v in data.values())} points)")
    for label, pts in data.items():
        print(f"  {label}: n={len(pts)}")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
