"""Recipe: the two-line ggplot — mpg displ vs hwy.

The canonical "hello world" of ggplot2 — used as Acceptance Criterion #5
for biobabel.r_translate's exact-match path.
"""

from __future__ import annotations

from pathlib import Path

from ggplot2_py import aes, geom_point, ggplot, ggsave
from ggplot2_py.datasets import mpg


def main(out_path: Path = Path("minimal_scatter.png")) -> Path:
    p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
    ggsave(str(out_path), plot=p, width=6, height=4, dpi=150)
    return out_path


if __name__ == "__main__":
    print(f"wrote {main().resolve()}")
