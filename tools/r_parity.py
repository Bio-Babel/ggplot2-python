"""R <-> Python parity harness for the ggplot2-python remediation.

The R installation at ``RSCRIPT`` (ggplot2 4.0.2) is the **gold standard**. This
module runs R snippets and returns their stdout so a Python fix can be checked
against the R reference on the *same* inputs (project principle: both-side
internal-computation verification, not static reading).

Usage
-----
    import sys; sys.path.insert(0, "tools")
    from r_parity import run_r, py_layout

    ref = run_r('cat(as.character(calc_element("plot.background", theme_grey())$fill))',
                fns=("calc_element",))
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from typing import Iterable, List, Tuple

RSCRIPT = "/home/groups/xiaojie/nianping/Conda_Files/envs/ggrepel-dev/bin/Rscript"


def run_r(
    snippet: str,
    *,
    library: str = "ggplot2",
    fns: Iterable[str] = (),
    timeout: int = 120,
) -> str:
    """Run an R *snippet* with ``library(<library>)`` preloaded; return stdout.

    Parameters
    ----------
    snippet : str
        R code. Use ``cat()`` to emit comparable values.
    library : str
        Package to ``library()`` first (default ``"ggplot2"``).
    fns : iterable of str
        Names of *internal* (unexported) ggplot2 functions to bind into scope via
        ``getFromNamespace(name, "ggplot2")`` (e.g. ``calc_element``,
        ``validate_guide``) so the snippet can call them directly.
    timeout : int
        Seconds before aborting.

    Returns
    -------
    str
        The snippet's stdout.

    Raises
    ------
    RuntimeError
        If R exits non-zero (message includes stderr and the snippet).
    """
    preamble = [f"suppressMessages(library({library}))"]
    for fn in fns:
        preamble.append(f'{fn} <- getFromNamespace("{fn}", "{library}")')
    body = "\n".join(preamble) + "\n" + textwrap.dedent(snippet) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        res = subprocess.run(
            [RSCRIPT, path], capture_output=True, text=True, timeout=timeout
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"R exited {res.returncode}\n--- stderr ---\n{res.stderr}\n"
                f"--- snippet ---\n{body}"
            )
        return res.stdout
    finally:
        os.unlink(path)


def py_layout(grob) -> List[Tuple[str, int, int, int, int]]:
    """Return a gtable's layout as sorted ``(name, t, l, b, r)`` tuples.

    Works with both ``gtable_py.Gtable`` (``.layout`` is a dict of parallel lists)
    and a pandas-DataFrame layout. Use to compare a Python gtable's cell topology
    against an R ``gt$layout`` dump.
    """
    lay = grob.layout
    cols = ("name", "t", "l", "b", "r")
    if hasattr(lay, "columns"):  # pandas DataFrame
        seqs = [list(lay[c]) for c in cols]
    else:  # dict of parallel lists
        seqs = [list(lay[c]) for c in cols]
    rows = [
        (str(n), int(t), int(l), int(b), int(r))
        for n, t, l, b, r in zip(*seqs)
    ]
    return sorted(rows)
