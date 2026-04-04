"""Result recorder for tutorial validation.

Produces CSV files with the canonical schema expected by
``update_port_status.py --sync-tutorials``:

    tutorial,function,tier,metric,value,threshold,pass
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


# Canonical column names — must match update_port_status.py expectations.
FIELDNAMES = ("tutorial", "function", "tier", "metric", "value", "threshold", "pass")


class ResultRecorder:
    """Record quantitative comparisons between R and Python outputs.

    Parameters
    ----------
    tutorial_name : str
        Tutorial identifier (e.g. ``"ggplot2"``).  Used as the ``tutorial``
        column value and to derive the output path
        ``validation/results_<tutorial_name>.csv``.
    output_dir : str or None
        Directory for the results CSV.  Defaults to the directory containing
        this file.
    """

    def __init__(self, tutorial_name: str, output_dir: str | None = None) -> None:
        self.tutorial_name = tutorial_name
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(
            output_dir, f"results_{tutorial_name}.csv"
        )
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        function: str,
        metric: str,
        *,
        value: Any,
        threshold: Any,
        tier: int = 1,
    ) -> None:
        """Record a single comparison point.

        Parameters
        ----------
        function : str
            Name of the function or comparison checkpoint.
        metric : str
            Comparison metric (``"exact"``, ``"pearson_r"``, ``"numeric"``,
            ``"set_overlap"``, ``"shape"``, …).
        value
            Observed value from the Python implementation.
        threshold
            Acceptance threshold (R reference or numeric bound).
        tier : int
            Validation tier (1 = strict, 2 = moderate, 3 = lenient).
        """
        passed = self._evaluate(metric, value, threshold)
        self.records.append({
            "tutorial": self.tutorial_name,
            "function": function,
            "tier": tier,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "pass": str(passed).lower(),
        })
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {function}: value={value} threshold={threshold} "
              f"({metric}, tier={tier})")

    def log(self, function: str, message: str) -> None:
        """Print an informational message (not recorded in CSV).

        Parameters
        ----------
        function : str
            Context label.
        message : str
            Free-form message.
        """
        print(f"  [INFO] {function}: {message}")

    def save(self) -> None:
        """Write accumulated records to the results CSV."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.records)

    def summary(self) -> str:
        """Return a one-line pass/fail summary."""
        total = len(self.records)
        passed = sum(1 for r in self.records if r["pass"] == "true")
        failed = total - passed
        return f"Validation: {passed}/{total} passed, {failed} failed"

    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(metric: str, value: Any, threshold: Any) -> bool:
        """Determine pass/fail for a single check."""
        if metric == "exact":
            return value == threshold
        if metric in ("numeric", "numeric_abs"):
            try:
                return abs(float(value) - float(threshold)) <= 1e-9
            except (TypeError, ValueError):
                return False
        if metric == "numeric_tol":
            # threshold is (reference, tolerance) tuple
            try:
                ref, tol = threshold
                return abs(float(value) - float(ref)) <= float(tol)
            except (TypeError, ValueError):
                return False
        if metric == "pearson_r":
            try:
                return float(value) >= float(threshold)
            except (TypeError, ValueError):
                return False
        if metric == "set_overlap":
            try:
                return float(value) >= float(threshold)
            except (TypeError, ValueError):
                return False
        if metric == "gte":
            try:
                return float(value) >= float(threshold)
            except (TypeError, ValueError):
                return False
        if metric == "shape":
            return value == threshold
        if metric == "bool":
            return bool(value) == bool(threshold)
        # Fallback: exact comparison
        return value == threshold
