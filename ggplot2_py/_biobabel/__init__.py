"""biobabel manifest factory.

Loads schema-v1 YAML contract files and validates them via Pydantic.
Wired into pyproject.toml as the `biobabel.manifest` entry point.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from biobabel.manifest_api import PackageManifest

_HERE = Path(__file__).parent


def get_manifest() -> PackageManifest:
    data = yaml.safe_load((_HERE / "package.yaml").read_text(encoding="utf-8")) or {}

    for subdir, field in (
        ("symbols", "symbols"),
        ("workflows", "workflows"),
        ("templates", "templates"),
        ("concepts", "concepts"),
        ("idioms", "idioms"),
        ("anti_patterns", "anti_patterns"),
        ("compositions", "compositions"),
    ):
        items = list(data.get(field, []) or [])
        for yfile in sorted((_HERE / subdir).glob("*.yaml")):
            loaded = yaml.safe_load(yfile.read_text(encoding="utf-8"))
            if loaded is None:
                continue
            if isinstance(loaded, list):
                items.extend(loaded)
            else:
                items.append(loaded)
        if items:
            data[field] = items

    return PackageManifest.model_validate(data)
