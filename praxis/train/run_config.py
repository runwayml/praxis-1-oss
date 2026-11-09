"""The YAML run config of the fine-tune scripts.

``scripts/finetune.py``, ``scripts/compute_norm_stats.py`` and
``scripts/init_heads.py`` read the same file. ``--set key=value`` changes one
value: ``--set train.steps=100``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


class ConfigLoader(yaml.SafeLoader):
    """The YAML loader of the run config. It reads ``1e-4`` as a number.

    YAML 1.1 reads a number with an exponent and without a dot as a string.
    """


ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9]+(\.[0-9]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def load_run_config(path: str | Path, overrides: list[str]) -> dict:
    """Read the run config at ``path`` and apply the ``key=value`` overrides."""
    cfg = yaml.load(Path(path).read_text(), Loader=ConfigLoader)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.load(value, Loader=ConfigLoader)
    return cfg
