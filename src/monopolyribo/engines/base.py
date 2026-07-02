# Imports ------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd
# --------------------------------------------------


@dataclass
class EngineFit:
    name: str
    results: dict[str, pd.DataFrame] = field(default_factory = dict)
    models: dict[str, Any] = field(default_factory = dict)
    diagnostics: pd.DataFrame = field(default_factory = pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory = dict)


class ModelEngine(Protocol):
    name: str

    def fit(self, dataset: Any) -> EngineFit:
        ...