# Imports ------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_df
# --------------------------------------------------


@dataclass
class MonoPolyResult:
    abundance: pd.DataFrame
    fraction_effects: dict[str, pd.DataFrame]
    redistribution: pd.DataFrame
    allocation: pd.DataFrame
    joint: pd.DataFrame
    integrated: pd.DataFrame
    classification: pd.DataFrame
    stability: pd.DataFrame
    diagnostics: pd.DataFrame
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_result_table('abundance', self.abundance)
        _validate_result_table('redistribution', self.redistribution)
        _validate_result_table('allocation', self.allocation)
        _validate_result_table('joint', self.joint)
        _validate_result_table('integrated', self.integrated)
        _validate_result_table('classification', self.classification)
        _validate_result_table('stability', self.stability)
        _validate_result_table('diagnostics', self.diagnostics)

        if not isinstance(self.fraction_effects, dict):
            raise TypeError('Fraction effects must be provided as a dictionary.')

        for fraction, results in self.fraction_effects.items():
            if not isinstance(fraction, str) or not fraction:
                raise ValueError('Fraction effect names must be nonempty strings.')

            _validate_result_table(f'fraction_effects_{fraction}', results)

        if not isinstance(self.metadata, dict):
            raise TypeError('Result metadata must be provided as a dictionary.')

    def __getitem__(self, key: str) -> pd.DataFrame:
        try:
            return self.dataframes[key]
        except KeyError as exc:
            raise KeyError(
                f'Unknown result table {key!r}. Available tables are {sorted(self.dataframes)}.'
            ) from exc

    @property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        tables = {
            'abundance': self.abundance,
            'redistribution': self.redistribution,
            'allocation': self.allocation,
            'joint': self.joint,
            'integrated': self.integrated,
            'classification': self.classification,
            'stability': self.stability,
            'diagnostics': self.diagnostics
        }

        for fraction, results in self.fraction_effects.items():
            table_name = f'fraction_effects_{fraction}'

            if table_name in tables:
                raise ValueError(f'Duplicate result table name {table_name!r}.')

            tables[table_name] = results

        return tables

    def write_csv(self, directory: str | Path) -> None:
        self._write_tables(directory, 'csv')

    def write_parquet(self, directory: str | Path) -> None:
        self._write_tables(directory, 'parquet')

    def _write_tables(self, directory: str | Path, file_format: str) -> None:
        output_directory = Path(directory)
        output_directory.mkdir(parents = True, exist_ok = True)

        for name, results in self.dataframes.items():
            output_path = output_directory / f'{name}.{file_format}'
            write_df(results, str(output_path), file_format)


def _validate_result_table(name: str, results: pd.DataFrame) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError(f'Result table {name!r} must be a pandas DataFrame.')