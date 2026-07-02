# Imports ------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
# --------------------------------------------------


OutputFormat = Literal['csv', 'parquet']


def to_camel(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError('Column names must be strings.')

    if not name:
        raise ValueError('Column names must be nonempty strings.')

    parts = name.split('_')

    return parts[0] + ''.join(part.capitalize() for part in parts[1:])


def write_df(dataframe: pd.DataFrame, path: str | Path, file_format: OutputFormat) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('The output object must be a pandas DataFrame.')

    output_path = Path(path)

    if output_path.exists() and output_path.is_dir():
        raise ValueError('The output path must refer to a file, not a directory.')

    if file_format not in {'csv', 'parquet'}:
        raise ValueError('The output format must be either CSV or Parquet.')

    output_path.parent.mkdir(parents = True, exist_ok = True)

    output = dataframe.copy()
    output.columns = [to_camel(str(column)) for column in output.columns]

    if output.columns.duplicated().any():
        raise ValueError('Column names must remain unique after conversion to camel case.')

    if file_format == 'csv':
        output.to_csv(output_path)
    else:
        output.to_parquet(output_path)