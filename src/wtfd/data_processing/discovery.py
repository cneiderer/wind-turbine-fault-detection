"""
wtfd.data
=========

Data loading and file discovery utilities for the wind turbine fault detection
project.

This module is responsible for:
- discovering wind farm directories
- discovering turbine CSV files
- loading SCADA CSV files
- loading feature description metadata
- loading event information metadata

Design Notes
------------
- All dataset CSV files are assumed to be semicolon-delimited.
- Functions in this module should remain lightweight and reusable.
- Memory-conscious defaults are used for large SCADA files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


def find_wind_farm_directories(
    raw_data_dir: str | Path,
    require_datasets_subdir: bool = False,
) -> list[Path]:
    """
    Discover wind farm directories within the raw data directory.

    Parameters
    ----------
    raw_data_dir : str | Path
        Path to the root raw data directory.
    require_datasets_subdir : bool, optional
        If True, only include directories that contain a "datasets"
        subdirectory.

    Returns
    -------
    list[Path]
        Sorted list of wind farm directory paths.
    """
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    if not raw_data_dir.is_dir():
        raise ValueError(f"Provided path is not a directory: {raw_data_dir}")

    wind_farm_dirs: list[Path] = []

    for path in raw_data_dir.iterdir():
        if not path.is_dir():
            continue

        if require_datasets_subdir and not (path / "datasets").is_dir():
            continue

        wind_farm_dirs.append(path)

    wind_farm_dirs = sorted(wind_farm_dirs, key=lambda p: p.name)

    if not wind_farm_dirs:
        raise ValueError(f"No wind farm directories found in: {raw_data_dir}")

    return wind_farm_dirs


def find_turbine_csv_files(
    wind_farm_dir: str | Path,
    datasets_subdir: str = "datasets",
    recursive: bool = False,
) -> list[Path]:
    """
    Discover turbine CSV files within a wind farm directory.

    Parameters
    ----------
    wind_farm_dir : str | Path
        Path to a wind farm directory.
    datasets_subdir : str, optional
        Name of the subdirectory containing turbine datasets.
    recursive : bool, optional
        Whether to search recursively within the datasets directory.

    Returns
    -------
    list[Path]
        Sorted list of turbine CSV file paths.
    """
    wind_farm_dir = Path(wind_farm_dir)

    if not wind_farm_dir.exists():
        raise FileNotFoundError(f"Wind farm directory does not exist: {wind_farm_dir}")

    if not wind_farm_dir.is_dir():
        raise ValueError(f"Provided path is not a directory: {wind_farm_dir}")

    datasets_dir = wind_farm_dir / datasets_subdir

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets subdirectory not found: {datasets_dir}")

    if not datasets_dir.is_dir():
        raise ValueError(f"Datasets path is not a directory: {datasets_dir}")

    csv_files = list(datasets_dir.rglob("*.csv")) if recursive else list(datasets_dir.glob("*.csv"))
    csv_files = sorted(csv_files, key=lambda p: p.name)

    if not csv_files:
        raise ValueError(f"No CSV files found in datasets directory: {datasets_dir}")

    return csv_files


def load_scada_csv(
    csv_path: str | Path,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    memory_map: bool = True,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load a turbine SCADA CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the turbine CSV file.
    nrows : int | None, optional
        Number of rows to read. Useful for lightweight EDA sampling.
    usecols : list[str] | None, optional
        Optional subset of columns to load.
    memory_map : bool, optional
        Whether to memory-map the file when possible.
    low_memory : bool, optional
        Passed through to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded SCADA dataframe.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"SCADA CSV file not found: {csv_path}")

    return pd.read_csv(
        csv_path,
        sep=";",
        nrows=nrows,
        usecols=usecols,
        memory_map=memory_map,
        low_memory=low_memory,
    )


def iter_scada_csv_chunks(
    csv_path: str | Path,
    chunksize: int = 100_000,
    usecols: list[str] | None = None,
    memory_map: bool = True,
    low_memory: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Iterate over a SCADA CSV file in chunks.

    Parameters
    ----------
    csv_path : str | Path
        Path to the turbine CSV file.
    chunksize : int, optional
        Number of rows per chunk.
    usecols : list[str] | None, optional
        Optional subset of columns to load.
    memory_map : bool, optional
        Whether to memory-map the file when possible.
    low_memory : bool, optional
        Passed through to pandas.read_csv.

    Yields
    ------
    pd.DataFrame
        Chunk of the SCADA dataframe.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"SCADA CSV file not found: {csv_path}")

    yield from pd.read_csv(
        csv_path,
        sep=";",
        chunksize=chunksize,
        usecols=usecols,
        memory_map=memory_map,
        low_memory=low_memory,
    )


def load_feature_description(
    wind_farm_dir: str | Path,
    filename: str = "feature_description.csv",
) -> pd.DataFrame:
    """
    Load the feature description metadata for a wind farm.

    Parameters
    ----------
    wind_farm_dir : str | Path
        Path to the wind farm directory.
    filename : str, optional
        Name of the feature description file.

    Returns
    -------
    pd.DataFrame
        Loaded feature description table.
    """
    path = Path(wind_farm_dir) / filename

    if not path.exists():
        raise FileNotFoundError(f"Feature description file not found: {path}")

    return pd.read_csv(path, sep=";", low_memory=False)


def load_event_info(
    wind_farm_dir: str | Path,
    filename: str = "event_info.csv",
) -> pd.DataFrame:
    """
    Load the event information metadata for a wind farm.

    Parameters
    ----------
    wind_farm_dir : str | Path
        Path to the wind farm directory.
    filename : str, optional
        Name of the event info file.

    Returns
    -------
    pd.DataFrame
        Loaded event information table.
    """
    path = Path(wind_farm_dir) / filename

    if not path.exists():
        raise FileNotFoundError(f"Event info file not found: {path}")

    return pd.read_csv(path, sep=";", low_memory=False)