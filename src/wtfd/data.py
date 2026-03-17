"""
wtfd.data
=========

Data access and dataset-construction utilities for the Wind Turbine Fault
Detection (WTFD) project.

This module centralizes the logic used to:

1. Discover the raw dataset structure across wind farms and turbine files.
2. Load raw SCADA turbine CSV files and accompanying metadata files
   (e.g., feature descriptions and event information).
3. Inspect dataset schemas and row counts without unnecessarily loading
   large files into memory.
4. Convert raw turbine CSV files into optimized Parquet files for faster,
   safer downstream analysis.
5. Generate metadata summary tables that support reproducibility and
   later preprocessing, feature engineering, and modeling work.

Design Notes
------------
- Raw data should be treated as read-only and left unchanged.
- This module is intentionally focused on I/O and dataset organization.
- Cleaning, gap handling, and feature engineering should be handled in
  separate modules/notebooks.
- For large SCADA datasets, converting to per-turbine Parquet files helps
  reduce notebook memory pressure and improves load times.

Typical Usage
-------------
>>> from pathlib import Path
>>> from wtfd.data import convert_scada_csvs_to_parquet_parts
>>> raw_dir = Path("../data/raw/zenodo_windfarm_data")
>>> out_dir = Path("../data/interim/scada_by_turbine")
>>> metadata_df = convert_scada_csvs_to_parquet_parts(
...     raw_data_dir=raw_dir,
...     output_dir=out_dir,
...     sep=";",
...     timestamp_col="time_stamp",
... )
"""

from __future__ import annotations

from pathlib import Path
import gc
from typing import Iterable
import pandas as pd

from wtfd.preprocessing import normalize_column_name, optimize_dtypes


def get_farm_dirs(raw_data_dir: str | Path) -> list[Path]:
    """
    Return all wind farm directories found under the raw dataset directory.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the top-level raw dataset directory. This directory is
        expected to contain subdirectories such as ``Wind Farm A``,
        ``Wind Farm B``, and ``Wind Farm C``.

    Returns
    -------
    list[pathlib.Path]
        A sorted list of wind farm directory paths.

    Notes
    -----
    This function assumes the dataset uses the naming pattern
    ``Wind Farm *`` for farm folders.
    """
    raw_data_dir = Path(raw_data_dir)
    return sorted(raw_data_dir.glob("Wind Farm *"))


def get_turbine_files(farm_dir: str | Path) -> list[Path]:
    """
    Return all turbine dataset CSV files for a given wind farm.

    Parameters
    ----------
    farm_dir : str or pathlib.Path
        Path to a wind farm directory containing a ``datasets`` subdirectory.

    Returns
    -------
    list[pathlib.Path]
        A sorted list of turbine CSV file paths.

    Notes
    -----
    The expected structure is:

    ``<farm_dir>/datasets/*.csv``
    """
    farm_dir = Path(farm_dir)
    dataset_dir = farm_dir / "datasets"
    return sorted(dataset_dir.glob("*.csv"))


def count_rows_fast(file_path: str | Path, encoding: str = "utf-8") -> int:
    """
    Count the number of data rows in a CSV file without fully loading it.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file.
    encoding : str, optional
        File encoding to use when opening the file, by default ``"utf-8"``.

    Returns
    -------
    int
        Number of data rows in the file, excluding the header row.

    Notes
    -----
    This function is useful for building row-count summaries across many
    turbine files while minimizing memory usage.

    It assumes:
    - the file has exactly one header row
    - the file is text-readable with the provided encoding
    """
    file_path = Path(file_path)
    with open(file_path, "r", encoding=encoding) as f:
        return max(sum(1 for _ in f) - 1, 0)


def get_file_columns(file_path: str | Path, sep: str = ";") -> list[str]:
    """
    Read only the header row of a CSV file and return normalized column names.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file.
    sep : str, optional
        Delimiter used in the CSV file, by default ``";"``.

    Returns
    -------
    list[str]
        A list of normalized column names.

    Notes
    -----
    Normalization is delegated to ``wtfd.preprocessing.normalize_column_name``.
    This function is particularly useful for schema comparison across files.
    """
    file_path = Path(file_path)
    df0 = pd.read_csv(file_path, sep=sep, nrows=0)
    return [normalize_column_name(col) for col in df0.columns]


def discover_dataset_structure(raw_data_dir: str | Path) -> pd.DataFrame:
    """
    Build a lightweight summary of the raw dataset directory structure.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the top-level raw dataset directory.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per wind farm and the following columns:

        - ``wind_farm``: original farm directory name
        - ``wind_farm_id``: normalized farm identifier
        - ``num_turbine_files``: number of turbine CSV files found
        - ``has_feature_description``: whether ``feature_description.csv`` exists
        - ``has_event_info``: whether ``event_info.csv`` exists

    Notes
    -----
    This function is useful for early-stage EDA and dataset validation.
    """
    rows: list[dict[str, object]] = []

    for farm_dir in get_farm_dirs(raw_data_dir):
        rows.append(
            {
                "wind_farm": farm_dir.name,
                "wind_farm_id": normalize_column_name(farm_dir.name),
                "num_turbine_files": len(get_turbine_files(farm_dir)),
                "has_feature_description": (farm_dir / "feature_description.csv").exists(),
                "has_event_info": (farm_dir / "event_info.csv").exists(),
            }
        )

    return pd.DataFrame(rows)


def load_turbine_csv(
    file_path: str | Path,
    sep: str = ";",
    normalize_columns: bool = True,
    timestamp_col: str | None = None,
    optimize_memory: bool = False,
) -> pd.DataFrame:
    """
    Load a single turbine SCADA CSV file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the turbine CSV file.
    sep : str, optional
        Delimiter used in the CSV file, by default ``";"``.
    normalize_columns : bool, optional
        If True, normalize column names using
        ``wtfd.preprocessing.normalize_column_name``, by default True.
    timestamp_col : str or None, optional
        Name of the timestamp column to parse into pandas datetime format.
        If None, no timestamp parsing is applied, by default None.
    optimize_memory : bool, optional
        If True, downcast numeric columns using
        ``wtfd.preprocessing.optimize_dtypes``, by default False.

    Returns
    -------
    pandas.DataFrame
        Loaded turbine dataframe.

    Notes
    -----
    This function does not add metadata columns such as ``wind_farm`` or
    ``turbine_id``. Those are typically added during dataset combination.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep=sep)

    if normalize_columns:
        df.columns = [normalize_column_name(col) for col in df.columns]

    if timestamp_col is not None and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    if optimize_memory:
        df = optimize_dtypes(df)

    return df


def load_feature_description(
    farm_dir: str | Path,
    sep: str = ";",
    normalize_columns: bool = True,
) -> pd.DataFrame:
    """
    Load the ``feature_description.csv`` file for a wind farm.

    Parameters
    ----------
    farm_dir : str or pathlib.Path
        Path to the wind farm directory.
    sep : str, optional
        Delimiter used in the CSV file, by default ``";"``.
    normalize_columns : bool, optional
        If True, normalize column names using
        ``wtfd.preprocessing.normalize_column_name``, by default True.

    Returns
    -------
    pandas.DataFrame
        Feature description dataframe for the wind farm.

    Notes
    -----
    This file is critical for understanding the physical meaning of turbine
    signals and for building farm-specific feature mappings.
    """
    farm_dir = Path(farm_dir)
    file_path = farm_dir / "feature_description.csv"

    df = pd.read_csv(file_path, sep=sep)

    if normalize_columns:
        df.columns = [normalize_column_name(col) for col in df.columns]

    return df


def load_event_info(
    farm_dir: str | Path,
    sep: str = ";",
    normalize_columns: bool = True,
) -> pd.DataFrame:
    """
    Load the ``event_info.csv`` file for a wind farm.

    Parameters
    ----------
    farm_dir : str or pathlib.Path
        Path to the wind farm directory.
    sep : str, optional
        Delimiter used in the CSV file, by default ``";"``.
    normalize_columns : bool, optional
        If True, normalize column names using
        ``wtfd.preprocessing.normalize_column_name``, by default True.

    Returns
    -------
    pandas.DataFrame
        Event information dataframe for the wind farm.

    Notes
    -----
    This file is expected to contain turbine event annotations and will later
    support label construction for fault detection.
    """
    farm_dir = Path(farm_dir)
    file_path = farm_dir / "event_info.csv"

    df = pd.read_csv(file_path, sep=sep)

    if normalize_columns:
        df.columns = [normalize_column_name(col) for col in df.columns]

    return df


def iter_turbine_records(raw_data_dir: str | Path) -> Iterable[dict[str, object]]:
    """
    Yield metadata records describing each turbine CSV file in the raw dataset.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the top-level raw dataset directory.

    Yields
    ------
    dict[str, object]
        Dictionary containing:

        - ``wind_farm``: original wind farm name
        - ``wind_farm_id``: normalized wind farm identifier
        - ``turbine_id``: turbine file stem
        - ``source_csv``: path to the turbine CSV file

    Notes
    -----
    This generator is helpful when building summary tables or processing files
    sequentially without holding all metadata in memory at once.
    """
    for farm_dir in get_farm_dirs(raw_data_dir):
        farm_name = farm_dir.name
        farm_id = normalize_column_name(farm_name)

        for file_path in get_turbine_files(farm_dir):
            yield {
                "wind_farm": farm_name,
                "wind_farm_id": farm_id,
                "turbine_id": file_path.stem,
                "source_csv": file_path,
            }


def build_turbine_metadata_summary(
    raw_data_dir: str | Path,
    sep: str = ";",
    timestamp_col: str = "time_stamp",
) -> pd.DataFrame:
    """
    Build a metadata summary dataframe for all turbine CSV files.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the top-level raw dataset directory.
    sep : str, optional
        Delimiter used in the CSV files, by default ``";"``.
    timestamp_col : str, optional
        Timestamp column name to inspect for date coverage,
        by default ``"time_stamp"``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per turbine file and columns including:

        - ``wind_farm``
        - ``wind_farm_id``
        - ``turbine_id``
        - ``source_csv``
        - ``num_rows``
        - ``num_columns``
        - ``min_timestamp``
        - ``max_timestamp``

    Notes
    -----
    This function reads each turbine file once. It is useful for EDA and for
    documenting dataset coverage across farms and turbines.
    """
    rows: list[dict[str, object]] = []

    for record in iter_turbine_records(raw_data_dir):
        source_csv = Path(record["source_csv"])

        df = load_turbine_csv(
            source_csv,
            sep=sep,
            normalize_columns=True,
            timestamp_col=timestamp_col,
            optimize_memory=False,
        )

        rows.append(
            {
                "wind_farm": record["wind_farm"],
                "wind_farm_id": record["wind_farm_id"],
                "turbine_id": record["turbine_id"],
                "source_csv": str(source_csv),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "min_timestamp": df[timestamp_col].min() if timestamp_col in df.columns else pd.NaT,
                "max_timestamp": df[timestamp_col].max() if timestamp_col in df.columns else pd.NaT,
            }
        )

        del df
        gc.collect()

    return pd.DataFrame(rows)


def convert_scada_csvs_to_parquet_parts(
    raw_data_dir: str | Path,
    output_dir: str | Path,
    sep: str = ";",
    timestamp_col: str = "time_stamp",
    normalize_columns: bool = True,
    optimize_memory: bool = True,
    metadata_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Convert each turbine SCADA CSV file into an individual Parquet file.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the top-level raw dataset directory.
    output_dir : str or pathlib.Path
        Directory where Parquet files will be written.
    sep : str, optional
        Delimiter used in the CSV files, by default ``";"``.
    timestamp_col : str, optional
        Timestamp column name to parse into datetime format,
        by default ``"time_stamp"``.
    normalize_columns : bool, optional
        If True, normalize column names before saving, by default True.
    optimize_memory : bool, optional
        If True, downcast numeric dtypes before writing, by default True.
    metadata_csv_path : str or pathlib.Path or None, optional
        If provided, write the returned metadata dataframe to this CSV path,
        by default None.

    Returns
    -------
    pandas.DataFrame
        Metadata summary dataframe with one row per turbine Parquet file.

    Side Effects
    ------------
    - Writes one Parquet file per turbine dataset into ``output_dir``.
    - Optionally writes a metadata summary CSV if ``metadata_csv_path`` is set.

    Notes
    -----
    This function is designed to avoid notebook memory issues by processing
    one turbine file at a time rather than concatenating all SCADA data into a
    single in-memory dataframe.

    Each written Parquet file includes the original turbine data plus:
    - ``wind_farm``
    - ``turbine_id``

    The output filename pattern is:

    ``<wind_farm_id>_<turbine_id>.parquet``
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []

    for record in iter_turbine_records(raw_data_dir):
        source_csv = Path(record["source_csv"])
        wind_farm = str(record["wind_farm"])
        wind_farm_id = str(record["wind_farm_id"])
        turbine_id = str(record["turbine_id"])

        df = load_turbine_csv(
            source_csv,
            sep=sep,
            normalize_columns=normalize_columns,
            timestamp_col=timestamp_col,
            optimize_memory=optimize_memory,
        )

        df["wind_farm"] = wind_farm_id
        df["turbine_id"] = turbine_id

        parquet_path = output_dir / f"{wind_farm_id}_{turbine_id}.parquet"
        df.to_parquet(parquet_path, index=False)

        metadata_rows.append(
            {
                "wind_farm": wind_farm,
                "wind_farm_id": wind_farm_id,
                "turbine_id": turbine_id,
                "source_csv": str(source_csv),
                "parquet_part": str(parquet_path),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "min_timestamp": df[timestamp_col].min() if timestamp_col in df.columns else pd.NaT,
                "max_timestamp": df[timestamp_col].max() if timestamp_col in df.columns else pd.NaT,
            }
        )

        del df
        gc.collect()

    metadata_df = pd.DataFrame(metadata_rows)

    if metadata_csv_path is not None:
        metadata_csv_path = Path(metadata_csv_path)
        metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_csv_path, index=False)

    return metadata_df