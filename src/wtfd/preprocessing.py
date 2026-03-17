"""
wtfd.preprocessing
==================

Preprocessing utilities for the wind turbine fault detection project.

This module contains lightweight, reusable preprocessing functions intended
for early-stage data ingestion and dataset standardization. The functions
defined here are appropriate for use during exploratory analysis and raw
data combination, where the goal is to create a consistent structural format
without performing downstream cleaning or feature engineering.

Design Principles
-----------------
1. Preserve raw information as much as possible.
2. Do not assume that sensor names are semantically consistent across farms.
3. Do not remove, impute, or engineer values in this module.
4. Standardize only the structural representation of the data so that later
   notebooks and modules can operate on a predictable schema.

Typical Responsibilities
------------------------
- Normalize column names.
- Detect and standardize timestamp columns.
- Add metadata such as wind farm and turbine identifiers.
- Reorder columns into a consistent layout.
- Remove duplicate column labels if they arise during ingestion.
- Provide helper utilities for schema inspection and preprocessing summaries.

This module should be used by notebooks such as:
- 01_exploratory_data_analysis.ipynb
- 02_data_combination.ipynb

It should NOT be responsible for:
- missing-value imputation
- outlier treatment
- time-gap repair
- target engineering
- feature engineering for modeling
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Column name preprocessing
# ---------------------------------------------------------------------------

def normalize_column_name(name: object) -> str:
    """
    Normalize a single column name into a clean, analysis-friendly format.

    The normalization procedure is intentionally conservative:
    - converts to string
    - strips leading/trailing whitespace
    - lowercases text
    - replaces spaces and punctuation with underscores
    - collapses repeated underscores
    - removes leading/trailing underscores

    Parameters
    ----------
    name : object
        Original column name.

    Returns
    -------
    str
        Normalized column name.

    Examples
    --------
    >>> normalize_column_name(" Active Power ")
    'active_power'

    >>> normalize_column_name("Wind-Speed (m/s)")
    'wind_speed_m_s'
    """
    normalized = str(name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of a dataframe with normalized column names.

    This function standardizes the dataframe's column labels using
    `normalize_column_name`. It does not alter row values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with normalized column names.
    """
    df_out = df.copy()
    df_out.columns = [normalize_column_name(col) for col in df_out.columns]
    return df_out


def deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column names are unique by appending numeric suffixes when needed.

    If duplicate column labels exist, this function renames the later
    occurrences using a suffix pattern such as:
    - sensor_1
    - sensor_1_2
    - sensor_1_3

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with unique column names.
    """
    df_out = df.copy()

    seen: dict[str, int] = {}
    new_columns: list[str] = []

    for col in df_out.columns:
        col_str = str(col)
        if col_str not in seen:
            seen[col_str] = 1
            new_columns.append(col_str)
        else:
            seen[col_str] += 1
            new_columns.append(f"{col_str}_{seen[col_str]}")

    df_out.columns = new_columns
    return df_out


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def find_timestamp_column(
    df: pd.DataFrame,
    candidates: Iterable[str] | None = None,
) -> str | None:
    """
    Attempt to identify the timestamp column in a dataframe.

    The function first checks a list of candidate column names. If no exact
    candidate is found, it searches for common time-related patterns such as:
    - timestamp
    - time
    - datetime
    - date

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    candidates : Iterable[str] | None, optional
        Preferred candidate names to check first. If omitted, a default list
        of common timestamp column names is used.

    Returns
    -------
    str | None
        Name of the detected timestamp column, or None if no plausible match
        is found.
    """
    if candidates is None:
        candidates = (
            "timestamp",
            "datetime",
            "date_time",
            "time",
            "date",
        )

    columns = list(df.columns)
    column_set = {str(col).lower(): col for col in columns}

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in column_set:
            return str(column_set[candidate_lower])

    pattern = re.compile(r"(timestamp|datetime|date_time|time|date)")
    for col in columns:
        if pattern.search(str(col).lower()):
            return str(col)

    return None


def standardize_timestamp_column(
    df: pd.DataFrame,
    timestamp_col: str | None = None,
    output_col: str = "timestamp",
    utc: bool = False,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Standardize the dataframe's timestamp column into pandas datetime format.

    If `timestamp_col` is not provided, the function attempts to detect a
    timestamp column automatically using `find_timestamp_column`.

    The resulting standardized datetime column is written to `output_col`.
    If `timestamp_col` differs from `output_col`, the original column is
    preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    timestamp_col : str | None, optional
        Name of the source timestamp column. If None, the function attempts
        automatic detection.
    output_col : str, optional
        Name of the standardized datetime column. Default is "timestamp".
    utc : bool, optional
        Whether to parse timestamps in UTC. Default is False.
    dayfirst : bool, optional
        Whether to interpret ambiguous dates with day first. Default is False.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with a standardized datetime column.

    Raises
    ------
    ValueError
        If no timestamp column can be identified.
    """
    df_out = df.copy()

    if timestamp_col is None:
        timestamp_col = find_timestamp_column(df_out)

    if timestamp_col is None or timestamp_col not in df_out.columns:
        raise ValueError("No timestamp column could be identified in the dataframe.")

    df_out[output_col] = pd.to_datetime(
        df_out[timestamp_col],
        errors="coerce",
        utc=utc,
        dayfirst=dayfirst,
    )

    return df_out


def sort_by_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Sort a dataframe by its timestamp column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    timestamp_col : str, optional
        Name of the timestamp column. Default is "timestamp".

    Returns
    -------
    pandas.DataFrame
        Sorted copy of the dataframe.

    Raises
    ------
    KeyError
        If the timestamp column does not exist.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Column '{timestamp_col}' not found in dataframe.")

    return df.sort_values(by=timestamp_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def add_turbine_metadata(
    df: pd.DataFrame,
    wind_farm: str,
    turbine_id: str,
    source_file: str | None = None,
) -> pd.DataFrame:
    """
    Add turbine-level metadata columns to a dataframe.

    This function is intended for raw dataset combination so that each row
    can always be traced back to its source turbine and wind farm.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    wind_farm : str
        Name of the wind farm.
    turbine_id : str
        Unique turbine identifier.
    source_file : str | None, optional
        Original source filename or relative path, if available.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with metadata columns inserted at the front.
    """
    df_out = df.copy()

    insert_position = 0
    df_out.insert(insert_position, "wind_farm", wind_farm)
    insert_position += 1

    df_out.insert(insert_position, "turbine_id", turbine_id)
    insert_position += 1

    if source_file is not None:
        df_out.insert(insert_position, "source_file", source_file)

    return df_out


def reorder_core_columns(
    df: pd.DataFrame,
    first_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Reorder a dataframe so key identifier columns appear first.

    Any columns in `first_columns` that are present in the dataframe are moved
    to the front in the given order. Remaining columns follow afterward in
    their existing order.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    first_columns : Iterable[str] | None, optional
        Columns to place first. If omitted, a default ordering is used.

    Returns
    -------
    pandas.DataFrame
        Reordered copy of the dataframe.
    """
    if first_columns is None:
        first_columns = ["wind_farm", "turbine_id", "source_file", "timestamp"]

    present_first = [col for col in first_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in present_first]

    return df.loc[:, present_first + remaining].copy()


# ---------------------------------------------------------------------------
# End-to-end early-stage preprocessing
# ---------------------------------------------------------------------------

def preprocess_scada_dataframe(
    df: pd.DataFrame,
    wind_farm: str,
    turbine_id: str,
    source_file: str | None = None,
    timestamp_col: str | None = None,
    normalize_cols: bool = True,
    deduplicate_cols: bool = True,
    standardize_timestamp: bool = True,
    sort_timestamp: bool = True,
) -> pd.DataFrame:
    """
    Apply lightweight preprocessing to a raw SCADA dataframe.

    This is the main orchestration function for use in notebook 02
    (`02_data_combination.ipynb`). It performs structural standardization only.

    Operations may include:
    - normalize column names
    - deduplicate column names
    - standardize timestamp column
    - add metadata columns
    - sort by timestamp
    - reorder key columns

    Parameters
    ----------
    df : pandas.DataFrame
        Raw turbine dataframe.
    wind_farm : str
        Wind farm label to add as metadata.
    turbine_id : str
        Turbine identifier to add as metadata.
    source_file : str | None, optional
        Source filename or path for lineage tracking.
    timestamp_col : str | None, optional
        Explicit source timestamp column name. If None, automatic detection
        is attempted.
    normalize_cols : bool, optional
        Whether to normalize column names. Default is True.
    deduplicate_cols : bool, optional
        Whether to deduplicate duplicate column labels. Default is True.
    standardize_timestamp : bool, optional
        Whether to parse a timestamp column into a standardized `timestamp`
        column. Default is True.
    sort_timestamp : bool, optional
        Whether to sort rows by `timestamp`. Default is True.

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataframe suitable for storage in interim parquet form.
    """
    df_out = df.copy()

    if normalize_cols:
        df_out = normalize_column_names(df_out)

    if deduplicate_cols:
        df_out = deduplicate_column_names(df_out)

    if standardize_timestamp:
        inferred_timestamp_col = timestamp_col
        if inferred_timestamp_col is not None:
            inferred_timestamp_col = normalize_column_name(inferred_timestamp_col) \
                if normalize_cols else inferred_timestamp_col

        df_out = standardize_timestamp_column(
            df_out,
            timestamp_col=inferred_timestamp_col,
            output_col="timestamp",
        )

    df_out = add_turbine_metadata(
        df_out,
        wind_farm=wind_farm,
        turbine_id=turbine_id,
        source_file=source_file,
    )

    if sort_timestamp and "timestamp" in df_out.columns:
        df_out = sort_by_timestamp(df_out, timestamp_col="timestamp")

    df_out = reorder_core_columns(df_out)

    return df_out


# ---------------------------------------------------------------------------
# Schema and preprocessing summaries
# ---------------------------------------------------------------------------

def summarize_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a column-level schema summary for a dataframe.

    The summary is useful for exploratory analysis and pipeline diagnostics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per column and the following fields:
        - column_name
        - dtype
        - non_null_count
        - null_count
        - null_fraction
        - n_unique
    """
    summary = pd.DataFrame(
        {
            "column_name": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null_count": [int(df[col].notna().sum()) for col in df.columns],
            "null_count": [int(df[col].isna().sum()) for col in df.columns],
            "null_fraction": [float(df[col].isna().mean()) for col in df.columns],
            "n_unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )

    return summary


def summarize_preprocessing_result(df: pd.DataFrame) -> dict[str, object]:
    """
    Generate a lightweight summary of a preprocessed dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    dict[str, object]
        Dictionary containing structural summary statistics such as row count,
        column count, timestamp range, and identifier presence.
    """
    summary: dict[str, object] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "has_timestamp": "timestamp" in df.columns,
        "has_wind_farm": "wind_farm" in df.columns,
        "has_turbine_id": "turbine_id" in df.columns,
    }

    if "timestamp" in df.columns:
        summary["timestamp_min"] = df["timestamp"].min()
        summary["timestamp_max"] = df["timestamp"].max()
        summary["n_missing_timestamps"] = int(df["timestamp"].isna().sum())

    return summary


    # ---------------------------------------------------------------------------
# Memory optimization
# ---------------------------------------------------------------------------

def optimize_dtypes(
    df: pd.DataFrame,
    convert_int: bool = True,
    convert_float: bool = True,
    convert_object: bool = False,
    category_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Optimize dataframe dtypes to reduce memory usage.

    This function performs safe dtype downcasting without altering the
    semantic meaning of the data. It is intended for early-stage processing
    (e.g., during data combination) where memory efficiency is important.

    Operations include:
    - Downcasting integer types (e.g., int64 → int32/int16)
    - Downcasting float types (e.g., float64 → float32)
    - Optional conversion of low-cardinality object columns to category

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    convert_int : bool, optional
        Whether to downcast integer columns. Default is True.
    convert_float : bool, optional
        Whether to downcast float columns. Default is True.
    convert_object : bool, optional
        Whether to convert object columns to categorical where appropriate.
        Default is False (safer for early pipeline stages).
    category_threshold : float, optional
        Maximum ratio of unique values to total rows for converting an object
        column to category. Only used if convert_object=True.

    Returns
    -------
    pandas.DataFrame
        Copy of dataframe with optimized dtypes.

    Notes
    -----
    - This function does NOT modify timestamp columns.
    - This function avoids aggressive transformations that could affect modeling.
    - Object-to-category conversion is optional and conservative.
    """
    df_out = df.copy()

    # -----------------------------------------------------------------------
    # Integer downcasting
    # -----------------------------------------------------------------------
    if convert_int:
        int_cols = df_out.select_dtypes(include=["int", "int64", "int32"]).columns
        for col in int_cols:
            df_out[col] = pd.to_numeric(df_out[col], downcast="integer")

    # -----------------------------------------------------------------------
    # Float downcasting
    # -----------------------------------------------------------------------
    if convert_float:
        float_cols = df_out.select_dtypes(include=["float"]).columns
        for col in float_cols:
            df_out[col] = pd.to_numeric(df_out[col], downcast="float")

    # -----------------------------------------------------------------------
    # Object → category (optional)
    # -----------------------------------------------------------------------
    if convert_object:
        obj_cols = df_out.select_dtypes(include=["object"]).columns

        for col in obj_cols:
            n_unique = df_out[col].nunique(dropna=True)
            n_total = len(df_out)

            if n_total == 0:
                continue

            uniqueness_ratio = n_unique / n_total

            if uniqueness_ratio <= category_threshold:
                df_out[col] = df_out[col].astype("category")

    return df_out