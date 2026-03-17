"""
wtfd.preprocessing
==================

Lightweight preprocessing utilities for the wind turbine fault detection
project.

This module focuses on structural standardization rather than cleaning or
feature engineering.

Responsibilities
----------------
- normalize column names
- deduplicate duplicate column labels
- identify and standardize timestamp columns
- add metadata columns when needed
- sort by timestamp
- optimize dtypes for memory efficiency

This module intentionally does NOT:
- impute missing values
- remove outliers
- repair time gaps
- harmonize features across farms
- engineer modeling features
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def normalize_column_name(name: object) -> str:
    """
    Normalize a single column name.

    Parameters
    ----------
    name : object
        Original column name.

    Returns
    -------
    str
        Normalized column name.
    """
    normalized = str(name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def deduplicate_names(names: Iterable[object]) -> list[str]:
    """
    Deduplicate a sequence of column names.

    Parameters
    ----------
    names : Iterable[object]
        Column names.

    Returns
    -------
    list[str]
        Unique column names with numeric suffixes applied where needed.
    """
    seen: dict[str, int] = {}
    deduped: list[str] = []

    for name in names:
        col = str(name)
        if col not in seen:
            seen[col] = 1
            deduped.append(col)
        else:
            seen[col] += 1
            deduped.append(f"{col}_{seen[col]}")

    return deduped


def find_timestamp_column(
    df: pd.DataFrame,
    candidates: Iterable[str] | None = None,
) -> str | None:
    """
    Attempt to identify the timestamp column in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    candidates : Iterable[str] | None, optional
        Candidate timestamp column names.

    Returns
    -------
    str | None
        Matching timestamp column name, or None if not found.
    """
    if candidates is None:
        candidates = ("timestamp", "datetime", "date_time", "time", "date")

    columns = list(df.columns)
    lower_map = {str(col).lower(): str(col) for col in columns}

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    pattern = re.compile(r"(timestamp|datetime|date_time|time|date)")
    for col in columns:
        if pattern.search(str(col).lower()):
            return str(col)

    return None


def optimize_dtypes(
    df: pd.DataFrame,
    convert_int: bool = True,
    convert_float: bool = True,
    convert_object: bool = False,
    category_threshold: float = 0.05,
    skip_columns: set[str] | None = None,
) -> pd.DataFrame:
    """
    Optimize dataframe dtypes to reduce memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    convert_int : bool, optional
        Whether to downcast integer columns.
    convert_float : bool, optional
        Whether to downcast float columns.
    convert_object : bool, optional
        Whether to convert low-cardinality object columns to category.
    category_threshold : float, optional
        Maximum unique-to-total ratio for category conversion.
    skip_columns : set[str] | None, optional
        Columns to exclude from dtype optimization.

    Returns
    -------
    pd.DataFrame
        Dataframe with optimized dtypes.
    """
    skip_columns = skip_columns or set()

    if convert_int:
        int_cols = [c for c in df.select_dtypes(include=["integer"]).columns if c not in skip_columns]
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    if convert_float:
        float_cols = [c for c in df.select_dtypes(include=["floating"]).columns if c not in skip_columns]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")

    if convert_object:
        obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in skip_columns]
        n_total = len(df)

        for col in obj_cols:
            if n_total == 0:
                continue

            n_unique = df[col].nunique(dropna=True)
            uniqueness_ratio = n_unique / n_total

            if uniqueness_ratio <= category_threshold:
                df[col] = df[col].astype("category")

    return df


def preprocess_scada_for_eda(
    df: pd.DataFrame,
    timestamp_col: str | None = None,
    normalize_cols: bool = True,
    deduplicate_cols: bool = True,
    sort_timestamp: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Lightweight preprocessing for notebook-based EDA.

    This function avoids adding repeated metadata columns to every row, which
    helps reduce memory usage during notebook exploration.

    Parameters
    ----------
    df : pd.DataFrame
        Raw SCADA dataframe.
    timestamp_col : str | None, optional
        Explicit timestamp column name.
    normalize_cols : bool, optional
        Whether to normalize column names.
    deduplicate_cols : bool, optional
        Whether to deduplicate duplicate column labels.
    sort_timestamp : bool, optional
        Whether to sort by timestamp.
    copy : bool, optional
        Whether to operate on a shallow copy.

    Returns
    -------
    pd.DataFrame
        Lightly preprocessed dataframe.
    """
    df_out = df.copy(deep=False) if copy else df

    if normalize_cols:
        df_out.columns = [normalize_column_name(col) for col in df_out.columns]

    if deduplicate_cols:
        df_out.columns = deduplicate_names(df_out.columns)

    inferred_timestamp_col = timestamp_col
    if inferred_timestamp_col is not None and normalize_cols:
        inferred_timestamp_col = normalize_column_name(inferred_timestamp_col)

    if inferred_timestamp_col is None:
        inferred_timestamp_col = find_timestamp_column(df_out)

    if inferred_timestamp_col is None or inferred_timestamp_col not in df_out.columns:
        raise ValueError("No timestamp column could be identified in the dataframe.")

    df_out["timestamp"] = pd.to_datetime(
        df_out[inferred_timestamp_col],
        errors="coerce",
    )

    if sort_timestamp and "timestamp" in df_out.columns:
        df_out.sort_values(by="timestamp", inplace=True)
        df_out.reset_index(drop=True, inplace=True)

    return df_out


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
    copy: bool = True,
) -> pd.DataFrame:
    """
    Apply lightweight structural preprocessing to a raw SCADA dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw SCADA dataframe.
    wind_farm : str
        Wind farm identifier.
    turbine_id : str
        Turbine identifier.
    source_file : str | None, optional
        Source file path for lineage.
    timestamp_col : str | None, optional
        Explicit timestamp column name.
    normalize_cols : bool, optional
        Whether to normalize column names.
    deduplicate_cols : bool, optional
        Whether to deduplicate duplicate column labels.
    standardize_timestamp : bool, optional
        Whether to create a standardized timestamp column.
    sort_timestamp : bool, optional
        Whether to sort by timestamp.
    copy : bool, optional
        Whether to operate on a shallow copy.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe.
    """
    df_out = df.copy(deep=False) if copy else df

    if normalize_cols:
        df_out.columns = [normalize_column_name(col) for col in df_out.columns]

    if deduplicate_cols:
        df_out.columns = deduplicate_names(df_out.columns)

    if standardize_timestamp:
        inferred_timestamp_col = timestamp_col
        if inferred_timestamp_col is not None and normalize_cols:
            inferred_timestamp_col = normalize_column_name(inferred_timestamp_col)

        if inferred_timestamp_col is None:
            inferred_timestamp_col = find_timestamp_column(df_out)

        if inferred_timestamp_col is None or inferred_timestamp_col not in df_out.columns:
            raise ValueError("No timestamp column could be identified in the dataframe.")

        df_out["timestamp"] = pd.to_datetime(
            df_out[inferred_timestamp_col],
            errors="coerce",
        )

    df_out.insert(0, "wind_farm", wind_farm)
    df_out.insert(1, "turbine_id", turbine_id)

    if source_file is not None:
        df_out.insert(2, "source_file", source_file)

    if sort_timestamp and "timestamp" in df_out.columns:
        df_out.sort_values(by="timestamp", inplace=True)
        df_out.reset_index(drop=True, inplace=True)

    first_columns = ["wind_farm", "turbine_id", "source_file", "timestamp"]
    present_first = [col for col in first_columns if col in df_out.columns]
    remaining = [col for col in df_out.columns if col not in present_first]
    df_out = df_out.loc[:, present_first + remaining]

    return df_out