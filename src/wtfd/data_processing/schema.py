"""
wtfd.schema
===========

Schema inspection and metadata utilities for the wind turbine fault detection
project.

This module provides descriptive tools for understanding the structure of raw
and lightly preprocessed SCADA turbine datasets. It is designed to support the
Data Understanding and Data Combination stages of the pipeline by making schema
variation explicit across turbines and wind farms.

Purpose
-------
Wind farm datasets in this project do not share a guaranteed common schema, and
sensor names are not semantically consistent across farms. Because of this, the
project requires a schema inspection layer before any feature harmonization or
modeling can be performed safely.

This module therefore focuses on:
- describing dataframe schemas
- summarizing turbine-level structural metadata
- comparing schemas across turbines
- identifying column presence/absence patterns

This module should be used in:
- 01_exploratory_data_analysis.ipynb
- 02_data_combination.ipynb

This module should NOT be responsible for:
- feature harmonization across farms
- sensor-to-physical-variable mapping
- missing value imputation
- outlier handling
- feature engineering
- target creation
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    """
    Descriptive schema metadata for a single dataframe column.

    Attributes
    ----------
    column_name : str
        Name of the column.
    dtype : str
        Pandas dtype string for the column.
    non_null_count : int
        Number of non-missing values.
    null_count : int
        Number of missing values.
    null_fraction : float
        Fraction of values that are missing.
    n_unique : int
        Number of unique non-null values.
    """

    column_name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_fraction: float
    n_unique: int


@dataclass
class DataFrameSchemaSummary:
    """
    High-level structural summary for a dataframe.

    Attributes
    ----------
    n_rows : int
        Number of rows in the dataframe.
    n_columns : int
        Number of columns in the dataframe.
    column_names : list[str]
        Ordered list of dataframe column names.
    has_timestamp : bool
        Whether a standardized timestamp column exists.
    timestamp_min : str | None
        Minimum timestamp as string, if available.
    timestamp_max : str | None
        Maximum timestamp as string, if available.
    n_missing_timestamps : int | None
        Number of missing timestamps, if timestamp column exists.
    """

    n_rows: int
    n_columns: int
    column_names: list[str]
    has_timestamp: bool
    timestamp_min: str | None
    timestamp_max: str | None
    n_missing_timestamps: int | None


# ---------------------------------------------------------------------------
# Column-level schema extraction
# ---------------------------------------------------------------------------

def extract_column_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract column-level schema metadata from a dataframe.

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
    records: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        record = ColumnSchema(
            column_name=str(col),
            dtype=str(series.dtype),
            non_null_count=int(series.notna().sum()),
            null_count=int(series.isna().sum()),
            null_fraction=float(series.isna().mean()),
            n_unique=int(series.nunique(dropna=True)),
        )
        records.append(asdict(record))

    return pd.DataFrame(records)


def extract_dataframe_schema_summary(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """
    Extract a high-level structural summary for a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    timestamp_col : str, optional
        Name of the standardized timestamp column. Default is "timestamp".

    Returns
    -------
    dict[str, Any]
        Dictionary containing high-level schema summary information.
    """
    has_timestamp = timestamp_col in df.columns

    if has_timestamp:
        timestamp_min = df[timestamp_col].min()
        timestamp_max = df[timestamp_col].max()
        n_missing_timestamps = int(df[timestamp_col].isna().sum())

        timestamp_min_str = None if pd.isna(timestamp_min) else str(timestamp_min)
        timestamp_max_str = None if pd.isna(timestamp_max) else str(timestamp_max)
    else:
        timestamp_min_str = None
        timestamp_max_str = None
        n_missing_timestamps = None

    summary = DataFrameSchemaSummary(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        column_names=[str(col) for col in df.columns],
        has_timestamp=has_timestamp,
        timestamp_min=timestamp_min_str,
        timestamp_max=timestamp_max_str,
        n_missing_timestamps=n_missing_timestamps,
    )

    return asdict(summary)


def extract_full_schema(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """
    Extract both column-level and dataframe-level schema information.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    timestamp_col : str, optional
        Name of the standardized timestamp column. Default is "timestamp".

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - dataframe_summary: dict
        - column_schema: pandas.DataFrame
    """
    return {
        "dataframe_summary": extract_dataframe_schema_summary(
            df,
            timestamp_col=timestamp_col,
        ),
        "column_schema": extract_column_schema(df),
    }


# ---------------------------------------------------------------------------
# Metadata row generation
# ---------------------------------------------------------------------------

def build_turbine_metadata_row(
    df: pd.DataFrame,
    wind_farm: str,
    turbine_id: str,
    source_file: str | None = None,
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """
    Build a metadata summary row for a single turbine dataframe.

    This function is intended for constructing the
    `turbine_metadata_summary.csv` file during data combination.

    Parameters
    ----------
    df : pandas.DataFrame
        Input turbine dataframe.
    wind_farm : str
        Wind farm identifier.
    turbine_id : str
        Turbine identifier.
    source_file : str | None, optional
        Source CSV file path or name, if available.
    timestamp_col : str, optional
        Name of the timestamp column. Default is "timestamp".

    Returns
    -------
    dict[str, Any]
        A metadata dictionary suitable for conversion into a summary table row.
    """
    summary = extract_dataframe_schema_summary(df, timestamp_col=timestamp_col)

    metadata_row: dict[str, Any] = {
        "wind_farm": wind_farm,
        "turbine_id": turbine_id,
        "source_file": source_file,
        "n_rows": summary["n_rows"],
        "n_columns": summary["n_columns"],
        "has_timestamp": summary["has_timestamp"],
        "timestamp_min": summary["timestamp_min"],
        "timestamp_max": summary["timestamp_max"],
        "n_missing_timestamps": summary["n_missing_timestamps"],
        "column_names": "|".join(summary["column_names"]),
    }

    return metadata_row


def build_turbine_metadata_summary(
    turbine_dfs: Iterable[tuple[str, str, pd.DataFrame, str | None]],
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Build a metadata summary dataframe for multiple turbines.

    Parameters
    ----------
    turbine_dfs : Iterable[tuple[str, str, pandas.DataFrame, str | None]]
        Iterable of tuples in the form:
        (wind_farm, turbine_id, dataframe, source_file)
    timestamp_col : str, optional
        Name of the timestamp column. Default is "timestamp".

    Returns
    -------
    pandas.DataFrame
        Dataframe containing one metadata row per turbine.
    """
    rows: list[dict[str, Any]] = []

    for wind_farm, turbine_id, df, source_file in turbine_dfs:
        row = build_turbine_metadata_row(
            df=df,
            wind_farm=wind_farm,
            turbine_id=turbine_id,
            source_file=source_file,
            timestamp_col=timestamp_col,
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema comparison across turbines
# ---------------------------------------------------------------------------

def extract_column_presence_record(
    df: pd.DataFrame,
    wind_farm: str,
    turbine_id: str,
) -> pd.DataFrame:
    """
    Build a long-format column presence record for a turbine dataframe.

    Each row represents one column that exists in the given dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    wind_farm : str
        Wind farm identifier.
    turbine_id : str
        Turbine identifier.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with columns:
        - wind_farm
        - turbine_id
        - column_name
        - dtype
    """
    return pd.DataFrame(
        {
            "wind_farm": wind_farm,
            "turbine_id": turbine_id,
            "column_name": [str(col) for col in df.columns],
            "dtype": [str(df[col].dtype) for col in df.columns],
        }
    )


def compare_turbine_schemas(
    turbine_dfs: Iterable[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Compare schemas across multiple turbine dataframes.

    This function returns a wide binary-style presence matrix showing whether
    each column exists for each turbine.

    Parameters
    ----------
    turbine_dfs : Iterable[tuple[str, str, pandas.DataFrame]]
        Iterable of tuples in the form:
        (wind_farm, turbine_id, dataframe)

    Returns
    -------
    pandas.DataFrame
        Wide dataframe indexed by column_name with one indicator column per
        turbine in the format '{wind_farm}__{turbine_id}'.
    """
    records: list[pd.DataFrame] = []

    for wind_farm, turbine_id, df in turbine_dfs:
        presence_df = extract_column_presence_record(
            df=df,
            wind_farm=wind_farm,
            turbine_id=turbine_id,
        )
        presence_df["turbine_key"] = f"{wind_farm}__{turbine_id}"
        presence_df["present"] = 1
        records.append(presence_df[["column_name", "turbine_key", "present"]])

    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)

    comparison = (
        combined.pivot_table(
            index="column_name",
            columns="turbine_key",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .sort_index()
        .reset_index()
    )

    comparison.columns.name = None
    return comparison


def summarize_column_presence(
    schema_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how often each column appears across turbines.

    Parameters
    ----------
    schema_comparison_df : pandas.DataFrame
        Output of `compare_turbine_schemas`.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per column and summary statistics including:
        - column_name
        - turbines_present
        - presence_fraction
    """
    if schema_comparison_df.empty:
        return pd.DataFrame(columns=["column_name", "turbines_present", "presence_fraction"])

    value_cols = [col for col in schema_comparison_df.columns if col != "column_name"]
    n_turbines = len(value_cols)

    summary = schema_comparison_df.copy()
    summary["turbines_present"] = summary[value_cols].sum(axis=1)
    summary["presence_fraction"] = (
        summary["turbines_present"] / n_turbines if n_turbines > 0 else 0.0
    )

    return summary[["column_name", "turbines_present", "presence_fraction"]].sort_values(
        by=["turbines_present", "column_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


def get_schema_union(
    turbine_dfs: Iterable[tuple[str, str, pd.DataFrame]],
) -> list[str]:
    """
    Compute the union of column names across multiple turbine dataframes.

    Parameters
    ----------
    turbine_dfs : Iterable[tuple[str, str, pandas.DataFrame]]
        Iterable of tuples in the form:
        (wind_farm, turbine_id, dataframe)

    Returns
    -------
    list[str]
        Sorted list of all distinct column names across turbines.
    """
    union_cols: set[str] = set()

    for _, _, df in turbine_dfs:
        union_cols.update(str(col) for col in df.columns)

    return sorted(union_cols)


def get_schema_intersection(
    turbine_dfs: Iterable[tuple[str, str, pd.DataFrame]],
) -> list[str]:
    """
    Compute the intersection of column names across multiple turbine dataframes.

    Note
    ----
    This function is descriptive only. It should not be used to enforce a
    shared modeling schema, because matching raw sensor names do not guarantee
    matching physical meanings across wind farms.

    Parameters
    ----------
    turbine_dfs : Iterable[tuple[str, str, pandas.DataFrame]]
        Iterable of tuples in the form:
        (wind_farm, turbine_id, dataframe)

    Returns
    -------
    list[str]
        Sorted list of column names shared by all turbines.
    """
    turbine_list = list(turbine_dfs)

    if not turbine_list:
        return []

    intersection_cols = {str(col) for col in turbine_list[0][2].columns}

    for _, _, df in turbine_list[1:]:
        intersection_cols &= {str(col) for col in df.columns}

    return sorted(intersection_cols)


# ---------------------------------------------------------------------------
# Convenience reporting helpers
# ---------------------------------------------------------------------------

def summarize_schema_comparison(
    turbine_dfs: Iterable[tuple[str, str, pd.DataFrame]],
) -> dict[str, Any]:
    """
    Generate high-level summary statistics for schema variability.

    Parameters
    ----------
    turbine_dfs : Iterable[tuple[str, str, pandas.DataFrame]]
        Iterable of tuples in the form:
        (wind_farm, turbine_id, dataframe)

    Returns
    -------
    dict[str, Any]
        Dictionary with summary metrics describing schema overlap and variation.
    """
    turbine_list = list(turbine_dfs)

    if not turbine_list:
        return {
            "n_turbines": 0,
            "n_union_columns": 0,
            "n_intersection_columns": 0,
            "intersection_fraction_of_union": None,
        }

    union_cols = get_schema_union(turbine_list)
    intersection_cols = get_schema_intersection(turbine_list)

    intersection_fraction = (
        len(intersection_cols) / len(union_cols) if len(union_cols) > 0 else None
    )

    return {
        "n_turbines": len(turbine_list),
        "n_union_columns": len(union_cols),
        "n_intersection_columns": len(intersection_cols),
        "intersection_fraction_of_union": intersection_fraction,
    }