# WTDF Architecture

## Overview

The `wtfd` project is designed as a structured, modular machine learning pipeline for wind turbine fault detection. The architecture separates reusable Python package logic from notebook-based analysis and presentation so that the workflow remains reproducible, memory-safe, and grading-friendly.

At a high level, the project follows a layered data pipeline:

1. **Raw layer**: semicolon-delimited CSV files from the Zenodo wind farm dataset.
2. **Interim layer**: one Parquet file per turbine for efficient downstream processing.
3. **Clean layer**: cleaned turbine-level datasets with validated timestamps, reduced noise, and standardized metadata.
4. **Feature layer**: model-ready features aligned using `feature_description.csv` rather than raw column names.
5. **Modeling layer**: train, validate, and evaluate fault detection models.

## Design Principles

### 1. Package-first, notebook-supported

The `wtfd` Python package contains the reusable processing logic. Notebooks act as orchestration, explanation, and presentation layers.

This keeps the codebase:

* easier to maintain
* easier to test
* easier to grade
* less fragile than placing all logic directly inside notebooks

### 2. Memory-safe processing

The full dataset should not be loaded into memory at once. Processing is performed turbine-by-turbine, which supports:

* large-scale data handling on modest hardware
* incremental saving to Parquet
* modular debugging
* resumable execution

### 3. Schema-aware processing

Sensor names are not consistent across farms, so features cannot be aligned by raw column name alone. The architecture therefore treats `feature_description.csv` as the authoritative source for cross-farm feature mapping.

### 4. Layered transformation

Each stage has a focused responsibility:

* raw ingestion
* structural understanding
* cleaning
* feature creation
* modeling

This prevents notebooks from becoming monolithic and keeps transformations traceable.

## Project Structure

```text
wtfd/
├── data/
│   ├── raw/
│   │   └── zenodo_windfarm_data/
│   │       ├── Wind Farm A/
│   │       ├── Wind Farm B/
│   │       └── Wind Farm C/
│   ├── interim/
│   ├── processed/
│   └── artifacts/
├── docs/
│   ├── architecture.md
│   └── data_pipeline.md
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_combination.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_modeling.ipynb
├── src/
│   └── wtfd/
│       ├── config/
│       ├── data/
│       ├── preprocessing/
│       ├── schema/
│       ├── interim/
│       ├── cleaning/
│       ├── features/
│       ├── modeling/
│       └── utils/
└── tests/
```

## Package Responsibilities

### `wtfd.config`

Handles YAML-based configuration for paths, thresholds, and notebook-independent runtime settings.

### `wtfd.data`

Responsible for:

* farm and turbine file discovery
* raw CSV loading with `sep=";"`
* metadata extraction from file and folder structure

### `wtfd.preprocessing`

Provides lightweight and full preprocessing functions, depending on the stage of the pipeline.

Examples:

* EDA-focused preprocessing
* standard timestamp parsing
* basic type normalization

### `wtfd.schema`

Responsible for schema extraction and comparison.

Examples:

* listing columns for each turbine
* comparing structural differences across farms
* generating schema presence summaries

### `wtfd.interim`

Responsible for raw-to-interim conversion.

Examples:

* transforming one CSV into one Parquet file
* writing turbine manifests
* preserving provenance fields for downstream reuse

### `wtfd.cleaning`

Responsible for cleaning rules applied after the interim layer is created.

Examples:

* duplicate handling
* timestamp validation
* missingness threshold application
* invalid value checks

### `wtfd.features`

Responsible for feature alignment and feature engineering.

Examples:

* mapping sensors using `feature_description.csv`
* creating time-based features
* creating rolling and aggregation features
* preparing model-ready matrices

### `wtfd.modeling`

Responsible for training and evaluation support.

Examples:

* dataset splitting
* baseline model training
* metric calculation
* experiment summaries

## Notebook Responsibilities

### `01_exploratory_data_analysis.ipynb`

Documents the structure and quality of the raw data using a two-pass EDA strategy.

### `02_data_combination.ipynb`

Builds the interim turbine-level Parquet layer.

### `03_data_cleaning.ipynb`

Applies explicit cleaning rules and documents their impact.

### `04_feature_engineering.ipynb`

Constructs aligned, model-ready features.

### `05_modeling.ipynb`

Trains and evaluates fault detection models.

## Two-Pass EDA Strategy

The exploratory analysis notebook uses two passes for efficiency and clarity.

### Pass 1: all turbines, summaries only

A single loop computes compact summary outputs such as:

* metadata
* schema presence
* overall missingness
* column missingness
* timestamp coverage
* time gap statistics
* gap bucket counts

### Pass 2: representative turbines only

A second pass loads selected turbines for visual analysis only. This avoids plotting overhead inside the master loop while still providing interpretable charts for the report.

## Data Alignment Constraint

A central architectural constraint is that feature names are not standardized across farms. As a result:

* raw column names are not sufficient for cross-farm feature alignment
* farm-specific schemas must be preserved during ingestion and EDA
* later harmonization must be based on `feature_description.csv`

This is one of the most important methodological decisions in the project.

## Why One Parquet Per Turbine

Using one Parquet file per turbine provides several advantages:

* supports out-of-core processing
* improves notebook responsiveness
* allows selective reloads for debugging
* separates raw ingestion from downstream modeling
* reduces repeated CSV parsing costs

It also creates a stable intermediate representation that can be reused by cleaning, feature engineering, and modeling notebooks.

## Recommended Execution Flow

1. Discover raw files by farm and turbine.
2. Run EDA to understand structure and data quality.
3. Convert each turbine dataset to an interim Parquet file.
4. Apply cleaning rules to produce processed turbine data.
5. Map and engineer features using metadata descriptions.
6. Assemble model-ready datasets.
7. Train, validate, and evaluate models.

## Future Extensions

Potential optional improvements include:

* manifest tracking for all turbine artifacts
* structured logging per pipeline stage
* experiment tracking for modeling runs
* validation checks at each data layer
* lightweight CLI entry points for batch execution

## Summary

The `wtfd` architecture is intentionally modular, memory-safe, and schema-aware. It supports a disciplined transition from raw wind farm CSV files to a model-ready machine learning pipeline while preserving the transparency and structure needed for academic evaluation.
