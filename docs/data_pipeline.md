# WTDF Data Pipeline

## Purpose

This document describes the end-to-end data pipeline for the `wtfd` wind turbine fault detection project, from raw dataset ingestion through model-ready feature creation.

The pipeline is designed to be:

* modular
* memory-safe
* reproducible
* transparent for notebook-based academic review

## Source Data

The project uses the Zenodo wind farm dataset stored under:

```text
data/raw/zenodo_windfarm_data/
  Wind Farm A/
  Wind Farm B/
  Wind Farm C/
```

Each farm contains:

* `datasets/*.csv` for turbine-level SCADA data
* `feature_description.csv`
* `event_info.csv`

All CSV files are semicolon-delimited and must be read with `sep=";"`.

## Core Constraint

Sensor names are not consistent across farms. This means:

* raw features cannot be aligned by column name alone
* schema comparison must be handled explicitly
* cross-farm feature harmonization must rely on `feature_description.csv`

Because of this, the pipeline separates structural understanding from feature alignment.

## Pipeline Stages

## 1. Raw Data Discovery

### Objective

Locate all farms, turbines, and metadata files without loading the full dataset into memory.

### Inputs

* raw farm directories
* turbine CSV files
* farm-level metadata files

### Outputs

* file manifests
* turbine identifiers
* farm-to-file mappings

### Implemented in

* `wtfd.data`

### Notes

This stage should be lightweight and primarily path-based.

---

## 2. Exploratory Data Analysis (EDA)

### Objective

Understand the structure, quality, and temporal behavior of the raw turbine datasets.

### Strategy

The EDA stage uses a two-pass design.

#### Pass 1: all turbines, summary metrics only

A single loop across all turbines computes compact summaries such as:

* metadata
* schema presence
* overall missingness
* column missingness
* timestamp coverage
* time gap statistics
* gap bucket counts

#### Pass 2: representative turbines only

A second loop loads selected turbines for plots and illustrative examples.

### Why this design works

* avoids loading all turbines simultaneously
* avoids expensive plotting in the main loop
* supports broad coverage and focused interpretation

### Outputs

* summary DataFrames
* farm/turbine metadata tables
* representative EDA plots
* documented structural findings

### Implemented in

* notebook: `01_exploratory_data_analysis.ipynb`
* support modules: `wtfd.data`, `wtfd.preprocessing`, `wtfd.schema`

---

## 3. Raw-to-Interim Conversion

### Objective

Convert raw turbine CSV files into a more efficient intermediate storage format.

### Design choice

Create **one Parquet file per turbine**.

### Why one Parquet per turbine

* memory-safe for large datasets
* avoids repeated raw CSV parsing
* enables targeted reloads
* keeps pipeline stages modular
* simplifies error isolation and restartability

### Inputs

* raw turbine CSV
* farm and turbine metadata
* optional schema/provenance information

### Processing

* load one turbine CSV
* apply minimal standardized preprocessing
* preserve key provenance fields
* write turbine-level Parquet output

### Outputs

* `data/interim/.../*.parquet`
* optional manifest of generated turbine files

### Implemented in

* notebook: `02_data_combination.ipynb`
* support module: `wtfd.interim`

---

## 4. Data Cleaning

### Objective

Apply explicit cleaning rules to interim turbine datasets before feature generation.

### Potential cleaning tasks

* timestamp validation
* duplicate removal
* impossible or invalid value checks
* missingness-based filtering
* data type correction
* consistency checks across records

### Inputs

* turbine-level interim Parquet files

### Outputs

* cleaned turbine-level datasets
* before/after row and column summaries
* cleaning audit tables

### Implemented in

* notebook: `03_data_cleaning.ipynb`
* support module: `wtfd.cleaning`

### Notes

Cleaning rules should be explicit, justified, and documented for grading transparency.

---

## 5. Feature Mapping and Feature Engineering

### Objective

Construct aligned, model-ready features from cleaned turbine data.

### Important requirement

Because farm schemas differ, feature mapping must use `feature_description.csv` instead of raw sensor names.

### Potential feature engineering tasks

* sensor mapping and harmonization
* temporal features
* lag features
* rolling statistics
* operating-state features
* aggregated fault-relevant indicators
* supervised target creation from event metadata, if applicable

### Inputs

* cleaned turbine data
* `feature_description.csv`
* `event_info.csv` if used for target creation or event linkage

### Outputs

* processed feature tables
* feature dictionaries and mapping tables
* model-ready datasets

### Implemented in

* notebook: `04_feature_engineering.ipynb`
* support module: `wtfd.features`

---

## 6. Modeling and Evaluation

### Objective

Train and evaluate fault detection models using the engineered feature layer.

### Potential tasks

* train/validation/test split creation
* baseline model comparison
* hyperparameter tuning
* metric reporting
* error analysis
* feature importance or interpretability analysis

### Inputs

* model-ready feature tables
* labels or targets

### Outputs

* trained model artifacts
* evaluation metrics
* plots and result summaries

### Implemented in

* notebook: `05_modeling.ipynb`
* support module: `wtfd.modeling`

---

## Recommended Directory Flow

```text
data/
├── raw/
│   └── zenodo_windfarm_data/
├── interim/
│   └── turbine_parquet/
├── processed/
│   ├── cleaned/
│   └── features/
└── artifacts/
    ├── eda/
    ├── cleaning/
    └── modeling/
```

## Data Layer Responsibilities

### Raw

Stores untouched source data exactly as provided.

### Interim

Stores efficient, turbine-level converted data with minimal transformation.

### Processed

Stores cleaned and engineered datasets intended for modeling.

### Artifacts

Stores outputs such as summary tables, plots, manifests, and model results.

## Reproducibility Practices

To keep the pipeline reproducible:

* centralize paths and thresholds in YAML config files
* keep transformation logic in package modules
* use notebooks for orchestration and interpretation only
* preserve provenance from farm and turbine sources
* write intermediate outputs rather than recomputing everything repeatedly

## Performance Considerations

The dataset should be processed incrementally. Recommended practices include:

* iterate turbine-by-turbine
* avoid concatenating all turbines into one giant DataFrame unless absolutely necessary
* write artifacts at stage boundaries
* compute summaries in one pass where possible
* reserve plotting for representative subsets

## Grading-Friendly Structure

Each notebook should clearly show:

1. the objective of the stage
2. the package functions being used
3. the outputs generated
4. the interpretation of those outputs
5. how the stage feeds the next stage

This makes the project easier to evaluate while preserving software engineering discipline.

## Key Methodological Statement

A major methodological feature of this pipeline is that cross-farm alignment is deferred until metadata-driven mapping is available. This is necessary because the same physical concept may appear under different raw sensor names in different farms.

That decision protects the validity of downstream modeling and should be documented explicitly in both the EDA and feature engineering stages.

## Summary

The `wtfd` data pipeline transforms raw, schema-inconsistent wind farm CSV data into a modular, turbine-level, model-ready machine learning workflow. Its structure emphasizes memory safety, traceability, and correct feature alignment, making it suitable for both practical analysis and academic review.
