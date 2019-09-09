# moaney

A fictional company with real problems.

## Setup

We use a `Makefile` to encapsulate repeatable tasks.

First, create a data directory with:

```bash
make dirs
```

then fetch the data file with:

```bash
make data
```

This should create a file: `data/consumer_complaints.csv`.

Python requirements can be installed with:

```bash
make requirements
```

## Directory structure

The project has this schematic structure

```
mlops_complainer
├── .gitignore
├── Makefile
├── requirements.txt
├── README.md
├── data
│   └── ... created by Makefile, data (including persisted models) live here ...
└── experiments
    └── ... exploratory scripts live here ...
```

The experiments directory contains scripts to be run interactively or through CDSW experiments.