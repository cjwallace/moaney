# moaney

A fictional company with real problems.


## Business context

Moaney is a consumer finance company that handles financial complaints for customers.
In order to perform effective triage, they have support representatives that specialize in certain kinds of issue.
They want a ML app that can take the raw text of a complaint submitted via a web form and suggest the kind of issue it's about so that it can be directed to the appropriate support representative.

Complainer is a proof-of-concept ML app that allocates mortgage complaints from [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) to a defined ontology of issues.
The data comes from the [Consumer Financial Protection Bureau](ttps://www.consumerfinance.gov).
An explanation of each field in the dataset is given in the [field reference](https://cfpb.github.io/api/ccdb/fields.html).


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