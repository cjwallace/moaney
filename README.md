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
├── experiments
    └── ... exploratory scripts live here ...
├── jobs
│   └── ... CDSW job scripts live here ...
└── complainer
    ├── ... modular code ...
    └── tests
        └── ... tests ...
```

The `experiments` directory contains scripts to be run interactively or through CDSW experiments.

The `complainer` directory is an installable python library containing code upon which some experiments and jobs depend.

The `jobs` directory contains imperative scripts that tie together the `complainer` module functionality (and some other well used tools from the python ecosystem).
Jobs are how we "do things" with this project, and are where all I/O and side-effectful actions occur.


## Development

When developing the complainer library, run tests with

```bash
make test
```