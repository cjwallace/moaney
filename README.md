# moaney

A fictional company with real problems.

# ## Setup

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
