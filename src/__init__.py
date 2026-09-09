"""Support code for the notebooks; the training itself stays in the cells.

    data      splits as `datasets.Dataset`, the system prompt, prompt building, pretty printing
    metrics   deterministic checks and the metric aggregation
    infer     load, generate, judge, log-probabilities: thin wrappers over transformers
    report    `evaluate` runs a model over every test and saves a run; `show` prints the tables

`data`, `metrics` and `report.show` work without torch, so the data and the
results notebooks open on any machine.
"""

__version__ = "0.7.0"
