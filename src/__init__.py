"""Support code for the notebooks; the training itself stays in the cells.

    filter    the request filter: splits, the one-line format, exact metrics, `evaluate` / `show`
    data      the assistant's splits as `datasets.Dataset`, the system prompt, prompt building
    metrics   deterministic checks and the metric aggregation for the assistant
    infer     load, generate, judge, log-probabilities: thin wrappers over transformers
    report    the assistant's `evaluate` over both tests and `show` for the tables

Everything except `infer` works without torch, so the data and the results
notebooks open on any machine.
"""

__version__ = "0.8.0"
