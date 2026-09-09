"""Support code for the notebooks. The experiments themselves live in the cells.

    data      splits as `datasets.Dataset`, system prompt, pretty printing
    metrics   deterministic checks, Wilson interval, metric aggregation
    report    run files, comparison tables, charts

Only `data` needs the `datasets` package; `metrics` and `report` are
standard library plus matplotlib for the charts. Nothing here imports torch.
"""

__version__ = "0.6.0"
