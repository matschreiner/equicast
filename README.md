# equicast

Modular equivariant machine learning for weather prediction.

## Installation

```bash
pip install -e .
```

## Quickstart

```bash
python config/train.py --fiddler config/fiddlers/benchmark.py
```

If it runs, you're set.

## Training on your own dataset

Create an anemoi-graph from your dataset and make sure the graph edges match the structure expected by your backbone. For example, if your model uses both grid and mesh nodes, those node types must also be present in your graph.

Equicast uses [anemoi-graphs](https://github.com/ecmwf/anemoi-graphs) for graph construction and [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets) as the dataset format.

## Non-equivariant models

It's straightforward to use a non-equivariant backbone: swap the backbone for a non-equivariant one and use `GraphDataHandler` instead of `EquivariantGraphDataHandler`.

## Custom data formats

Implement the `BaseDataHandler` interface to adapt your data to the model. It acts as an adapter between your data format and the rest of the pipeline.
