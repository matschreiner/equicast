Core classes of the framework are


## dataset

**Responsibility:**
Read and return a timeseries of n 'physical' frames in any format — graph based, image based, or other formats.

**Functions:**
- `__getitem__`: `idx: int -> list[physical_data]` — return a timeseries of n 'physical' frames in any format, starting from frame idx.

**Instantiation parameters:**
- `datasource`: a datasource object responsible for reading data.
- `num_steps`: timeseries length.

**Notes:**
Should be agnostic towards the data pipeline and shouldn't, for example, be concerned with whether these frames will be used
for autoregressive steps or for multistep inputs. Also shouldn't handle statistics as this couples the dataset
to downstream tasks.


## backbone

**Responsibility:**
Map `backbone_input_data` to `backbone_output_data`.

**Second responsibility (possibly):**
Conditional sampling given an input.
For generative models the forward pass used in training is generally different from the sampling process — e.g. skipping the
encoder in a VAE, or SDE/ODE solving in score-based models.

**Functions:**
- `forward`: `backbone_in_data -> backbone_out_data`
- `predict` (possibly): `backbone_in_data -> backbone_out_data`

**Instantiation parameters:**
Preferably Python primitives.

**Notes:**
Should also be agnostic towards the data pipeline — should only receive input and produce an output. Input can also include noise
for training a VAE/diffusion backbone.
The backbone shouldn't be responsible for anything other than mapping input to output. It should ideally instantiate from Python
primitives. It should handle all processing logic including residual connections, etc., and should know what is physical space vs.
model space. Shouldn't be responsible for any orchestration or normalization.
Although current backbones take a data handler/feature index at instantiation simply to fetch in/out dimensions from those,
this is strictly not necessary and reduces portability of the backbones since they cannot instantiate in other frameworks
that don't have the `data_handler` concept.


## data_handler

**Responsibility:**
Handle the data pipeline.

**Functions:**
- `prepare_backbone_input`: `physical_data -> backbone_in_data`
- `prepare_backbone_target`: `physical_data -> backbone_target_data`
- `update_with_output`: `physical_data, backbone_out_data -> physical_data`
- `to_cf`: `physical_data -> cf_compliant_data`

**Instantiation parameters:**
- Data statistics.

**Notes:**
Since input and output data for the backbone may be different — different feature sets, number of input frames, injected noise for
generative models, etc. — we need different processing for inputs and outputs. This data handler makes sure that the backbone
model can interface cleanly with the dataset producing timeseries with any representation of physical data.

Since the abstract backbone makes no assumptions about the data format, this class can also be used as the single source of truth
to handle sharding and calculate communication groups, which can be passed along to a backbone that implements the sharding
functionality as well.

### Training pipeline
```
backbone_input  = prepare_backbone_input(physical_data)
backbone_target = prepare_backbone_target(physical_data)
backbone_output = backbone(backbone_input)

loss = loss_fn(backbone_output, backbone_target)
```

### Prediction pipeline
```
backbone_input     = prepare_backbone_input(physical_data)
backbone_output    = backbone(backbone_input)
physical_data_next = update_with_output(physical_data, backbone_output)
cf_compliant_data  = to_cf(physical_data_next)           # optional
```
This ensures the prediction is in physical space and CF-compliant, and is agnostic to architectural details and data processing.

### Autoregressive prediction
```
backbone_input_t1  = prepare_backbone_input(physical_data_t1)
backbone_output_t2 = backbone(backbone_input_t1)
physical_data_t2   = update_with_output(physical_data_t1, backbone_output_t2)

backbone_input_t2  = prepare_backbone_input(physical_data_t2)
backbone_output_t3 = backbone(backbone_input_t2)
...
```
This logic can be used for both multistep training and downstream inference.

`to_cf` maps physical data to a CF-compliant format, so we can interface our model cleanly with external visualisation and benchmarking tools.


## model

**Responsibility:**
1. Orchestration of `data_handler` and `backbone` in the training pipeline.
2. Autoregressive prediction with the backbone.

**Instantiation parameters:**
- `backbone`: a backbone object.
- `data_handler`: a data handler object.
- `loss`: loss function that matches the task — KL loss for VAEs, MSE for deterministic models, EquivariantMSE for equivariant models, CRPS, etc.

**Notes:**
Owning the `data_handler` and `backbone` creates a portable model that knows how to process physical data and perform downstream
tasks, decoupled from the dataset it was trained on.
Since both are injected into the model there is no coupling between the core model and the data pipeline or the backbone
architecture. Because the model follows the principle of interface segregation — i.e. only depends on abstract functions from the
`data_handler` and the `backbone` — it can be used to perform autoregressive predictions, multistep training, or other downstream
tasks using any format for the physical data and any backbone architecture, while being able to output CF-compliant output.


By following these concepts, multiple architectural features become available within the backbone without any bespoke logic in
the model class. To integrate a backbone from another framework, implement a `data_handler` with the four abstract functions —
`prepare_backbone_input`, `prepare_backbone_target`, `update_with_output`, and `to_cf` — and it will work with the same model
and training loop while strictly containing all architectural decisions within the backbone.


# Bespoke pipelines

## Equivariant model pipeline

### Dataset

Graphs are created with anemoi-graphs. The data input node names are `'grid'` by default and the dataset puts node features
from the anemoi dataset on `graph['grid'].data`.

### PaiNN backbone

`forward(graph) -> dict[str, Tensor]`

**Required attributes on `graph['grid']`:**
- `input_scalar`:    `Tensor [n_nodes, in_scalar_dim]`   — z-normalized scalar fields (forcing + prognostic)
- `input_vector`:    `Tensor [n_nodes, in_vector_dim, 2]` — norm-normalized vector pairs (forcing + prognostic)
- `residual_scalar`: `Tensor [n_nodes, out_scalar_dim]`  — normalized prognostic + diagnostic scalars (residual)
- `residual_vector`: `Tensor [n_nodes, out_vector_dim, 2]` — normalized output vector pairs (residual)

Each edge type in the graph also requires `edge_index [2, n_edges]`, `edge_dirs [n_edges, 2]`, and `edge_length [n_edges]`.

**Output:**
- `'scalar'`: `Tensor [n_nodes, out_scalar_dim]`   — scalar predictions with residual added
- `'vector'`: `Tensor [n_nodes, out_vector_dim, 2]` — vector predictions with residual added

The forward pass runs PaiNN message-passing blocks over each edge type in sequence, then adds the learned increments
to `residual_scalar` and `residual_vector` before returning.

### Data handler

`prepare_backbone_input` packs the graph attributes PaiNN expects onto `graph['grid']`: `u_p` and `v_p` pairs are packed
into `input_vector [n_nodes, in_vector_dim, 2]` with norm-normalized magnitudes; scalar fields are z-normalized into
`input_scalar [n_nodes, in_scalar_dim]`. Prognostic and diagnostic fields are also written to `residual_scalar` and
`residual_vector` so PaiNN can apply residual connections.

`prepare_backbone_target` takes the next timestep and extracts normalized target scalars into `'scalar'` and
target vector pairs into `'vector'` in a dictionary, matching the shapes of PaiNN's output tensors.

`update_with_output` takes PaiNN's `{'scalar', 'vector'}` output, denormalizes them, and writes the predicted fields back
into physical space on the graph — unpacking `vector` back to `u_p` / `v_p` components.

`to_cf` maps the graph data back to a CF-compliant format.

### Loss

Computes scalar MSE between `output['scalar']` and `target['scalar']`, plus a squared-norm difference between
`output['vector']` and `target['vector']`.


## Standard graph model pipeline

Graphs are created with anemoi-graphs. `prepare_backbone_input` packs all physical scalar fields as node features,
z-normalizes them, and attaches them to `graph['grid'].x`. Prognostic features are also saved as `'residual_scalar'`.
There are no vector features — all fields are treated as scalars.

`prepare_backbone_target` packs the target prognostic and diagnostic scalar fields from the next timestep into
`'scalar_output'` on the graph.

The backbone outputs `{'scalar': Tensor [n_nodes, n_out_features]}`. `update_with_output` denormalizes and writes the
predicted scalars back into physical space on the graph. `to_cf` maps `graph['grid'].x` back to a CF-compliant format.


## Possible future pipelines

- **Multidomain models** — the graph provider supplies graphs from multiple domains while the backbone stays the same.
- **Sharding** — the data handler shards the graph and calculates communication groups; the backbone implements a sharded forward pass with all-reduce and/or all-gather ops as needed.
- **Image-based models** — the dataset produces image-based data, the backbone is a CNN or U-Net, and the data handler converts physical data to and from image format.
