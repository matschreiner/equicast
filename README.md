# Introduction
Equicast is a framework for training and running physical forecasting models. The central design principle is modularity:
model architecture, data pipeline, dataset, and training logic are fully decoupled. All major dependencies are injected, and
the framework is built on interface segregation - components depend on abstract base classes rather than concrete
implementations. As long as the interface defined by those base classes is satisfied, autoregressive training and prediction,
multistep training, and integration with CF-compliant tools all follow without any additional glue code.



# Core classes

## dataset

**Responsibility:**
Read and return a timeseries of the physical representation of data. For example this can be graph based or image based. 

**Functions:**
- `__getitem__`: `idx: int -> list[physical_representation]` - return a timeseries of n frames of data in its physical representation, starting from frame idx.

**Instantiation parameters:**
- `datasource`: a datasource object responsible for reading data.
- `num_steps`: timeseries length.

**Notes:**
Should be agnostic towards the data pipeline and shouldn't, for example, be concerned with whether these frames will be used
for autoregressive steps or for multistep inputs. Also shouldn't handle statistics as this couples the dataset
to downstream tasks.


## backbone

**Responsibility:**
Map `backbone_in_representation` to `backbone_out_representation`.

**Second responsibility (possibly):**
Conditional sampling given an input.
For generative models the forward pass used in training is generally different from the sampling process - e.g. skipping the
encoder in a VAE, or SDE/ODE solving in score-based models.

**Functions:**
- `forward`: `backbone_in_representation -> backbone_out_representation`
- `predict` (possibly): `backbone_in_representation -> backbone_out_representation`

**Instantiation parameters:**
Preferably Python primitives.

**Notes:**
Should also be agnostic towards the data pipeline - should only receive input and produce an output. Input can also include noise
for training a VAE/diffusion backbone or communication groups for sharding.
The backbone shouldn't be responsible for anything other than mapping input to output. It should ideally instantiate from Python
primitives. It should handle all processing logic including residual connections, etc., and shouldn't know what is physical space vs.
model space. Shouldn't be responsible for any orchestration or normalization.

Even though the ideal is to instantiate only from primitives, the current backbones take a data handler/feature index at instantiation simply to delegate calculation of input/output dimensions to those,
this is strictly not necessary and reduces portability of the backbones since they cannot instantiate in other frameworks that don't have the `data_handler` concept.


## data_handler

**Responsibility:**
Handle the data pipeline.

**Functions:**
- `prepare_backbone_input`: `physical_representation -> backbone_in_representation`
- `prepare_backbone_target`: `physical_representation -> backbone_target_representation`
- `update_with_output`: `physical_representation, backbone_out_representation -> physical_representation`
    overwrite fields in the physical representation with the backbone output, so we can write predictions back to physical space even though some features are missing from the output
- `to_cf`: `physical_representation -> cf_compliant_data`
    maps from physical representation to a CF-compliant format for visualisation and benchmarking.

**Instantiation parameters:**
- Data statistics.

**Notes:**
Since input and output data for the backbone may be different - different feature sets, number of input frames, injected noise for
generative models, etc. - we need different processing for inputs and outputs. This data handler makes sure that the backbone
model can interface cleanly with the dataset producing timeseries with any representation of physical data.

Since the abstract backbone makes no assumptions about the data format, this class can also be used to handle sharding and calculate communication groups, 
which can be passed along to a backbone that implements sharding functionality.

a few examples of how the data_handler enables different pipelines with the backbone

### Training pipeline
```
backbone_input_t1  = prepare_backbone_input(physical_representation_t1)
backbone_target_t2 = prepare_backbone_target(physical_representation_t2)
backbone_output_t2 = backbone(backbone_input_t1)

loss = loss_fn(backbone_output_t2, backbone_target_t2)
```

### Prediction pipeline
```
backbone_input_t1     = prepare_backbone_input(physical_representation)
backbone_output_t2    = backbone(backbone_input_t1)
updated_physical_representation_t2 = update_with_output(physical_representation_t2, backbone_output_t2) # updated with prognostic/forcing fields from the backbone output
cf_compliant_data_t2  = to_cf(updated_physical_representation_t2)           # optional
```
This ensures the prediction is in physical space and CF-compliant, and is agnostic to architectural details and data processing.

### Autoregressive prediction
```
backbone_input_t1  = prepare_backbone_input(physical_representation_t1)
backbone_output_t2 = backbone(backbone_input_t1)
physical_representation_t2   = update_with_output(physical_representation_t2, backbone_output_t2)
# overwrites forcing and prognostic fields in physical_representation_t2 with predictions from the backbone

backbone_input_t2  = prepare_backbone_input(physical_representation_t2)
backbone_output_t3 = backbone(backbone_input_t2)
...
```
This logic can be used for both multistep training and downstream inference.

`to_cf` maps the physical representation of data to a CF-compliant format, so we can interface our model cleanly with external visualisation and benchmarking tools.


## Model

**Responsibility:**
1. Orchestration of `data_handler` and `backbone` in the training pipeline.
2. Autoregressive prediction with the backbone.

**Instantiation parameters:**
- `backbone`: a backbone object.
- `data_handler`: a data handler object.
- `loss`: loss function that matches the task - KL loss for VAEs, MSE for deterministic models, EquivariantMSE for equivariant models, CRPS, etc.
- `optimization_factory`: factory for creating the model optimizer.
- `scheduler_factory`: factory for creating the learning rate scheduler.

**Notes:**
Owning the `data_handler` and `backbone` creates a portable model that knows how to process any `physical_representation`
and perform downstream tasks, decoupled from the training dataset. To integrate a backbone from another framework, implement
a `data_handler` with the four abstract functions - `prepare_backbone_input`, `prepare_backbone_target`,
`update_with_output`, and `to_cf` that interfaces the model with the dataset, and it will work with the same model and training loop while strictly containing all
architectural details within the backbone.


# Bespoke pipelines

## Equivariant model pipeline

### Dataset

Graphs are created with anemoi-graphs. The data input node names are `'grid'` by default and the dataset puts node features
from the anemoi dataset on `graph['grid'].data`. This is the physical representation of the data. 

### PaiNN backbone

`forward(graph) -> dict[str, Tensor]`

**Required attributes on `graph['grid']`:**
- `input_scalar`:    `Tensor [n_nodes, in_scalar_dim]`   - z-normalized scalar fields (forcing + prognostic scalars)
- `input_vector`:    `Tensor [n_nodes, in_vector_dim, 2]` - norm-normalized vector fields (forcing + prognostic vectors)
- `residual_scalar`: `Tensor [n_nodes, out_scalar_dim]`  - normalized residual scalar fields (prognostic + diagnostic scalars) 
- `residual_vector`: `Tensor [n_nodes, out_vector_dim, 2]` - normalized residual vector fields (prognostic + diagnostic vectors)

Each edge type in the graph also requires `edge_index [2, n_edges]`, `edge_dirs [n_edges, 2]`, and `edge_length [n_edges]`.

**Output:**
- `'scalar'`: `Tensor [n_nodes, out_scalar_dim]`   - scalar predictions with residual added
- `'vector'`: `Tensor [n_nodes, out_vector_dim, 2]` - vector predictions with residual added

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
into physical space on the graph - unpacking `vector` back to `u_p` / `v_p` components.

`to_cf` maps the graph data back to a CF-compliant format.

### Loss

Computes scalar MSE between `output['scalar']` and `target['scalar']`, plus a squared-norm difference between
`output['vector']` and `target['vector']`.


## Standard Message Passing / Transformer backbone pipeline

### Dataset

Graphs are created with anemoi-graphs. The data input node names are `'grid'` by default and the dataset puts node features
from the anemoi dataset on `graph['grid'].data`.

### Backbone
MPNN / Transformer
`forward(graph) -> dict[str, Tensor]`

**Required attributes on `graph['grid']`:**
- `input_scalar`:    `Tensor [n_nodes, in_scalar_dim]`  - z-normalized scalar fields (forcing + prognostic)
- `residual_scalar`: `Tensor [n_nodes, out_scalar_dim]` - normalized prognostic + diagnostic scalars (residual)

Each edge type in the graph also requires `edge_index [2, n_edges]`, `edge_dirs [n_edges, 2]`, and `edge_length [n_edges]`.

**Output:**
- `'scalar'`: `Tensor [n_nodes, out_scalar_dim]` - scalar predictions with residual added

### Data handler

`prepare_backbone_input` packs all physical scalar fields as node features, z-normalizes them into
`input_scalar [n_nodes, in_scalar_dim]`, and attaches them to `graph['grid']`. Prognostic and diagnostic fields are also
written to `residual_scalar` so the backbone can apply a residual connection. There are no vector features.

`prepare_backbone_target` takes the next timestep and extracts normalized target scalars into `'scalar'` in a dictionary,
matching the shape of the backbone's output tensor.

`update_with_output` takes the backbone's `{'scalar'}` output, denormalizes it, and writes the predicted fields back into
physical space on the graph.

`to_cf` maps the graph data back to a CF-compliant format.

### Loss

Computes scalar MSE between `output['scalar']` and `target['scalar']`.


## Possible future pipelines

- **Multidomain models** - the graph provider supplies graphs from multiple domains while the backbone stays the same. Only thing that needs updating is the dataset.
- **Sharding** - the data handler shards the graph and calculates communication groups; the backbone implements a sharded forward pass with all-reduce and/or all-gather ops as needed.
- **Image-based models** - the dataset produces image-based data, the backbone is a CNN or U-Net, and the data handler converts physical data to and from image format.
