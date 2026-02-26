import fiddle as fdl


class DatasetPath(fdl.Tag):
    "Path to the training dataset zarr store."


class GraphPath(fdl.Tag):
    "Path to the graph .pt file."


class FeatureConfigPath(fdl.Tag):
    "Path to the feature config YAML file."
