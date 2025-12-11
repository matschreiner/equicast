class Scaler:
    def __init__(self, statistics):
        self.statistics = statistics
        self.std = self.statistics["stdev"]
        self.mean = self.statistics["mean"]

    def transform(self, data):
        """Scale data to normalized space (z-score normalization)."""
        data = (data - self.mean) / self.std
        return data

    def inverse_transform(self, data):
        """Unscale data back to physical units."""
        data = data * self.std + self.mean
        return data

    def __call__(self, graph):
        return self.transform(graph)
