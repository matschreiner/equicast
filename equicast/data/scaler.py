class Scaler:
    def __init__(self, statistics):
        self.statistics = statistics
        self.std = self.statistics["stdev"]
        self.mean = self.statistics["mean"]

    def transform(self, data):
        data = (data - self.mean) / self.std
        return data

    def __call__(self, graph):
        return self.transform(graph)
