class Scaler:
    def __init__(self, statistics):
        self.statistics = statistics
        self.std = self.statistics["stdev"]
        self.mean = self.statistics["mean"]

    def transform(self, graph):
        x = graph["data"].raw
        x = (x - self.mean) / self.std
        graph["data"].raw = x
        return graph

    def __call__(self, graph):
        return self.transform(graph)
