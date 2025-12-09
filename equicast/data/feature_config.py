from dataclasses import dataclass


@dataclass
class FeatureConfig:
    forcing: list[str]
    prognostic: list[str]
    diagnostic: list[str]

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def __repr__(self):
        return f"FeatureConfig(forcing={len(self.forcing)}, prognostic={len(self.prognostic)}, diagnostic={len(self.diagnostic)})"
