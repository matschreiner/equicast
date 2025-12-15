from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    forcing: list[str]
    prognostic: list[str]
    diagnostic: list[str]
    forcing_vector: dict[str, list[str]] = field(default_factory=dict)
    prognostic_vector: dict[str, list[str]] = field(default_factory=dict)
    diagnostic_vector: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        for field_name in [
            "forcing_vector",
            "prognostic_vector",
            "diagnostic_vector",
        ]:
            if field_name not in data:
                data[field_name] = {}
        return cls(**data)

    def __repr__(self):
        n_vec = (
            len(self.forcing_vector)
            + len(self.prognostic_vector)
            + len(self.diagnostic_vector)
        )
        return f"FeatureConfig(forcing={len(self.forcing)}, prognostic={len(self.prognostic)}, diagnostic={len(self.diagnostic)}, vectors={n_vec})"
