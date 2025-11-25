import torch
from anemoi.utils.config import DotDict


class FeatureRouter(torch.nn.Module):
    def __init__(self, features, name_to_index):
        self.features = features
        self.name_to_index = name_to_index

        self.forcing_idxs = self._get_data_idxs(features["forcing"])
        self.prognostic_idxs = self._get_data_idxs(features["prognostic"])
        self.diagnostic_idxs = self._get_data_idxs(features["diagnostic"])

    def _get_data_idxs(self, names):
        return [self.name_to_index[name] for name in names]

    def transform(self, batch):
        cond = batch.data[..., 0, :, :]
        target = batch.data[..., 1, :, :]

        forcing = cond[..., self.forcing_idxs]
        prognostic = cond[..., self.prognostic_idxs]
        cond = torch.concatenate([forcing, prognostic], dim=-1)

        prognostic = target[..., self.prognostic_idxs]
        diagnostic = target[..., self.diagnostic_idxs]
        target = torch.concatenate([prognostic, diagnostic], dim=-1)

        batch["cond"] = cond
        batch["target"] = target

        return DotDict(
            {
                "cond": cond,
                "target": target,
                "idx": batch["idx"],
            }
        )
