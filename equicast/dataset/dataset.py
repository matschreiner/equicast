import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset


class AnemoiDataset(Dataset):
    def __init__(self, path, variables):
        self.data = open_dataset(path)
        self.data.name_to_index
        self.forcing_idxs = self._get_data_idxs(variables.forcing)
        self.prognostic_idxs = self._get_data_idxs(variables.prognostic)
        self.diagnostic_idxs = self._get_data_idxs(variables.diagnostic)

        pass

    def _get_data_idxs(self, names):
        return [self.data.name_to_index[name] for name in names]

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        cond = torch.tensor(self.data[idx].squeeze())
        target = torch.tensor(self.data[idx + 1].squeeze())

        forcing = cond[self.forcing_idxs]
        prognostic = cond[self.prognostic_idxs]
        cond = torch.concatenate([forcing, prognostic])

        prognostic = target[self.prognostic_idxs]
        diagnostic = target[self.diagnostic_idxs]
        target = torch.concatenate([prognostic, diagnostic])

        batch = {"condition": cond.T, "target": target.T}

        return batch
