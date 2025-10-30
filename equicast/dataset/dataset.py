from anemoi.datasets import open_dataset

#  from torch.utils.data import Dataset


class Dataset:
    def __init__(self, path, forcing=None, prognostic=None, diagnostic=None):
        self.data = open_dataset(path)
        self.data.name_to_index
        self.forcing_idxs = self._get_data_idxs(forcing)
        self.prognostic_idxs = self._get_data_idxs(prognostic)
        self.diagnostic_idxs = self._get_data_idxs(diagnostic)

        pass

    def _get_data_idxs(self, names):
        return [self.data.name_to_index[name] for name in names]

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        frame = self.data[idx].squeeze()

        forcing = frame[self.forcing_idxs]
        prognostic = frame[self.prognostic_idxs]
        diagnostic = frame[self.diagnostic_idxs]

        batch = {
            "forcing": forcing,
            "prognostic": prognostic,
            "diagnostic": diagnostic,
        }

        return batch
