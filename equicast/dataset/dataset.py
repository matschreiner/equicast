import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from torch.utils.data import Dataset

from equicast import DTYPE
from equicast.data.feature_router import FeatureRouter
from equicast.data.scaler import Scaler
from equicast.utils import utils


class AnemoiDataset(Dataset):
    def __init__(self, path, graph_provider):
        super().__init__()
        self.data = open_dataset(path)
        self.statistics = utils.cast_dict(self.data.statistics, torch.Tensor)
        self.name_to_index = self.data.name_to_index
        self.graph_provider = graph_provider
        self.scaler = Scaler(self.statistics)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        graph = self.graph_provider.get_graph(idx)

        raw = torch.Tensor(self.data[idx : idx + 2]).squeeze().permute([2, 0, 1])
        graph["data"].raw = raw
        return graph
        #  return {"cond": cond, "target": target, "name_to_index": self.name_to_index}

    #  def _get_single_item(self, idx):
    #      graph = self.graph_provider.get_graph(idx)
    #      state = torch.Tensor(self.data[idx]).squeeze().T
    #      graph["data"].state = state
    #
    #      return graph


"""
    González-Flórez, C.1; Baordo, F.1; Guedj S.2 Geer, A.3
    1 Danish Meteorological Institute, Copenhagen, Denmark.
    2 MET Norway, Oslo, Norway.
    3 ECMWF Reading, UK
    Machine Learning (ML) is increasingly used in Earth System Observation and Prediction, offering new opportunities to improve the use of satellite radiances in numerical weather prediction (NWP) systems. In polar regions, many surface-sensitive microwave channels remain unassimilated due to limitations in modelling the radiative transfer of snow and sea ice. To address this challenge, ECMWF has developed a hybrid empirical–physical ML observation operator for sea ice, initially trained on AMSR2 radiances and later extended to handle additional conical-scanning sensors. The hybrid approach uses a two-layer neural-network model to generate the sea-ice emissivity, while the surface emissivity of the ocean and atmospheric radiative transfer rely on physical modelling.
    Building on the latest ECMWF developments, this study extends the hybrid ML framework to AMSU‑A, a cross-track scanning sensor, addressing future observational needs in the context of the upcoming EUMETSAT Polar System – Sterna programme. Using data extracted from the ECMWF Integrated Forecasting System (IFS) 49r1, including AMSU-A observations from Metop-B and Metop-C, the results demonstrate that the hybrid model can be successfully trained on the low-peaking AMSU-A channels. The trained model accurately reproduces observed brightness temperatures over both Arctic and Antarctic sea ice.
    These findings provide a foundation for further extending the hybrid approach to other cross-track sensors, such as MHS. In the longer term, this development could enable the assimilation of surface-sensitive AMSU-A channels, and eventually other similar channels, over sea-ice-affected polar regions within the IFS, offering the potential to improve analysis accuracy and forecast skill.
"""
