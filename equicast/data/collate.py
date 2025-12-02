from copy import deepcopy


def collate_with_graph_single(batch):
    datas = [b["data"] for b in batch]
    graph = batch[0]["graph"]  # keep one, don't collate

    # let PyTorch collate the data (tensors)
    data = torch.utils.data.default_collate(datas)

    return {"data": data, "graph": graph}
