from equicast.data import feature_router


def test_feature_router_instantiate(dataset, batch, features):
    features = {
        "forcing": ["lsm", "cos_julian_day"],
        "prognostic": ["10u"],
        "diagnostic": ["msl"],
    }
    name_to_idx = dataset.data.name_to_index
    selector = feature_router.FeatureRouter(features, name_to_idx)

    batch = selector.transform(batch)
    assert batch.cond.shape[-1] == 3
    assert batch.target.shape[-1] == 2
