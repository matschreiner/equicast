import pickle

from pytest import fixture


@fixture
def batch():
    with open("test/res/data.pkl", "rb") as f:
        data = pickle.load(f)
    return data
