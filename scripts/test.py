import argparse

from anemoi.datasets import open_dataset

from equicast.dataset import Dataset


def main(args):
    path = "/home/masc/remote/data/era5_anemoi.zarr"
    diagnostic = ["t2m"]
    forcing = ["cos_latitude"]
    prognostic = ["sp"]

    ds = Dataset(path, forcing=forcing, prognostic=prognostic, diagnostic=diagnostic)

    for d in ds:
        __import__("pdb").set_trace()  # TODO delme


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('arg', arg)
    # argparser.add_argument('--kwarg', kwarg)
    args = argparser.parse_args()
    main(args)
