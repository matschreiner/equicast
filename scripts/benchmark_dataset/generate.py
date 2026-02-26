"""Save the last 5 frames of the ERA5 zarr dataset as a small benchmark zarr."""

import zarr

DATASET_PATH = "storage/era5-o96-2024-tail200-6h.zarr"
OUT_PATH = "resources/benchmark.zarr"
N_FRAMES = 5


def main():
    src = zarr.open(DATASET_PATH, "r")
    dst = zarr.open(OUT_PATH, "w")

    for key in src.keys():
        arr = src[key]
        if key == "data":
            # data has shape [time, features, 1, nodes] — slice time dimension
            dst.array(key, data=arr[-N_FRAMES:], chunks=arr.chunks, dtype=arr.dtype)
        else:
            zarr.copy(arr, dst, name=key)

    dst.attrs.update(src.attrs)
    print(f"Saved {N_FRAMES} frames → {OUT_PATH}")


if __name__ == "__main__":
    main()
