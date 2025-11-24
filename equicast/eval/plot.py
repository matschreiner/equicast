import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def plot_global_field(ax, lat, lon, field, title=None, vmin=None, vmax=None):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    field = np.asarray(field)

    triang = tri.Triangulation(lon, lat)

    im = ax.tripcolor(
        triang,
        field,
        shading="gouraud",  # "flat" = piecewise constant, "gouraud" = smooth
        vmin=vmin,
        vmax=vmax,
        transform=ax.projection,
        cmap="viridis",
    )

    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True)

    if title:
        ax.set_title(title)


def make_plot(module, batch, path):
    pred = module(batch)[:, -1].detach().cpu().numpy()
    target = batch["target"][:, -1].detach().cpu().numpy()

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(3, 1, figsize=(20, 10), subplot_kw=dict(projection=proj))

    coords = batch["graph"]["data"].x.T.cpu()

    lat_rad = coords[0]  # [-π/2, π/2]
    lat_deg = lat_rad * 180 / np.pi

    lon_rad = coords[1]  # [0, 2π)
    lon_deg = lon_rad * 180 / np.pi
    lon_deg = ((lon_deg + 180) % 360) - 180  # wrap to [-180, 180)

    plot_global_field(ax[0], lat=lat_deg, lon=lon_deg, field=pred)
    plot_global_field(ax[1], lat=lat_deg, lon=lon_deg, field=target)
    plot_global_field(
        ax[2],
        lat=lat_deg,
        lon=lon_deg,
        field=pred - target,
    )

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
