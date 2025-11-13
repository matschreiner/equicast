import tempfile
import webbrowser

import numpy as np
import plotly.graph_objects as go


def plot_field_interactive(
    field, coords, n_lat=256, n_lon=512, center_lon=90, center_lat=0
):
    field = field.detach()
    lon = ((coords[:, 1].numpy() * 180 / np.pi + 180) % 360) - 180
    lat = np.clip(coords[:, 0].numpy() * 180 / np.pi, -90, 90)
    val = field.numpy()

    lon_edges = np.linspace(-180, 180, n_lon)
    lat_edges = np.linspace(-90, 90, n_lat)

    H, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges), weights=val)
    C, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges))
    Z = np.divide(H, C, out=np.full_like(H, np.nan, dtype=float), where=C > 0)
    Z = np.nan_to_num(Z, nan=np.nanmean(Z))

    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    Lon, Lat = np.meshgrid(lon_c, lat_c)

    Lon = np.hstack((Lon, Lon[:, [0]] + 360))
    Lat = np.hstack((Lat, Lat[:, [0]]))
    Z = np.hstack((Z, Z[:, [0]]))

    Lon_r = np.radians(Lon)
    Lat_r = np.radians(Lat)
    x = np.cos(Lat_r) * np.cos(Lon_r)
    y = np.cos(Lat_r) * np.sin(Lon_r)
    z = np.sin(Lat_r)

    fig = go.Figure(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=Z,
            colorscale="Viridis",
            cmin=np.nanmin(Z),
            cmax=np.nanmax(Z),
            showscale=True,
        )
    )

    lon_rad = np.radians(center_lon)
    lat_rad = np.radians(center_lat)
    cx = np.cos(lat_rad) * np.cos(lon_rad)
    cy = np.cos(lat_rad) * np.sin(lon_rad)
    cz = np.sin(lat_rad)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=cx * 2, y=cy * 2, z=cz * 2)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Interactive Spherical Heatmap",
    )

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        fig.write_html(f.name, include_plotlyjs="cdn")
        webbrowser.open(f.name)

    __import__("sys").exit()  # TODO delme
