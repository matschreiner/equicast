import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_field(
    field,
    latlon,
    ax=None,
    title=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    **kwargs,
):
    """
    Plot a field on a map using lat/lon coordinates.

    Parameters
    ----------
    field : array-like
        Field values to plot
    latlon : array-like, shape (n_points, 2)
        Latitude and longitude coordinates in radians or degrees
        If radians: lat in [-π/2, π/2], lon in [0, 2π) or [-π, π]
        If degrees: lat in [-90, 90], lon in [0, 360) or [-180, 180]
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure with PlateCarree projection
    title : str, optional
        Title for the plot
    vmin, vmax : float, optional
        Min/max values for colormap scaling
    cmap : str, optional
        Matplotlib colormap name (default: "viridis")
    **kwargs : dict
        Additional keyword arguments passed to tripcolor

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used for plotting
    im : matplotlib.collections.TriMesh
        The image object (useful for adding colorbars)
    """
    field = np.asarray(field)
    latlon = np.asarray(latlon)

    lat = latlon[:, 0]
    lon = latlon[:, 1]

    # Convert radians to degrees if needed
    if np.abs(lat).max() <= np.pi / 2 + 0.1:
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi

    # Normalize longitude to [-180, 180]
    lon = ((lon + 180) % 360) - 180

    # Create axis if not provided
    if ax is None:
        _, ax = plt.subplots(
            figsize=(12, 6), subplot_kw=dict(projection=ccrs.PlateCarree())
        )

    # Create triangulation and plot
    triang = tri.Triangulation(lon, lat)
    im = ax.tripcolor(
        triang,
        field,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        **kwargs,
    )

    # Add map features
    ax.coastlines()  # type: ignore
    ax.set_global()  # type: ignore
    ax.gridlines(draw_labels=True)  # type: ignore

    if title:
        ax.set_title(title)

    return ax, im


def make_video(
    fields,
    latlon,
    output_path=None,
    ax=None,
    fps=10,
    title_template="Frame {frame}",
    vmin=None,
    vmax=None,
    cmap="viridis",
    figsize=(12, 6),
    dpi=100,
    add_colorbar=True,
    **kwargs,
):
    """
    Create a video from a sequence of field predictions.

    Parameters
    ----------
    fields : list or array-like, shape (n_frames, n_points)
        List of field values for each timestep
    latlon : array-like, shape (n_points, 2)
        Latitude and longitude coordinates (same for all frames)
    output_path : str, optional
        Output video file path (e.g., "output.mp4", "output.gif")
        If None and ax is provided, returns animation without saving
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure with PlateCarree projection
        Useful for multi-panel videos
    fps : int, optional
        Frames per second (default: 10)
    title_template : str or callable, optional
        Title template with {frame} placeholder, or callable that takes frame index
        Examples: "Timestep {frame}", "Hour {frame}", lambda i: f"T+{i*6}h"
    vmin, vmax : float, optional
        Min/max values for colormap. If None, uses global min/max across all frames
    cmap : str, optional
        Matplotlib colormap name (default: "viridis")
    figsize : tuple, optional
        Figure size (default: (12, 6)). Ignored if ax is provided
    dpi : int, optional
        Resolution for video frames (default: 100)
    add_colorbar : bool, optional
        Whether to add a colorbar (default: True). Only used if ax is None
    **kwargs : dict
        Additional keyword arguments passed to plot_field

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
        The animation object. If output_path is provided, also saves to file
    """
    fields = np.asarray(fields)

    # Calculate global vmin/vmax if not provided (ensures consistent colormap)
    if vmin is None:
        vmin = fields.min()
    if vmax is None:
        vmax = fields.max()

    # Setup figure and axis if not provided
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    else:
        fig = ax.get_figure()

    # Initialize plot with first frame using plot_field
    initial_title = (
        title_template(0)
        if callable(title_template)
        else title_template.format(frame=0)
    )

    # Remove 'title' from kwargs if present to avoid conflict
    plot_kwargs = {k: v for k, v in kwargs.items() if k != "title"}

    ax, im = plot_field(
        fields[0],
        latlon,
        ax=ax,
        title=initial_title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **plot_kwargs,
    )

    # Convert latlon once for all frames
    latlon_arr = np.asarray(latlon)
    lat = latlon_arr[:, 0]
    lon = latlon_arr[:, 1]
    if np.abs(lat).max() <= np.pi / 2 + 0.1:
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi
    lon = ((lon + 180) % 360) - 180
    triang = tri.Triangulation(lon, lat)

    # Add colorbar only if we created our own figure
    if own_fig and add_colorbar:
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40)

    def update(frame_idx):
        im.set_array(fields[frame_idx])

        # Update title
        if callable(title_template):
            title = title_template(frame_idx)
        else:
            title = title_template.format(frame=frame_idx)
        ax.set_title(title)

        return (im,)

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(fields), interval=1000 / fps, blit=False
    )

    # Save video if output path provided
    if output_path is not None:
        if output_path.endswith(".gif"):
            anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)

        # Close figure only if we created it
        if own_fig:
            plt.close(fig)

    return anim


def make_comparison_video(
    predictions,
    targets,
    latlon,
    output_path,
    fps=10,
    title_template="Frame {frame}",
    vmin=None,
    vmax=None,
    cmap="viridis",
    figsize=(16, 18),
    dpi=100,
    show_error=True,
    **kwargs,
):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Compute colormap range from ground truth only
    if vmin is None:
        vmin = targets.min()
    if vmax is None:
        vmax = targets.max()

    # ------- Create figure -------
    n_panels = 3 if show_error else 2
    if figsize == (16, 18) and not show_error:
        figsize = (16, 12)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=figsize,
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    if n_panels == 2:
        axes = list(axes)

    # ------- Use plot_field() for all initial images -------
    # Prediction panel
    _, im_pred = plot_field(
        predictions[0],
        latlon,
        ax=axes[0],
        title="Prediction",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    # Target panel
    _, im_tgt = plot_field(
        targets[0],
        latlon,
        ax=axes[1],
        title="Ground Truth",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    # Error panel (optional)
    if show_error:
        _, im_err = plot_field(
            predictions[0] - targets[0],
            latlon,
            ax=axes[2],
            title="Error (Pred - Truth)",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
    else:
        im_err = None

    lat = latlon[:, 0]
    lon = latlon[:, 1]
    if np.abs(lat).max() <= np.pi / 2 + 0.1:
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi
    lon = ((lon + 180) % 360) - 180

    def update(frame_idx):
        im_pred.set_array(predictions[frame_idx])
        im_tgt.set_array(targets[frame_idx])
        if show_error:
            im_err.set_array(predictions[frame_idx] - targets[frame_idx])

        if callable(title_template):
            title = title_template(frame_idx)
        else:
            title = title_template.format(frame=frame_idx)

        axes[0].set_title(f"Prediction – {title}")
        axes[1].set_title(f"Ground Truth – {title}")
        if show_error:
            axes[2].set_title(f"Error – {title}")

        return ()

    # ------- Create animation -------
    anim = FuncAnimation(
        fig,
        update,
        frames=len(predictions),
        interval=1000 / fps,
        blit=False,
    )

    # ------- Save -------
    plt.tight_layout()
    if output_path.endswith(".gif"):
        anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    else:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)

    plt.close(fig)
    return anim
