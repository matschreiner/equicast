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
    center=None,
    extent=None,
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
    center : tuple (lat, lon), optional
        Center point for zoomed view in degrees
    extent : tuple (width, height), optional
        Width and height of zoomed view in degrees. Requires center.
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

    # Set extent (zoom) or global view
    if center is not None and extent is not None:
        center_lat, center_lon = center
        width, height = extent
        ax.set_extent(
            [
                center_lon - width / 2,
                center_lon + width / 2,
                center_lat - height / 2,
                center_lat + height / 2,
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
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


def plot_field_with_vectors(
    graph,
    data_handler,
    vector_field: str,
    scalar_field: str,
    ax=None,
    subsample: int = 20,
    arrow_length: float = 1.0,
    center=None,
    extent=None,
    normalize_vectors: bool = False,
    **kwargs,
):
    """
    Plot a scalar field with vector arrows overlaid.

    Parameters
    ----------
    graph : HeteroData
        Graph containing grid data with lat/lon coordinates
    data_handler : EquivariantGraphDataHandler
        Data handler with feature name mappings and vector configuration
    vector_field : str
        Name of vector field (e.g., "wind_500"), must be in prognostic_vector config
    scalar_field : str
        Name of scalar field (e.g., "t_500")
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure
    subsample : int, optional
        Only plot every Nth vector for clarity (default: 20)
    arrow_length : float, optional
        Arrow length multiplier (default: 1.0). Higher = longer arrows.
    center : tuple (lat, lon), optional
        Center point for zoomed view in degrees
    extent : tuple (width, height), optional
        Width and height of zoomed view in degrees
    normalize_vectors : bool, optional
        If True, normalize vectors to unit length (show direction only)
    **kwargs : dict
        Additional keyword arguments passed to plot_field

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used for plotting
    im : matplotlib.collections.TriMesh
        The scalar field image
    quiv : matplotlib.quiver.Quiver
        The quiver plot object
    """
    nodes = data_handler.nodes

    # Extract lat/lon coordinates
    latlon = graph[nodes].x.numpy()

    # Get scalar field data
    scalar_idx = data_handler.name_to_index[scalar_field]
    scalar_data = graph[nodes].data[..., scalar_idx].numpy()

    # Get vector field index (position in prognostic_vector dict)
    vector_names = list(data_handler.feature_config.prognostic_vector.keys())
    if vector_field not in vector_names:
        raise ValueError(
            f"Vector field '{vector_field}' not found in prognostic_vector config. "
            f"Available: {vector_names}"
        )
    vector_idx = vector_names.index(vector_field)

    # Extract vector components from input_vector tensor [nodes, num_vectors, 2]
    vectors = graph[nodes]["input_vector"]
    u = vectors[:, vector_idx, 0].numpy()
    v = vectors[:, vector_idx, 1].numpy()

    # Normalize vectors to unit length if requested
    if normalize_vectors:
        magnitude = np.sqrt(u**2 + v**2)
        magnitude = np.where(magnitude == 0, 1, magnitude)  # Avoid division by zero
        u = u / magnitude
        v = v / magnitude

    # Plot scalar field background
    ax, im = plot_field(scalar_data, latlon, ax=ax, center=center, extent=extent, **kwargs)

    # Convert lat/lon for quiver plot (same conversion as plot_field)
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    if np.abs(lat).max() <= np.pi / 2 + 0.1:
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi
    lon = ((lon + 180) % 360) - 180

    # Subsample vectors for visibility
    indices = np.arange(0, len(lat), subsample)

    # Convert arrow_length to quiver scale (inverse relationship)
    # Base scale of 50 gives reasonable arrows, divide by arrow_length
    quiver_scale = 50.0 / arrow_length

    # Add quiver plot
    quiv = ax.quiver(
        lon[indices],
        lat[indices],
        u[indices],
        v[indices],
        transform=ccrs.PlateCarree(),
        scale=quiver_scale,
        scale_units="inches",
        color="black",
        alpha=0.7,
        width=0.002,
    )

    return ax, im, quiv


def make_video_with_vectors(
    graphs,
    data_handler,
    vector_field: str,
    scalar_field: str,
    output_path: str,
    fps: int = 5,
    subsample: int = 20,
    arrow_length: float = 1.0,
    title_template="Frame {frame}",
    vmin: float = None,
    vmax: float = None,
    cmap: str = "coolwarm",
    figsize: tuple = (12, 6),
    dpi: int = 100,
    center=None,
    extent=None,
    normalize_vectors: bool = False,
):
    """
    Create a video of scalar field with vector arrows over time.

    Parameters
    ----------
    graphs : list
        List of prepared HeteroData graphs (with input_vector populated)
    data_handler : EquivariantGraphDataHandler
        Data handler with feature name mappings
    vector_field : str
        Name of vector field (e.g., "wind_500")
    scalar_field : str
        Name of scalar field (e.g., "t_500")
    output_path : str
        Output video file path (e.g., "output.mp4", "output.gif")
    fps : int, optional
        Frames per second (default: 5)
    subsample : int, optional
        Only plot every Nth vector (default: 20)
    arrow_length : float, optional
        Arrow length multiplier (default: 1.0). Higher = longer arrows.
    title_template : str or callable, optional
        Title template with {frame} placeholder
    vmin, vmax : float, optional
        Min/max values for colormap
    cmap : str, optional
        Colormap name (default: "coolwarm")
    figsize : tuple, optional
        Figure size (default: (12, 6))
    dpi : int, optional
        Resolution for video frames (default: 100)
    center : tuple (lat, lon), optional
        Center point for zoomed view in degrees
    extent : tuple (width, height), optional
        Width and height of zoomed view in degrees
    normalize_vectors : bool, optional
        If True, normalize vectors to unit length (show direction only)

    Returns
    -------
    anim : FuncAnimation
        The animation object
    """
    nodes = data_handler.nodes

    # Get indices for scalar and vector fields
    scalar_idx = data_handler.name_to_index[scalar_field]
    vector_names = list(data_handler.feature_config.prognostic_vector.keys())
    if vector_field not in vector_names:
        raise ValueError(
            f"Vector field '{vector_field}' not found. Available: {vector_names}"
        )
    vector_idx = vector_names.index(vector_field)

    # Extract all scalar and vector data
    scalar_frames = []
    u_frames = []
    v_frames = []

    for graph in graphs:
        scalar_frames.append(graph[nodes].data[..., scalar_idx].numpy())
        vectors = graph[nodes]["input_vector"]
        u_frames.append(vectors[:, vector_idx, 0].numpy())
        v_frames.append(vectors[:, vector_idx, 1].numpy())

    scalar_frames = np.array(scalar_frames)
    u_frames = np.array(u_frames)
    v_frames = np.array(v_frames)

    # Normalize vectors to unit length if requested
    if normalize_vectors:
        magnitude = np.sqrt(u_frames**2 + v_frames**2)
        magnitude = np.where(magnitude == 0, 1, magnitude)  # Avoid division by zero
        u_frames = u_frames / magnitude
        v_frames = v_frames / magnitude

    # Get lat/lon from first graph
    latlon = graphs[0][nodes].x.numpy()
    lat = latlon[:, 0]
    lon = latlon[:, 1]

    # Convert to degrees if needed
    if np.abs(lat).max() <= np.pi / 2 + 0.1:
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi
    lon = ((lon + 180) % 360) - 180

    # Calculate vmin/vmax if not provided
    if vmin is None:
        vmin = scalar_frames.min()
    if vmax is None:
        vmax = scalar_frames.max()

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # Initial plot
    triang = tri.Triangulation(lon, lat)
    im = ax.tripcolor(
        triang,
        scalar_frames[0],
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )

    ax.coastlines()

    # Set extent (zoom) or global view
    if center is not None and extent is not None:
        center_lat, center_lon = center
        width, height = extent
        ax.set_extent(
            [
                center_lon - width / 2,
                center_lon + width / 2,
                center_lat - height / 2,
                center_lat + height / 2,
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_global()

    ax.gridlines(draw_labels=True)

    # Subsample indices for vectors
    indices = np.arange(0, len(lat), subsample)

    # Convert arrow_length to quiver scale (inverse relationship)
    quiver_scale = 10.0 / arrow_length

    # Initial quiver
    quiv = ax.quiver(
        lon[indices],
        lat[indices],
        u_frames[0][indices],
        v_frames[0][indices],
        transform=ccrs.PlateCarree(),
        scale=quiver_scale,
        scale_units="inches",
        color="black",
        alpha=0.7,
        width=0.002,
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40)

    # Set initial title
    initial_title = (
        title_template(0)
        if callable(title_template)
        else title_template.format(frame=0)
    )
    ax.set_title(initial_title)

    def update(frame_idx):
        # Update scalar field
        im.set_array(scalar_frames[frame_idx])

        # Update vectors
        quiv.set_UVC(u_frames[frame_idx][indices], v_frames[frame_idx][indices])

        # Update title
        if callable(title_template):
            title = title_template(frame_idx)
        else:
            title = title_template.format(frame=frame_idx)
        ax.set_title(title)

        return im, quiv

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(graphs), interval=1000 / fps, blit=False
    )

    # Save
    if output_path.endswith(".gif"):
        anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    else:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)

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
