import matplotlib.pyplot as plt
import seaborn as sns

def multiline_plot(
    data_dict,
    title="Wealth Medians for Multiple Scenarios",
    xlabel="Age",
    ylabel="Wealth to Income Ratio",
    xlim=(25, 95),
    ylim=(0, 12),
    figsize=(8, 5),
    theme_style="white",      # e.g., "white", "whitegrid", "dark", "ticks"
    context="notebook",           # "notebook", "paper", "poster" also possible
    palette=None,
    linewidth=2.5,
    show_grid=True,
    grid_style="--",
    grid_alpha=0.6,
    grid_color="gray",
    font="Arial",
    font_size=11,
    axis_color="#333333",
    legend=True,
    legend_loc="upper left",
    save_path=None
):
    """
    Create a polished, multi-line chart suitable for executive presentations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary where each key is a label (str) and each value is either:
        - a tuple/list (x_values, y_values), or
        - a DataFrame/dict with columns ["x"] and ["y"].
        
        Example:
            {
                "Baseline": (x_array, y_array),
                "Alternative": {
                    "x": x_array, 
                    "y": y_array
                }
            }

    title : str
        Chart title displayed at the top.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    xlim : tuple (x_min, x_max) or None
        The x-axis range. If None, auto-determined.
    ylim : tuple (y_min, y_max) or None
        The y-axis range. If None, auto-determined.
    figsize : tuple (width, height)
        Figure size in inches.
    theme_style : str
        Seaborn style theme. Examples: "white", "whitegrid", "darkgrid", "ticks".
    context : str
        Seaborn context. Examples: "notebook", "paper", "talk", "poster".
    palette : list of str or None
        List of colors for each line (e.g., ["#005AB5", "#DC3220"]). 
        If None, a default Seaborn palette is used.
    linewidth : float
        Thickness of the plotted lines.
    show_grid : bool
        Whether to display a grid.
    grid_style : str
        Linestyle for grid lines (e.g., "--" for dashed).
    grid_alpha : float
        Transparency of the grid lines (0.0 to 1.0).
    grid_color : str
        Color of the grid lines.
    font : str
        Font family (e.g., "Arial", "Helvetica").
    font_size : int
        Base font size for labels and ticks.
    axis_color : str
        Color for axis spines and text elements.
    legend : bool
        Whether to show a legend.
    legend_loc : str
        Legend location (e.g., "upper left", "best").
    save_path : str or None
        If provided, saves the figure to this path (e.g. "plot.png" or ".pdf").
        
    Returns
    -------
    (fig, ax) : tuple
        Matplotlib Figure and Axes objects, for additional customization if needed.
    """
    
    # Set Seaborn theme and context
    sns.set_theme(style=theme_style, context=context)
    
    # Set global rcParams for a clean, professional look
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = axis_color
    plt.rcParams["axes.labelcolor"] = axis_color
    plt.rcParams["text.color"] = axis_color
    plt.rcParams["xtick.color"] = axis_color
    plt.rcParams["ytick.color"] = axis_color
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["font.size"] = font_size

    # Create figure and axes
    fig, ax = plt.subplots()

    # Handle a custom color palette, if provided
    if palette is not None:
        color_cycle = iter(palette)
    else:
        # If no custom palette is provided, Seaborn automatically uses its default
        color_cycle = None

    for label, data in data_dict.items():
        # Handle the data format
        if isinstance(data, (tuple, list)):
            x_values, y_values = data
        else:
            x_values = data["x"]
            y_values = data["y"]

        # Choose color if a custom palette is given
        if color_cycle is not None:
            color = next(color_cycle)
            ax.plot(
                x_values,
                y_values,
                label=label,
                color=color,
                linewidth=linewidth
            )
        else:
            ax.plot(
                x_values,
                y_values,
                label=label,
                linewidth=linewidth
            )
    
    # Set labels, title, and axes limits
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Configure grid
    if show_grid:
        ax.grid(color=grid_color, linestyle=grid_style, linewidth=0.6, alpha=grid_alpha)

    # Show legend if requested and if there's more than one line
    if legend and len(data_dict) > 0:
        ax.legend(loc=legend_loc, frameon=False, fontsize=12)

    # Save figure if needed
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig, ax
