import numpy as np

# Colours and types

Types = np.array(["T4", "T5"])
Type_colours = np.array(["#17becf", "#ff7f0e"])
Subtypes = np.array(["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])
Subtype_colours = np.array(
    [
        "#1f77b4",
        "#9edae5",
        "#98df8a",
        "#bcbd22",
        "#d62728",
        "#ffbb78",
        "#ff9896",
        "#9467bd",
    ]
)

def add_scale_bar(
    ax, length=5, label="5 μm", location="lower right", offset=0.1, linewidth=2
):
    """
    Add a scale bar to the plot.

    Args:
        ax: matplotlib axis
        length: length of the scale bar in plot units
        label: label text
        location: 'lower right', 'lower left', etc.
        offset: fraction of axis width/height to offset from edges
        linewidth: thickness of the scale bar
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    if location == "lower right":
        start_x = xlim[1] - offset * x_span - length
        start_y = ylim[0] + offset * y_span
    elif location == "lower left":
        start_x = xlim[0] + offset * x_span
        start_y = ylim[0] + offset * y_span
    else:
        raise ValueError("Unsupported location")

    # Plot the scale bar
    ax.plot(
        [start_x, start_x + length], [start_y, start_y], color="black", lw=linewidth
    )

    # Add the label
    ax.text(
        start_x + length / 2,
        start_y - 0.02 * y_span,
        label,
        ha="center",
        va="top",
        fontsize=10,
    )

def subtype_contour_plot(ax, dictionary):
    """ Helper function for contour plot (fig. 2a) of neuron subtypes

    Parameters
    ----------
    ax : _type_
        _description_
    dictionary : _type_
        _description_
    """

    for i in range(len(Subtypes)):

        s = Subtypes[i]

        x = dictionary[s]["x"]
        y = dictionary[s]["y"]
        xx = dictionary[s]["xx"]
        yy = dictionary[s]["yy"]
        zz = dictionary[s]["zz"]

        # Plot the contour
        ax.contour(xx, yy, zz, levels=5, cmap="viridis")
        ax.scatter(
            x, y, s=0.3, color="gray", alpha=0.01, rasterized=True
        ) 
        ax.scatter(
            dictionary["offsets"][i][0], dictionary["offsets"][i][1], c="r", zorder=500
        )

    ax.set_aspect("equal")

    ax.grid(True)
    ax.set_axis_off()
    add_scale_bar(ax, length=1, label=r"$1 \sigma$", location = 'lower left')