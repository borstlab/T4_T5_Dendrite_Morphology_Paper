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

def ax_compass(ax,
                start_point = [0,3], 
                arrow_length = 1,
                arrow_scale = 0.3, 
                arrow_text = ["a","b","c","d"], 
                arrow_props = {
                    'head_width': 0.4,
                    'head_length': 0.3,
                    'width': 0.2,
                    'length_includes_head': True, # Ensures arrow tip is at the destination
                    'zorder': 5 # Places arrows on top of other elements
                }, 
                fontsize = 10
            ):
    """Add simple compass to ax. arrow text order = [LEFT, RIGHT, DOWN, UP]
    """
    # Arrow pointing LEFT
    ax.arrow(start_point[0], start_point[1], -arrow_length, 0, color='k', **arrow_props)
    ax.text(start_point[0] - arrow_length - arrow_scale, start_point[1], arrow_text[0], color='k', ha='right', va='center', fontsize=fontsize)
    # Arrow pointing RIGHT
    ax.arrow(start_point[0], start_point[1], arrow_length, 0, color='k', **arrow_props)
    ax.text(start_point[0] + arrow_length + arrow_scale, start_point[1], arrow_text[1], color='k', ha='left', va='center', fontsize=fontsize)
    # Arrow pointing DOWN
    ax.arrow(start_point[0], start_point[1], 0, -arrow_length, color='k', **arrow_props)
    ax.text(start_point[0], start_point[1] - arrow_length - arrow_scale, arrow_text[2], color='k', ha='center', va='top', fontsize=fontsize)
    # Arrow pointing UP
    ax.arrow(start_point[0], start_point[1], 0, arrow_length, color='k', **arrow_props)
    ax.text(start_point[0], start_point[1] + arrow_length + arrow_scale, arrow_text[3], color='k', ha='center', va='bottom', fontsize=fontsize)

def subtype_contour_plot(ax, dictionary):
    """ Helper function for contour plot (fig. 2a) of neuron subtypes from the pre-generated dictionary `Contour_plot.pkl`
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

def asymmetric_mad(data):
    """
    Computes the asymmetric median absolute deviation in a vectorized manner.
    """
    # Calculate the median for each column (axis=0)
    median = np.nanmedian(data, axis=0)

    # Calculate absolute deviations from the median
    diffs = np.abs(data - median)

    # Create boolean masks for values below and above the median
    below = data < median
    above = data > median

    # Where `below` is False, place NaN; otherwise, keep the deviation
    diffs_low = np.where(below, diffs, np.nan)
    # Where `above` is False, place NaN; otherwise, keep the deviation
    diffs_high = np.where(above, diffs, np.nan)

    # Calculate the median of the deviations, ignoring the NaNs
    mad_low = np.nanmedian(diffs_low, axis=0)
    mad_high = np.nanmedian(diffs_high, axis=0)

    return median, mad_low, mad_high

def create_bootstrap_pmf(d, x0=0, x1=1500, num_bins=25, n_bootstrap=1000):

    # Building observed histogram
    bins = np.linspace(x0, x1, num_bins + 1)

    # bin centres
    x_values = (bins[:-1] + bins[1:]) / 2

    # Observed data histogram
    counts, _ = np.histogram(d, range=(x0, x1), bins=num_bins)
    counts = counts / counts.sum()

    # bootstrap
    n_samples = len(d)
    bootstrap_indices = np.random.choice(
        n_samples, size=(n_bootstrap, n_samples), replace=True
    )
    bootstrap_samples = d[bootstrap_indices]

    # histogram computation
    bootstrap_pmfs = np.array(
        [np.histogram(bootstrap_samples[i], bins=bins)[0] for i in range(n_bootstrap)]
    )
    bootstrap_pmfs = bootstrap_pmfs / bootstrap_pmfs.sum(axis=1, keepdims=True)

    # Calculate 95% confidence intervals
    ci_lower = np.percentile(bootstrap_pmfs, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_pmfs, 97.5, axis=0)

    return x_values, counts, ci_lower, ci_upper, bootstrap_pmfs


def point_value_PMF_df(
    ax,
    df,
    DV,
    group_col,
    groups,
    colours,
    x0,
    x1,
    num_bins,
    num_bootstraps,
    line_kwargs=dict(),
    fill_kwargs=dict(),
):

    for i in range(len(groups)):
        t = groups[i]
        c = colours[i]
        # get data
        d = df.loc[df[group_col] == t, DV].values
        # bootstrap pmf
        x, counts, l, u, pmfs = create_bootstrap_pmf(
            d, x0=x0, x1=x1, num_bins=num_bins, n_bootstrap=num_bootstraps
        )
        # plot
        ax.plot(x, counts, c=c, label=t, **line_kwargs)
        ax.fill_between(x, l, u, color=c, **fill_kwargs)
        ax.set_xlim([x0, x1])

    return ax