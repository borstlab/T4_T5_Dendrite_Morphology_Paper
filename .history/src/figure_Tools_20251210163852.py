import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .paper_ANOVA import ANOVAModel

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

# suppress the all NaN warning from numpy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

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

    # Calculate the median of the deviations
    with np.errstate(invalid = "ignore"):
        mad_low = np.nanmedian(diffs_low, axis=0)
        mad_high = np.nanmedian(diffs_high, axis=0)

    # convert NaN values to 0
    if data.ndim == 1:
        if np.isnan(mad_low):
            mad_low = 0
        if np.isnan(mad_high):
            mad_high = 0
    else:

        mad_low[np.isnan(mad_low)] = 0
        mad_high[np.isnan(mad_high)] = 0

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

def repeated_measures_PMF_df(ax, df, DV, group_col, groups, colours, x0, x1, num_bins, line_kwargs = dict(), fill_kwargs = dict()):

    # Building observed histogram
    bins = np.linspace(x0, x1, num_bins + 1)
    # Bin centers
    x_values = (bins[:-1] + bins[1:]) / 2

    for i in range(len(groups)):
        s = groups[i]
        c = colours[i]

        sub_df = df.loc[df[group_col] == s]

        # subtypes = df.Subtype.values
        all_ids = sub_df.ID.values
        unique_ids = np.unique(all_ids)
        values = sub_df[DV].values

        # build an array of histograms for some given subtype.
        data = np.zeros((len(unique_ids), num_bins))
        for i in range(len(unique_ids)):

            curr_id = unique_ids[i]
            d = values[np.where(all_ids == curr_id)]

            counts, _ = np.histogram(d, bins = num_bins, range = (x0,x1))
            counts = counts / counts.sum()
            data[i] = counts

        median, mad_low, mad_high = asymmetric_mad(data)

        ax.plot(x_values, median, c = c, label = s, **line_kwargs)
        ax.fill_between(x_values, median - mad_low, median + mad_high, color = c, **fill_kwargs)

def regPlot(ax, df, DV, IV_col, group_col, groups, colours, line_kwargs, point_kwargs, fill_kwargs, legend_kwargs):
    
    formula = f'{DV} ~ C({group_col}) * {IV_col}'

    # fit model - don't bootstrap effect sizes
    model = ANOVAModel(df, formula)
    model.fit(compute_ci = False);

    slopes = model.simple_slopes(IV_col,group_col)
    x_grid = np.linspace(df[IV_col].min(),df[IV_col].max(), 200)

    for i in range(len(groups)):
        g = groups[i]
        c = colours[i]

        intercept = slopes.loc[slopes[group_col] == g,'intercept'].values[0]
        slope = slopes.loc[slopes[group_col] == g,'slope'].values[0]

        # fitted y
        y_fit = intercept + slope * x_grid
        # slopes label
        l = fr"{g}: $\beta$={slope:.3f}"
        # line
        ax.plot(x_grid, y_fit, label = l, c = c, **line_kwargs)

        # ci
        ci_low = intercept + slopes.loc[slopes[group_col] == g,'ci_low'].values[0] * x_grid
        ci_high = intercept + slopes.loc[slopes[group_col] == g,'ci_high'].values[0] * x_grid

        ax.fill_between(
            x_grid, ci_low, ci_high, color = c, **fill_kwargs
        )

    x = df[IV_col].values
    y = df[DV].values

    ax.scatter(x,y, **point_kwargs)

    ax.legend(**legend_kwargs)

def group_radar_plot(ax, df, dv_col, group_col, groups, colours, plot_type = 'bar',bin_range = (-np.pi, np.pi), bins = 60, alpha = 0.7):
    # will iterate over range(len(curr_types))
    for i in range(len(groups)):

        s = groups[i]
        c = colours[i]
        d = df.loc[df[group_col] == s,dv_col].values
        r, theta = np.histogram(d, range = bin_range, bins = bins)
        r = r / r.sum()
        
        if plot_type == 'bar':
            width = theta[1] - theta[0]
            ax.bar(theta[:-1], r, width=width, color=c, alpha=alpha, label = s)
        elif plot_type == 'radar':
            bin_centers = (theta[:-1] + theta[1:]) / 2
            # make sure we have a closed loop
            bin_centers = np.append(bin_centers, bin_centers[0])
            r = np.append(r, r[0])
            ax.plot(bin_centers, 
                r, 
                color = c, 
                alpha = alpha,
                label = s)

    return ax

def radial_PMF(ax, df, given_Subtypes, n_bins, tick_fontsize = 10):

    bins = np.linspace(-np.pi, np.pi, n_bins)
    x_values = (bins[:-1] + bins[1:])/ 2

    for s in given_Subtypes:

        # get line colour
        c = Subtype_colours[np.where(Subtypes == s)][0]
        
        # Internal
        d_internal = df.loc[(df.Subtype == s) & (df.isExternal == False) & (df.Radial_angle_signed != 0), 'Radial_angle_signed'].values
        counts_internal, _ = np.histogram(d_internal, bins=bins)
        counts_internal = counts_internal / counts_internal.sum()

        # External
        d_external = df.loc[(df.Subtype == s) & (df.isExternal == True) & (df.Radial_angle_signed != 0), 'Radial_angle_signed'].values
        counts_external, _ = np.histogram(d_external, bins=bins)
        counts_external = counts_external / counts_external.sum()

        # Append the first point to the end to close the loop
        closed_x_values = np.append(x_values, x_values[0])
        closed_counts_internal = np.append(counts_internal, counts_internal[0])
        closed_counts_external = np.append(counts_external, counts_external[0])

        # Plotting
        ax.plot(closed_x_values, closed_counts_internal, label=s, c = c)
        ax.plot(closed_x_values, closed_counts_external, c = c, ls='--')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Define the positions in radians and the corresponding labels
    tick_locations = [0, np.pi/2, np.pi, -np.pi/2]
    tick_labels = ['0', r'$\frac{\pi}{2}$', r'$\pm\pi$', r'$-\frac{\pi}{2}$']

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, fontsize = tick_fontsize)

    # Explicitly set the theta limits to cover the full circle
    ax.set_thetamin(-180)
    ax.set_thetamax(180)

def angle_heatmap(
    ax, label1, label2, ang_df, bins, label_fontsize=12, add_colorbar=True, cmap="hot"
):
    # Get all x and y
    x = ang_df[label1]
    y = ang_df[label2]

    # Compute the 2D histogram for the heatmap (density)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins], density=True)

    # Create meshgrid for pcolormesh using the provided bins.
    X, Y = np.meshgrid(xedges, yedges)
    pc = ax.pcolormesh(X, Y, heatmap.T, cmap=cmap, shading="auto")

    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)

    # Force the heatmap to be square.
    ax.set_aspect("equal", adjustable="box")

    # Set ticks in degrees
    deg_ticks = [0, np.pi]
    rad_labels = [r"$0$", r"$\pi$"]

    ax.set_xticks(deg_ticks)
    ax.set_yticks(deg_ticks)
    ax.set_xticklabels(rad_labels, fontsize=label_fontsize)
    ax.set_yticklabels(rad_labels, fontsize=label_fontsize)

    ax.tick_params(direction="out")

    # Add colorbar
    if add_colorbar:
        fig = ax.figure
        # Define colorbar axis below current axis
        cax = fig.add_axes([ax.get_position().x0, 
                            ax.get_position().y0 - 0.08, 
                            ax.get_position().width, 
                            0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(pc, cax=cax, orientation='horizontal')
        cbar.set_label("Density", fontsize=10)

    return ax


