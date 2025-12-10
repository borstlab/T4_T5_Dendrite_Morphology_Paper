from scipy.spatial import KDTree
import numpy as np

def nn_PMFs(ax, group, df, x0, x1, n_bins, n_boots):
    """Nearest neighbour PMF plot"""

    # get data arrays
    a = df.loc[df.Subtype == group + "a", ["Root_x", "Root_y", "Root_z"]]
    b = df.loc[df.Subtype == group + "b", ["Root_x", "Root_y", "Root_z"]]
    c = df.loc[df.Subtype == group + "c", ["Root_x", "Root_y", "Root_z"]]
    d = df.loc[df.Subtype == group + "d", ["Root_x", "Root_y", "Root_z"]]

    # KD Trees
    a_tree = KDTree(a.values)
    b_tree = KDTree(b.values)
    c_tree = KDTree(c.values)
    d_tree = KDTree(d.values)

    ### opposite subtypes

    # horizontal
    dists_a, inds = a_tree.query(b.values, k=1)
    dists_b, inds = b_tree.query(a.values, k=1)
    dists = np.hstack((dists_a, dists_b))
    point_value_PMF_1darray(
        ax,
        dists,
        label="",
        x0=x0,
        x1=x1,
        num_bins=n_bins,
        num_bootstraps=n_boots,
        fill_kwargs={"color": "#FFB000", "alpha": 0.3},
        line_kwargs={"color": "gray"},
    )

    # vertical
    dists_a, inds = c_tree.query(d.values, k=1)
    dists_b, inds = d_tree.query(c.values, k=1)
    dists = np.hstack((dists_a, dists_b))
    point_value_PMF_1darray(
        ax,
        dists,
        label="",
        x0=x0,
        x1=x1,
        num_bins=n_bins,
        num_bootstraps=n_boots,
        fill_kwargs={"color": "#FFB000", "alpha": 0.3},
        line_kwargs={"color": "gray", "ls": "--"},
    )

    ### Same Subtypes

    # horizontal
    dists_a, inds = a_tree.query(a.values, k=[2])
    dists_b, inds = b_tree.query(b.values, k=[2])
    dists = np.hstack((dists_a[:, 0], dists_b[:, 0]))
    point_value_PMF_1darray(
        ax,
        dists,
        label="",
        x0=x0,
        x1=x1,
        num_bins=n_bins,
        num_bootstraps=n_boots,
        fill_kwargs={"color": "#00C2A0", "alpha": 0.3},
        line_kwargs={"color": "gray"},
    )

    # vertical
    dists_a, inds = c_tree.query(c.values, k=[2])
    dists_b, inds = d_tree.query(d.values, k=[2])
    dists = np.hstack((dists_a[:, 0], dists_b[:, 0]))
    point_value_PMF_1darray(
        ax,
        dists,
        label="",
        x0=x0,
        x1=x1,
        num_bins=n_bins,
        num_bootstraps=n_boots,
        fill_kwargs={"color": "#00C2A0", "alpha": 0.3},
        line_kwargs={"color": "gray", "ls": "--"},
    )

    ### Orthogonal Subtypes

    # horizontal
    dists_a, inds = a_tree.query(c.values, k=1)
    dists_b, inds = a_tree.query(d.values, k=1)
    dists_c, inds = b_tree.query(c.values, k=1)
    dists_d, inds = b_tree.query(d.values, k=1)
    dists = np.hstack((dists_a, dists_b, dists_c, dists_d))
    point_value_PMF_1darray(
        ax,
        dists,
        label="",
        x0=x0,
        x1=x1,
        num_bins=n_bins,
        num_bootstraps=n_boots,
        fill_kwargs={"color": "#D40078", "alpha": 0.3},
        line_kwargs={"color": "gray"},
    )

    # vertical
    dists_a, inds = c_tree.query(a.values, k=1)
    dists_b, inds = c_tree.query(b.values, k=1)
    dists_c, inds = d_tree.query(a.values, k=1)
    dists_d, inds = d_tree.query(b.values, k=1)
    dists = np.hstack((dists_a, dists_b, dists_c, dists_d))
    point_value_PMF_1darray(
        ax,
        dists,
        label="",
        x0=x0,
        x1=x1,
        num_bins=n_bins,
        num_bootstraps=n_boots,
        fill_kwargs={"color": "#D40078", "alpha": 0.3},
        line_kwargs={"color": "gray", "ls": "--"},
    )

    ### sort out custom legend

    Opposite_patch = Patch(color = "#FFB000", alpha = 0.3, label = "Opposite Subtype")
    Same_patch = Patch(color = "#00C2A0", alpha = 0.3, label = "Same Subtype")
    Orthogonal_patch = Patch(color = "#D40078", alpha = 0.3, label = "Orthogonal Subtype")
    Hoz_line = Line2D([0], [0], color='gray', lw=2, label='Horizontal Subtypes (a,b)')
    Vert_line = Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Vertical Subtypes (c,d)')

    handles = [Opposite_patch, Orthogonal_patch, Same_patch, Hoz_line, Vert_line]
    ax.legend(frameon = False, handles = handles, fontsize = 8)

    return ax