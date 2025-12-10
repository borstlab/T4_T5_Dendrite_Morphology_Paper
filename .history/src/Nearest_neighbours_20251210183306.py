from scipy.spatial import KDTree
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import jax
from graph_tool.all import Graph
from graph_tool.topology import max_cardinality_matching

from .figure_Tools import point_value_PMF_1darray

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


@jax.jit
def squared_distance_matrix_with_threshold(coords, threshold_sq=jnp.inf):
    """
    Compute squared distance matrix with optional threshold using JAX.

    Parameters
    ----------
    coords : array_like, shape (n, 3)
        3D coordinates.
    threshold_sq : float
        Squared distance threshold. Entries with dist^2 > threshold_sq
        are set to inf. Default: no effective threshold.

    Returns
    -------
    dist_sq : jnp.ndarray, shape (n, n)
        Squared distance matrix with diag = inf and threshold applied.
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)

    # pairwise differences -> (n, n, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)

    # apply threshold (JAX-safe, no Python condition on traced values)
    dist_sq = jnp.where(dist_sq <= threshold_sq, dist_sq, jnp.inf)

    # forbid self-matching: set diagonal to inf
    n = dist_sq.shape[0]
    # safer than adding: directly overwrite diagonal
    dist_sq = dist_sq.at[jnp.arange(n), jnp.arange(n)].set(jnp.inf)

    return dist_sq


def find_optimal_assignment(coords, distance_threshold=None):
    """
    Find a unique pairwise matching of points that:

      1. Maximizes the number of pairs (maximum-cardinality matching),
      2. Among all such matchings, minimizes the total Euclidean distance.

    Unmatched points (e.g. odd n or restrictive threshold) are set to -1.

    Parameters
    ----------
    coords : array_like, shape (n, 3)
        3D coordinates.
    distance_threshold : float, optional
        If given, only pairs with distance <= distance_threshold are allowed.

    Returns
    -------
    pairs : list of (int, int)
        Matched index pairs (i, j) with i < j.
    distances : list of float
        Euclidean distances for each pair, same order as pairs.
    assignments : np.ndarray, shape (n,)
        assignments[i] = j if i matched to j, otherwise -1.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    if n <= 1:
        return [], [], np.full(n, -1, dtype=int)

    # squared distance threshold for JAX
    if distance_threshold is None:
        threshold_sq = jnp.inf
    else:
        threshold_sq = float(distance_threshold) ** 2

    # JAX: squared distances with diag=inf and thresholding
    dist_sq = np.asarray(squared_distance_matrix_with_threshold(coords, threshold_sq))

    # Euclidean distances (inf stays inf)
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(dist_sq)

    # build undirected graph_tool graph
    g = Graph(directed=False)
    g.add_vertex(n)

    # compute max finite distance to define weight transform
    finite_mask = np.isfinite(dist)
    # zero out diagonal or ensure they are not considered
    np.fill_diagonal(finite_mask, False)

    if not finite_mask.any():
        # no admissible edges
        return [], [], np.full(n, -1, dtype=int)

    max_d = float(dist[finite_mask].max())
    C = max_d + 1.0  # strictly larger than any d_ij

    # edge weights: w_ij = C - d_ij
    weight = g.new_edge_property("double")

    for i in range(n):
        for j in range(i + 1, n):
            if finite_mask[i, j]:
                d = float(dist[i, j])
                e = g.add_edge(i, j)
                weight[e] = C - d  # positive, enforces max cardinality + min distance

    if g.num_edges() == 0:
        # threshold or data killed all edges
        return [], [], np.full(n, -1, dtype=int)

    # maximum-weight matching on transformed weights:
    # because of w_ij = C - d_ij and C > max_d,
    # this is equivalent to:
    #   - maximum-cardinality,
    #   - minimum total distance among those.
    match_ep = max_cardinality_matching(g, weight=weight, edges=True)

    pairs = []
    distances = []
    assignments = np.full(n, -1, dtype=int)

    for e in g.edges():
        if match_ep[e]:
            u = int(e.source())
            v = int(e.target())
            i, j = (u, v) if u < v else (v, u)
            pairs.append((i, j))
            distances.append(float(dist[i, j]))
            assignments[i] = j
            assignments[j] = i

    # deterministic ordering
    order = np.argsort([p[0] for p in pairs])
    pairs = [pairs[k] for k in order]
    distances = [distances[k] for k in order]

    return pairs, distances, assignments