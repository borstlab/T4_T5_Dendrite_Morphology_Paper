import GeoJax as gj
import numpy as np
import pandas as pd

df_point = pd.read_pickle()


def generate_anisotropic_unit_vectors(n, scale=(1.0, 1.0, 1.0), orient_axis=None):
    """
    Generate anisotropic unit vectors on a sphere,
    optionally oriented to a hemisphere.

    Parameters
    ----------
    n : int
        Number of vectors.
    scale : tuple of 3 floats
        Scaling factors for (x, y, z).
    orient_axis : str or None
        Hemisphere to restrict to: one of ['+x','-x','+y','-y','+z','-z'].

    Returns
    -------
    (n,3) ndarray of unit vectors.
    """
    # 1. isotropic unit vectors
    X = np.random.normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # 2. anisotropic scaling
    X *= np.array(scale)

    # 3. renormalize
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # 4. enforce hemisphere restriction
    if orient_axis:
        axis, sign = orient_axis[1].lower(), orient_axis[0]
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        if sign == "-":
            X = np.where(X[:, idx : idx + 1] > 0, -X, X)
        else:  # "+"
            X = np.where(X[:, idx : idx + 1] < 0, -X, X)

    return X


def generate_random_bifurication(n, df, n_type):
    pc1 = df.loc[df_point.Type == n_type, "PC1"].mean()
    pc2 = df.loc[df_point.Type == n_type, "PC2"].mean()
    pc3 = df.loc[df_point.Type == n_type, "PC3"].mean()
    s1 = np.sqrt(pc2 / pc1)
    s2 = np.sqrt(pc3 / pc1)

    vecs_parent = generate_anisotropic_unit_vectors(
        n=n, scale=(1, s1, s2), orient_axis="-x"
    )

    vecs_child1 = generate_anisotropic_unit_vectors(
        n=n, scale=(1, s1, s2), orient_axis="+x"
    )
    vecs_child2 = generate_anisotropic_unit_vectors(
        n=n, scale=(1, s1, s2), orient_axis="+x"
    )
    return vecs_parent, vecs_child1, vecs_child2


def random_bifurication_sum(parent, child1, child2):
    normals = gj.cross(parent, child1)
    gamma = gj.angle(parent, child1, normals)

    normals = gj.cross(parent, child2)
    omega = gj.angle(parent, child2, normals)

    normals = gj.cross(child1, child2)
    theta = gj.angle(child1, child2, normals)

    return gamma + omega + theta


# dihedral beta function
def Dihedral_beta_random(p, c1, c2):

    # bisector of children
    bisector = gj.normalize(c1 + c2)
    # cross of children
    cross = gj.cross(c1, c2)
    # cross of cross and bisector
    normal = gj.cross(bisector, cross)
    # angle between bisector and parent from normal presepctive
    dih_B = gj.angle(p, bisector, normal)

    return dih_B