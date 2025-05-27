import numpy as np

def tilde(v: np.array((3))) -> np.array((3,3)):
    """Tilde-Operator: skew symmetric matrix that replaces a cross product
    Parameters
    ----------
    v: np.array((3))
        vector that should be used for the cross product

    Returns
    ---------
    T: np.array((3,3))
        skew symmetric matrix that replaces the cross product
    """


    ###Workspace Start###
    #T = ...
    #return T
    T = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return T
    ###Workspace End###
