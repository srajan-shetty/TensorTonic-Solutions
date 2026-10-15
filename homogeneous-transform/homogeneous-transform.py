import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    points = np.asarray(points)
    Tr = np.asarray(T)
    if points.ndim ==1:
        N = points.shape[0]
        points = np.append(points, 1)
        output = (Tr@points.T).T[:N]
    else:
        N = points.shape[1]
        points = np.column_stack([points, np.ones(len(points))])
        output = (Tr@points.T).T[:,:N]
    return output
    