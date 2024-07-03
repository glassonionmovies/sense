



import numpy as np

# Define the ARM points
arm_points = np.array([
    [9.5, 22],
    [7, 14],
    [-11, 15],
    [-11.5, 22.5],
    [-1, 22],
    [-3, 14],
    [12.5, 29.5],
    [-6, 31.5]
])

# Define the new CAM points
cam_points = np.array([
    [0.16191, -1.4733],
    [4.3496, -0.23822],
    [3.4421, 10.123],
    [-0.11927, 9.3196],
    [1.815, 3.6101],
    [4.7758, 4.0338],
    [-5.4736, -3.5278],
    [-2.9423, 5.9035]
])

def procrustes_analysis(X, Y):
    """
    Perform Procrustes analysis to find the optimal rotation, translation, and scaling
    that maps points in X to points in Y.

    Args:
        X: Source points (CAM points), shape (n, 2)
        Y: Target points (ARM points), shape (n, 2)

    Returns:
        A function that maps a new point in X to the corresponding point in Y.
    """
    # Center the points
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Compute the covariance matrix
    covariance_matrix = np.dot(Y_centered.T, X_centered)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix
    R = np.dot(U, Vt)

    # Compute the scaling factor
    scale = np.sum(S) / np.sum(X_centered ** 2)

    def transform(point):
        """
        Transform a new point using the computed Procrustes transformation.

        Args:
            point: The new point in the source space (CAM point).

        Returns:
            The corresponding point in the target space (ARM point).
        """
        return scale * np.dot(R, point - X_mean) + Y_mean

    return transform


# Perform Procrustes analysis once
transform = procrustes_analysis(cam_points, arm_points)

def cworld_to_aworld(cam_point):
    """
    Convert a CAM point to an ARM point using Procrustes analysis.

    Args:
        cam_point: The CAM point to convert, tuple (x, y).

    Returns:
        The corresponding ARM point, tuple (x, y).
    """
    # Convert cam_point tuple to numpy array
    cam_point_np = np.array(cam_point)

    # Return the transformed ARM point for the given CAM point
    transformed_arm_point = transform(cam_point_np)

    # Convert transformed ARM point back to tuple for consistency
    return transformed_arm_point

# Example usage:
#cam_point_tuple = (3.4421, 10.123)
#arm_point_tuple = cworld_to_aworld(cam_point_tuple)
#print(f"CAM Point: {cam_point_tuple} -> ARM Point: {arm_point_tuple}")
