import cv2
import numpy as np

def calculate_mean_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    image_points = image_points.reshape(-1, 2)
    errors = np.linalg.norm(image_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    return mean_error

def undistort_imgpoint(imgpoint, K, dist):
    """
    Undistorts a distorted image point.

    Args:
        imgpoint (np.ndarray): shape (2,) or (N, 2) image point(s) in pixels.
        K (np.ndarray): 3x3 intrinsic matrix.
        dist (np.ndarray): distortion coefficients.

    Returns:
        np.ndarray: undistorted image point(s) in pixel coordinates.
    """
    imgpoint = np.asarray(imgpoint, dtype=np.float32).reshape(-1, 1, 2)
    
    # Undistort and convert back to pixel coords
    undistorted = cv2.undistortPoints(imgpoint, K, dist, P=K)
    
    return undistorted.reshape(-1, 2) if undistorted.shape[0] > 1 else undistorted.reshape(2,)

def cam_to_image_coordinate(objpoint_cam, K):
    """
    Projects a 3D point in camera coordinates to 2D image coordinates.

    Args:
        objpoint_cam (np.ndarray): 3D point in camera coordinates, shape (3,) or (3,1)
        K (np.ndarray): Camera intrinsic matrix, shape (3,3)

    Returns:
        np.ndarray: 2D image coordinates [u, v], shape (2,)
    """
    # Ensure shape is (3,)
    objpoint_cam = np.asarray(objpoint_cam).reshape(3,)

    # Avoid divide-by-zero if Z is 0
    if objpoint_cam[2] == 0:
        raise ValueError("Z value (depth) cannot be zero for projection.")

    # Perspective projection
    x, y, z = objpoint_cam
    u = (K[0, 0] * x + K[0, 2] * z) / z
    v = (K[1, 1] * y + K[1, 2] * z) / z

    return np.array([u, v])

def world_to_cam_coordinate(R_world2cam, t_world2cam, objpoint):
    """
    Transform a 3D point from world coordinates to camera coordinates.

    Args:
        R_world2cam (np.ndarray): 3x3 rotation matrix.
        t_world2cam (np.ndarray): 3x1 or (3,) translation vector.
        objpoint (np.ndarray): 3D point in world coordinates. Shape (3,) or (3,1).

    Returns:
        np.ndarray: Transformed point in camera coordinates. Shape (3,1).
    """
    # Ensure correct shapes
    R_world2cam = np.asarray(R_world2cam).reshape(3, 3)
    t_world2cam = np.asarray(t_world2cam).reshape(3, 1)
    objpoint = np.asarray(objpoint).reshape(3, 1)

    # Apply transformation
    cam_point = R_world2cam @ objpoint + t_world2cam  # (3,1)

    return cam_point


def image_to_cam_coordinate(imgpoint, depth, camera_matrix, dist_coeffs):
    """
    Convert 2D image point and depth to 3D camera coordinates.
    
    Parameters:
        imgpoint: (x, y) pixel coordinates
        depth: Z in camera frame (must be > 0)
        camera_matrix: Intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients

    Returns:
        X_cam: 3D point in camera coordinates (3x1 ndarray)
    """
    if depth <= 0:
        return None  # Invalid depth

    imgpoint = np.array([[imgpoint]], dtype=np.float32)
    normalized = cv2.undistortPoints(imgpoint, camera_matrix, dist_coeffs)
    x_n, y_n = normalized[0, 0]
    
    X_cam = np.array([[x_n * depth], [y_n * depth], [depth]])
    return X_cam
