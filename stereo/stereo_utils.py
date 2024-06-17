import cv2
import numpy as np
from scipy import linalg

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P

def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]

def read_camera_parameters(camera_id, base_path):
    with open(f'{base_path}camera{camera_id}_intrinsics.dat', 'r') as inf:
        cmtx = []
        dist = []
        inf.readline()  # Skip header
        for _ in range(3):
            line = [float(val) for val in inf.readline().split()]
            cmtx.append(line)
        inf.readline()  # Skip empty line
        line = [float(val) for val in inf.readline().split()]
        dist.append(line)
    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, base_path):
    with open(f'{base_path}world_to_camera{camera_id}_rot_trans.dat', 'r') as inf:
        R_world_to_camera = []
        t_world_to_camera = []
        inf.readline()  # Skip header
        for _ in range(3):
            R_line = [float(val) for val in inf.readline().split()]
            R_world_to_camera.append(R_line)
        inf.readline()  # Skip empty line
        for _ in range(3):
            t_line = [float(val) for val in inf.readline().split()]
            t_world_to_camera.append(t_line)
    return np.array(R_world_to_camera), np.array(t_world_to_camera)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis=1)
    else:
        return np.concatenate([pts, [1]], axis=0)

def get_projection_matrix(camera_id, base_path):
    cmtx, dist = read_camera_parameters(camera_id, base_path)
    R_world_to_camera, t_world_to_camera = read_rotation_translation(camera_id, base_path)
    P = cmtx @ _make_homogeneous_rep_matrix(R_world_to_camera, t_world_to_camera)[:3, :]
    return P

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image from path: {image_path}")
    return img
