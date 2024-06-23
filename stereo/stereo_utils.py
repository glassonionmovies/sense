import cv2
import numpy as np
from scipy import linalg
import os
import yaml
import sys

DEBUG = False

left_camera_id = 0
right_camera_id = 1


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


def read_rotation_translation(camera_id, base_path, prefix):
    with open(f'{base_path}{prefix}camera{camera_id}_rot_trans.dat', 'r') as inf:
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
    R_world_to_camera, t_world_to_camera = read_rotation_translation(camera_id, base_path, 'latest_')
    P = cmtx @ _make_homogeneous_rep_matrix(R_world_to_camera, t_world_to_camera)[:3, :]
    return P

def get_projection_matrix_orig(camera_id, base_path):
    cmtx, dist = read_camera_parameters(camera_id, base_path)
    R_world_to_camera, t_world_to_camera = read_rotation_translation(camera_id, base_path, '')
    P = cmtx @ _make_homogeneous_rep_matrix(R_world_to_camera, t_world_to_camera)[:3, :]
    return P



def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image from path: {image_path}")
    return img


def get_and_save_camera_extrinsics(base_path, frame):
    image_width = frame.shape[1]
    left_image = frame[:, :image_width // 2]
    right_image = frame[:, image_width // 2:]

    # Read camera parameters for camera 0 and 1 (assuming IDs are 0 and 1)
    cmtx0, dist0 = read_camera_parameters(0, base_path)
    cmtx1, dist1 = read_camera_parameters(1, base_path)

    # Read inter-camera transformation (assuming camera IDs are 0 and 1)
    R1, T1 = read_rotation_translation(1, base_path, '')

    # Create the folder for saving calibration data if it doesn't exist
    camera_params_path = os.path.join(base_path, '')
    if not os.path.exists(camera_params_path):
        os.makedirs(camera_params_path)

    # Get the world to camera 0 rotation and translation
    R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, left_image)

    # Get rotation and translation from world directly to camera 1
    R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, cmtx1, dist1, R1, T1, left_image, right_image)

    save_extrinsic_calibration_parameters(base_path, R_W0, T_W0, R_W1, T_W1, prefix='latest_')

    return R_W0, T_W0, R_W1, T_W1


def get_world_space_origin(cmtx, dist, img_path):
    frame = img_path
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
    ret, rvec, tvec = cv2.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec


calibration_settings = {}


def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    if DEBUG:
        print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    if 'camera0' not in calibration_settings.keys():
        print(
            'camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0, image_path1):
    frame0 = image_path0
    frame1 = image_path1

    unitv_points = 5 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32').reshape((4, 1, 3))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    points, _ = cv2.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame0, origin, _p, col, 2)

    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv2.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv2.line(frame1, origin, _p, col, 2)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(base_path, R0, T0, R1, T1, prefix=''):
    camera0_rot_trans_filename = os.path.join(base_path, prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    camera1_rot_trans_filename = os.path.join(base_path, prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1


import cv2
import numpy as np


def rectify_images(left_image, right_image, base_path, prefix):
    # Print shapes of input images
    print(f"Original Left Image Shape: {left_image.shape}")
    print(f"Original Right Image Shape: {right_image.shape}")

    # Read camera parameters
    cmtx0, dist0 = read_camera_parameters(0, base_path)
    cmtx1, dist1 = read_camera_parameters(1, base_path)

    # Read rotation and translation matrices
    R0, T0 = read_rotation_translation(0, base_path, prefix)
    R1, T1 = read_rotation_translation(1, base_path, prefix)

    # Compute stereo rectification
    R, T = R1 @ R0.T, T1 - R1 @ T0
    R1_, R2_, P1_, P2_, Q, _, _ = cv2.stereoRectify(cmtx0, dist0, cmtx1, dist1, left_image.shape[:2][::-1], R, T)

    # Compute the rectification transform maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(cmtx0, dist0, R1_, P1_,
                                                       (left_image.shape[1], left_image.shape[0]), cv2.CV_32FC1)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(cmtx1, dist1, R2_, P2_,
                                                         (right_image.shape[1], right_image.shape[0]), cv2.CV_32FC1)

    # Apply the rectification transform maps to the images
    rectified_left_image = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
    rectified_right_image = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

    # Print shapes of rectified images
    print(f"Rectified Left Image Shape: {rectified_left_image.shape}")
    print(f"Rectified Right Image Shape: {rectified_right_image.shape}")

    return rectified_left_image, rectified_right_image


def get_rectification_maps_and_roi(base_path, prefix):
    image_size = (1280, 720)
    # Read camera parameters
    cmtx0, dist0 = read_camera_parameters(0, base_path)
    cmtx1, dist1 = read_camera_parameters(1, base_path)

    # Read rotation and translation matrices
    R0, T0 = read_rotation_translation(0, base_path, prefix)
    R1, T1 = read_rotation_translation(1, base_path, prefix)

    # Compute stereo rectification
    R, T = R1 @ R0.T, T1 - R1 @ T0
    R1_, R2_, P1_, P2_, Q, _, _ = cv2.stereoRectify(cmtx0, dist0, cmtx1, dist1, image_size, R, T)

    # Compute the rectification transform maps
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(cmtx0, dist0, R1_, P1_, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(cmtx1, dist1, R2_, P2_, image_size, cv2.CV_32FC1)

    # Compute the region of interest (ROI)
    left_roi = cv2.boundingRect(np.float32([[[0, 0]], [[image_size[0], 0]], [[0, image_size[1]]], [[image_size[0], image_size[1]]]]))
    right_roi = cv2.boundingRect(np.float32([[[0, 0]], [[image_size[0], 0]], [[0, image_size[1]]], [[image_size[0], image_size[1]]]]))

    return left_map_x, left_map_y, right_map_x, right_map_y, left_roi, right_roi
