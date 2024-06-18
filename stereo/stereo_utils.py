import cv2
import numpy as np
from scipy import linalg
import os
import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os

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
    print((f'{base_path}{prefix}camera{camera_id}_rot_trans.dat', 'r'))
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
    #R_world_to_camera, t_world_to_camera = read_rotation_translation(camera_id, base_path, 'world_to_')
    R_world_to_camera, t_world_to_camera = read_rotation_translation(camera_id, base_path, 'latest_')
    P = cmtx @ _make_homogeneous_rep_matrix(R_world_to_camera, t_world_to_camera)[:3, :]
    return P

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image from path: {image_path}")
    return img

def get_and_save_camera_extrinsics(base_path, frame):
  #cv.imshow('imgyyy', frame)
  """
  This function retrieves the extrinsic camera parameters (rotation and translation)
  for cameras 0 and 1, saves them to files with prefix 'latest', and returns
  the parameters.

  Args:
      base_path: The base path to store the calibration files.
      frame: The input frame to be split.

  Returns:
      A tuple containing four elements:
          - R_W0: Rotation matrix from world to camera 0.
          - T_W0: Translation vector from world to camera 0.
          - R_W1: Rotation matrix from world to camera 1.
          - T_W1: Translation vector from world to camera 1.
  """

  # Split the frame into left and right halves

  image_width = frame.shape[1]

  left_image = frame[:, :image_width // 2]
  right_image = frame[:, image_width // 2:]
  #cv.imshow('img left_image', left_image)

  # Read camera parameters for camera 0 and 1 (assuming IDs are 0 and 1)
  cmtx0, dist0 = read_camera_parameters(0, base_path)
  cmtx1, dist1 = read_camera_parameters(1, base_path)


  # Read inter-camera transformation (assuming camera IDs are 0 and 1)
  R1, T1 = read_rotation_translation(1, base_path,'')
  print(R1, T1)

  # Create the folder for saving calibration data if it doesn't exist
  #camera_params_path = os.path.join(base_path, 'camera_parameters')
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

    frame = img_path#cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    #cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    #cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    #cv.imshow('img', frame)
    #cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec


calibration_settings = {}


# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print(
            'camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = image_path0#cv.imread(image_path0, 1)
    frame1 = image_path1#cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    #cv.imshow('frame0', frame0)
    #cv.imshow('frame1', frame1)
    #cv.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(base_path, R0, T0, R1, T1, prefix=''):

    camera0_rot_trans_filename = os.path.join( base_path, prefix + 'camera0_rot_trans.dat')
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

    # R1 and T1 are just stereo calibration returned values
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
