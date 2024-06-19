import cv2
import numpy as np
from stereo_utils import load_image, parse_calibration_settings_file, get_and_save_camera_extrinsics, get_projection_matrix, DLT, _convert_to_homogeneous

def main():
    # Path to image file and camera parameters
    image_path = '/Users/ms/code/random/joined/camera2.png'
    base_path = 'permanent_calibration_ipad/'
    calibration_file = '/Users/ms/code/random/python_stereo_camera_calibrate/calibration_settings.yaml'

    # Load the specified image
    img = load_image(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return

    try:
        parse_calibration_settings_file(calibration_file)
    except Exception as e:
        print(f"Error parsing calibration file: {e}")
        return

    try:
        get_and_save_camera_extrinsics(base_path, img)
    except Exception as e:
        print(f"Error getting and saving camera extrinsics: {e}")
        return

    height, width, _ = img.shape

    # Divide the image into left and right halves
    left_image = img[:, :width // 2].copy()
    right_image = img[:, width // 2:].copy()

    # List to store selected points
    points = {'left': [], 'right': []}

    def click_event_left(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points['left'].append((x, y))
            cv2.circle(left_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Left Image', left_image)
            print(f"Selected Point L{len(points['left'])}: {points['left'][-1]}")

    def click_event_right(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points['right'].append((x, y))
            cv2.circle(right_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Right Image', right_image)
            print(f"Selected Point R{len(points['right'])}: {points['right'][-1]}")

    # Display the left and right images
    cv2.imshow('Left Image', left_image)
    cv2.imshow('Right Image', right_image)
    cv2.setMouseCallback('Left Image', click_event_left)
    cv2.setMouseCallback('Right Image', click_event_right)
    print("Select two points on both left and right images.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points['left']) == 2 and len(points['right']) == 2:
        try:
            P1 = get_projection_matrix(0, base_path)
            P2 = get_projection_matrix(1, base_path)

            pointL1 = _convert_to_homogeneous(points['left'][0])
            pointR1 = _convert_to_homogeneous(points['right'][0])
            pointL2 = _convert_to_homogeneous(points['left'][1])
            pointR2 = _convert_to_homogeneous(points['right'][1])

            point3D1 = DLT(P1, P2, pointL1, pointR1)
            point3D2 = DLT(P1, P2, pointL2, pointR2)

            spatial_distance = np.linalg.norm(point3D2 - point3D1)

            print(f"Calculated 3D Point Pair 1 in World Coordinates:")
            print(f"Point 1: {np.round(point3D1, 2)}")
            print(f"Point 2: {np.round(point3D2, 2)}")
            print(f"Spatial Distance: {np.round(spatial_distance, 2)} units")

            # Draw and annotate 3D points on images
            annotate_images_with_3d_points(left_image, right_image, points, point3D1, point3D2, spatial_distance)
        except Exception as e:
            print(f"Error in 3D point calculation: {e}")
    else:
        print("Please select two points on both the left and right images.")

def annotate_images_with_3d_points(left_image, right_image, points, point3D1, point3D2, spatial_distance):
    cv2.putText(left_image, f"{np.round(point3D1, 2)}", points['left'][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(left_image, f"{np.round(point3D2, 2)}", points['left'][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(right_image, f"{np.round(point3D1, 2)}", points['right'][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(right_image, f"{np.round(point3D2, 2)}", points['right'][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.arrowedLine(left_image, points['left'][0], points['left'][1], (0, 0, 255), 2)
    cv2.arrowedLine(right_image, points['right'][0], points['right'][1], (0, 0, 255), 2)

    spatial_distance_text = f"Spatial Distance: {np.round(spatial_distance, 2)} units"
    cv2.putText(left_image, spatial_distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right_image, spatial_distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Left Image with 3D Points', left_image)
    cv2.imshow('Right Image with 3D Points', right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
