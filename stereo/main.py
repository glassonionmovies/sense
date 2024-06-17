import cv2
import numpy as np
from stereo_utils import DLT, get_projection_matrix, _convert_to_homogeneous, load_image

def main():
    # Path to image file and camera parameters
    image_path = '/Users/ms/code/random/joined/camera0.png'
    base_path = 'permanent_calibration_ipad/'

    # Load the specified image
    img = load_image(image_path)
    if img is None:
        return

    height, width, _ = img.shape

    # Divide the image into left and right halves (example division)
    left_image = img[:, :width // 2]
    right_image = img[:, width // 2:]

    # List to store selected points
    points = {'left': [], 'right': []}

    # Mouse click event handler for left image
    def click_event_left(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points['left'].append((x, y))
            cv2.circle(left_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Left Image', left_image)
            if len(points['left']) == 1:
                print(f"Selected Point L1: {points['left'][0]}")
            elif len(points['left']) == 2:
                print(f"Selected Point L2: {points['left'][1]}")

    # Mouse click event handler for right image
    def click_event_right(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points['right'].append((x, y))
            cv2.circle(right_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Right Image', right_image)
            if len(points['right']) == 1:
                print(f"Selected Point R1: {points['right'][0]}")
            elif len(points['right']) == 2:
                print(f"Selected Point R2: {points['right'][1]}")

    # Display the left and right images
    cv2.imshow('Left Image', left_image)
    cv2.imshow('Right Image', right_image)
    cv2.setMouseCallback('Left Image', click_event_left)
    cv2.setMouseCallback('Right Image', click_event_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the selected points and calculate the 3D points
    if len(points['left']) == 2 and len(points['right']) == 2:
        # Get projection matrices for both cameras
        left_camera_id = 0
        right_camera_id = 1
        P1 = get_projection_matrix(left_camera_id, base_path)
        P2 = get_projection_matrix(right_camera_id, base_path)

        # Convert points to homogeneous coordinates
        pointL1 = _convert_to_homogeneous(points['left'][0])
        pointR1 = _convert_to_homogeneous(points['right'][0])
        pointL2 = _convert_to_homogeneous(points['left'][1])
        pointR2 = _convert_to_homogeneous(points['right'][1])

        # Calculate the 3D points using DLT
        point3D1 = DLT(P1, P2, pointL1, pointR1)
        point3D2 = DLT(P1, P2, pointL2, pointR2)

        # Calculate the spatial distance between the 3D points
        spatial_distance = np.linalg.norm(point3D2 - point3D1)

        # Print the 3D points and spatial distance
        print(f"Calculated 3D Point Pair 1 in World Coordinates:")
        print(f"Point 1: {np.round(point3D1, 2)}")
        print(f"Point 2: {np.round(point3D2, 2)}")
        print(f"Spatial Distance: {np.round(spatial_distance, 2)} units")

        # Print the point pairs in the specified format
        print(f"Point 1: Left {points['left'][0]} Right {points['right'][0]}")
        print(f"Point 2: Left {points['left'][1]} Right {points['right'][1]}")

        # Display the 3D points on the images
        cv2.putText(left_image, f"{np.round(point3D1, 2)}", points['left'][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(left_image, f"{np.round(point3D2, 2)}", points['left'][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(right_image, f"{np.round(point3D1, 2)}", points['right'][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(right_image, f"{np.round(point3D2, 2)}", points['right'][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw arrows from point 1 to point 2 and display spatial distance
        cv2.arrowedLine(left_image, points['left'][0], points['left'][1], (0, 0, 255), 2)
        cv2.arrowedLine(right_image, points['right'][0], points['right'][1], (0, 0, 255), 2)

        # Calculate and display spatial distance
        spatial_distance_text = f"Spatial Distance: {np.round(spatial_distance, 2)} units"
        cv2.putText(left_image, spatial_distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right_image, spatial_distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the images with 3D points, arrows, and spatial distance
        cv2.imshow('Left Image with 3D Points', left_image)
        cv2.imshow('Right Image with 3D Points', right_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please select two points on both the left and right images.")

if __name__ == "__main__":
    main()
