import cv2
import time
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
sys.path.append('/Users/ms/code/sense')

global processor_owl
global model_owl
global perform_owl
global processor_depth
global model_depth
global perform_depth
global box_to_mask

from stereo.stereo_utils import *
from stereo.stereo_utils import _convert_to_homogeneous

# export PYTORCH_ENABLE_MPS_FALLBACK=1
perform_depth = 1
DEBUG = False  # Set to True for debug mode, False for normal mode
perform_owl = 1

SHOW_OBJECT_DETECTION = False
SHOW_DEPTH_FRAME_FULL = False
SHOW_ANGLE_FRAME = True
SHOW_DEPTH_FRAME_BOXED = True



cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth Frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('Angle Frame', cv2.WINDOW_NORMAL)

cv2.moveWindow("Object Detection", 0, 0)    # Move to top left corner (0, 0)
cv2.moveWindow("Depth Frame", 0, 400)  # Move to bottom left corner (0, screen_height - window_height)
cv2.moveWindow("Angle Frame", 300, 0)  # Move offsite by 300 pixels to the right of Window 1

# Function to calculate length of a contour


def calculate_contour_length(contour):
    return cv2.arcLength(contour, True)

# Function to check if a contour is outside the padding area and the box

def world_calibrate(cal_img):
    global base_path
    base_path = '../stereo/permanent_calibration_ipad/'
    calibration_file = base_path+'calibration_settings.yaml'
    parse_calibration_settings_file(calibration_file)
    #img_to_calib = load_image('/Users/ms/code/random/saved/ref.jpg')
    img_to_calib = load_image(cal_img)
    R_W0, T_W0, R_W1, T_W1 = get_and_save_camera_extrinsics(base_path, img_to_calib)
    if DEBUG:
        print(R_W0, T_W0, R_W1, T_W1 )

    global P1, P2
    P1 = get_projection_matrix(left_camera_id, base_path)
    P2 = get_projection_matrix(right_camera_id, base_path)

    global P1_orig, P2_orig
    P1_orig = get_projection_matrix_orig(left_camera_id, base_path)
    P2_orig = get_projection_matrix_orig(right_camera_id, base_path)


def capture_webcam_and_display(texts, videofile=None):
    # Open the video capture device (webcam 0)
    speed=10
    if videofile is not None:
        cap = cv2.VideoCapture(videofile)
    else:
        cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        if DEBUG:
            print("Error: Unable to open webcam.")
        return

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    # Main loop to capture and display frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Increment frame count
        frame_count += 1

        if(frame_count  % speed == 0):
            read_image_and_display(frame, texts)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def is_contour_inside_box(contour, image_shape, box, padding_percentage=0.01):
    x, y, w, h = cv2.boundingRect(contour)
    image_height, image_width = image_shape[:2]
    padding_x = int(padding_percentage * image_width)
    padding_y = int(padding_percentage * image_height)

    if (x >= padding_x and y >= padding_y and
            (x + w) <= (image_width - padding_x) and
            (y + h) <= (image_height - padding_y) and
            box[0] <= x and box[1] <= y and box[2] >= (x + w) and box[3] >= (y + h)):
        return True
    return False

# Function to find the longest side of a polygon

def find_average_depth(depth_image, min_depth, max_depth):
    """
    Find the average depth of an object within a specified depth range from a depth image.

    Parameters:
    - depth_image_path: str, path to the depth image file
    - min_depth: int, minimum depth value to consider for the object
    - max_depth: int, maximum depth value to consider for the object

    Returns:
    - average_depth: float, average depth of the object
    """


    # Normalize the depth image if it's in 16-bit format
    if depth_image.dtype == np.uint16:
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Segment the object based on the specified depth range
    object_mask = cv2.inRange(depth_image, min_depth, max_depth)

    # Optionally, refine the mask using morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    # Extract the depth values within the object mask
    object_depth_values = depth_image[object_mask > 0]

    # Calculate the average, minimum, and maximum depth of the object
    if object_depth_values.size == 0:
        raise ValueError("No object found within the specified depth range.")

    average_depth = np.mean(object_depth_values)
    min_depth_found = np.min(object_depth_values)
    max_depth_found = np.max(object_depth_values)

    if DEBUG:
        print(f"Minimum Depth of the Object: {min_depth_found}")
        print(f"Maximum Depth of the Object: {max_depth_found}")




    colored_mask = cv2.applyColorMap(object_mask, cv2.COLORMAP_JET)

    depth_image_uint8 = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Overlay the mask on the depth image for visualization
    depth_colored = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(depth_colored, 0.7, colored_mask, 0.3, 0)

    # Display the images
    if SHOW_DEPTH_FRAME_BOXED:
        cv2.imshow('Depth Image', depth_colored)

    #cv2.imshow('Object Mask', colored_mask)
    #cv2.imshow('Overlay', overlay)


    return average_depth

def find_longest_side(points):
    max_length = 0
    longest_side = None
    for i in range(len(points)):
        length = np.linalg.norm(points[i] - points[(i + 1) % len(points)])
        if length > max_length:
            max_length = length
            longest_side = (points[i], points[(i + 1) % len(points)])
    return longest_side

# Function to calculate the angle between two points


def calculate_angle(point1, point2):
    return np.arctan2(point2[1] - point1[1], point2[0] - point1[0]) * 180 / np.pi

# Function to mask the angle of the longest side of the contour



def load_model_depth():
    global processor_depth
    global model_depth
    processor_depth = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model_depth = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

def load_model_owl():
    global processor_owl
    global model_owl
    processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Check if MPS is available
    if torch.backends.mps.is_available():
        # Set the default device to MPS
        torch.set_default_device("mps")
        if DEBUG:
            print("MPS is available. Using MPS as the default device.")
    else:
        # MPS is not available, so use the CPU
        torch.set_default_device("cpu")
        if DEBUG:
            print("MPS is not available. Using CPU as the default device.")

    model_owl.to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))  # Move the model to the default device
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def owl2(frame, texts, conf):
    inputs = processor_owl(text=texts, images=frame, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device
    outputs = model_owl(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([frame.shape[:2]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=conf)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    largest_box = None
    max_area = 0

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if DEBUG:
            print(f"{text[label]}  confidence {round(score.item(), 3)} at {box}")

        # Calculate area of the box
        area = (box[2] - box[0]) * (box[3] - box[1])

        # Track the largest box
        if area > max_area:
            max_area = area
            largest_box = box

        # Draw bounding box on the frame
        box = [int(b) for b in box]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{text[label]}: {round(score.item(), 3)}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate center coordinates of the box
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2

        # Draw a red circle at the center of the box
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Display center coordinates on the frame
        cv2.putText(frame, f"Center: ({center_x}, {center_y})", (box[0], box[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if SHOW_OBJECT_DETECTION:
            cv2.imshow('Object Detection', frame)

    return largest_box

def owl2_max_conf(frame, texts, conf):
    inputs = processor_owl(text=texts, images=frame, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device
    outputs = model_owl(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([frame.shape[:2]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=conf)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    max_conf_box = None
    max_conf_score = 0

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if DEBUG:
            print(f"{text[label]}  confidence {round(score.item(), 3)} at {box}")

        # Track the box with the highest confidence score
        if score.item() > max_conf_score:
            max_conf_score = score.item()
            max_conf_box = box

    if max_conf_box is not None:
        # Draw the bounding box with the highest confidence score on the frame
        box = [int(b) for b in max_conf_box]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{text[labels[0]]}: {round(max_conf_score, 3)}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate center coordinates of the box
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2

        # Draw a red circle at the center of the box
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Display center coordinates on the frame
        cv2.putText(frame, f"Center: ({center_x}, {center_y})", (box[0], box[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if SHOW_OBJECT_DETECTION:
            cv2.imshow('Object Detection', frame)

    return max_conf_box

def depth_anything(frame):
    inputs = processor_depth(images=frame, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device

    with torch.no_grad():
        outputs = model_depth(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()  # Changed to use CPU for visualization
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth_img = np.array(depth)

    if SHOW_DEPTH_FRAME_FULL:
        cv2.imshow('Depth Frame Full', depth_img)

    return output

SHOW_ANGLE_FRAME=1

def is_contour_at_frame_edge(contour, frame_shape):
    # Check if the contour is at the edge of the frame
    x, y, w, h = cv2.boundingRect(contour)
    height, width = frame_shape[:2]
    return (x == 0 or y == 0 or x + w == width or y + h == height)

def mask_angle(frame, box):
    # Calculate the center of the original box
    center_x = (box[2] + box[0]) / 2
    center_y = (box[3] + box[1]) / 2

    # Enlarge the box by 20% while keeping the center the same
    enlargement_factor = 0.20
    bigger_box = [
        center_x - (center_x - box[0]) * (1 + enlargement_factor),  # x_min
        center_y - (center_y - box[1]) * (1 + enlargement_factor),  # y_min
        center_x + (box[2] - center_x) * (1 + enlargement_factor),  # x_max
        center_y + (box[3] - center_y) * (1 + enlargement_factor)  # y_max
    ]

    box = bigger_box

    x_min, y_min, x_max, y_max = map(int, box)
    cropped_frame = frame[y_min:y_max, x_min:x_max]


    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(blurred_image)

    # Optionally, enhance sharpness using unsharp masking
    blurred = cv2.GaussianBlur(enhanced_gray, (0, 0), 3)
    sharp = cv2.addWeighted(enhanced_gray, 1.5, blurred, -0.5, 0)

    blurred_image=sharp
    #cv2.imshow('Blur Frame', cropped_frame)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Filter contours based on proximity to the edge of the frame and the rectangular box
    valid_contours = contours#[cnt for cnt in contours if is_contour_inside_box(cnt, cropped_frame.shape, box)]
    # Filter contours based on proximity to the edge of the frame and the rectangular box
    #valid_contours = [cnt for cnt in contours if
     #                 is_contour_inside_box(cnt, cropped_frame.shape, box) and not is_contour_at_frame_edge(cnt, cropped_frame.shape)]

    valid_contours = [cnt for cnt in contours if not is_contour_at_frame_edge(cnt, cropped_frame.shape)]
    print(valid_contours)

    # Calculate length for each contour
    lengths = [calculate_contour_length(contour) for contour in valid_contours]
    print(lengths)

    # Find the index of the contour with maximum length
    if lengths:
        max_length_index = lengths.index(max(lengths))
        longest_contour = valid_contours[max_length_index]

        # Approximate the contour with a polygon
        epsilon = 0.01 * cv2.arcLength(longest_contour, True)
        approx = cv2.approxPolyDP(longest_contour, epsilon, True)

        # Find the longest side of the polygon
        longest_side = find_longest_side(approx[:, 0])

        # Draw the longest side of the polygon in yellow
        cv2.line(cropped_frame, tuple(longest_side[0]), tuple(longest_side[1]), (0, 255, 255), thickness=5)  # Yellow thick line

        #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)  # Green box

        # Draw rotated rectangles around each valid contour
        # Draw rotated rectangle around the largest contour

        cv2.drawContours(cropped_frame, valid_contours, -1, (0, 0, 255), thickness=1)

        # Calculate the angle of the yellow line
        angle = calculate_angle(longest_side[0], longest_side[1])

        if SHOW_ANGLE_FRAME:
            cv2.imshow('cropped_frame', cropped_frame)

        if DEBUG:
            print("Angle of the yellow line:", angle)
        return angle
    else:
        if DEBUG:
            print("No contours found.")
        return -1

def show_depth_map(left_image, right_image):
    leftMapX, leftMapY, rightMapX, rightMapY, left_roi, right_roi=get_rectification_maps_and_roi(base_path,'')
    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(4)
    stereoMatcher.setNumDisparities(256)
    stereoMatcher.setBlockSize(7)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(45)

    fixedLeft = cv2.remap(left_image, leftMapX, leftMapY, cv2.INTER_LINEAR)
    fixedRight = cv2.remap(right_image, rightMapX, rightMapY, cv2.INTER_LINEAR)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    # Normalize depth map for visualization
    depth_normalized = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Scale for visualization
    DEPTH_VISUALIZATION_SCALE = 2048
    depth_visualized = depth_normalized * DEPTH_VISUALIZATION_SCALE

    # Convert to grayscale for better contrast
    depth_visualized_gray = (depth_visualized * 255).astype(np.uint8)

    # Apply colormap for better visualization
    depth_colored = cv2.applyColorMap(depth_visualized_gray, cv2.COLORMAP_JET)

    # Display the depth map
    depth_rejoined_image = np.concatenate((depth_colored, depth_colored), axis=1)
    cv2.imshow('Depth Map', depth_rejoined_image)

def find_matching_point(left_box, right_box, left_image, right_image):
    box_size_diff_threshold=0.8
    box_center_diff_threshold=0.1

    left_box_size = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
    right_box_size = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])
    left_point = ([(left_box[0] + left_box[2]) / 2, (left_box[1] + left_box[3]) / 2])
    right_point = ([(right_box[0] + right_box[2]) / 2, (right_box[1] + right_box[3]) / 2])

    # Convert points to homogeneous coordinates
    left_point = _convert_to_homogeneous(left_point)
    right_point = _convert_to_homogeneous(right_point)

    # Calculate the absolute difference between the sizes
    size_difference = abs(left_box_size - right_box_size)

    # Determine the smaller of the two box sizes
    smaller_box_size = min(left_box_size, right_box_size)

    # Calculate the normalized difference
    if smaller_box_size != 0:
        normalized_difference = size_difference / smaller_box_size
    else:
        normalized_difference = float('inf')  # Handle division by zero case

    # Compare with the box_size_diff_threshold
    if (normalized_difference < box_size_diff_threshold):
        return left_point, right_point
    else:
        return -1



def read_image_and_display(frame, texts):
    height, width, _ = frame.shape
    left_image = frame[:, :width // 2]
    right_image = frame[:, width // 2:]

    #show_depth_map(left_image, right_image)

    conf=0.35
    left_box = owl2_max_conf(left_image.copy(), texts, conf)
    right_box = owl2_max_conf(right_image, texts, conf)

    rejoined_image = np.concatenate((left_image, right_image), axis=1)
    cv2.imshow('Rejoined Image', rejoined_image)


    #depth_frame = depth_anything(frame)

    if left_box is not None and right_box is not None:
        mask_angle(left_image, left_box)

        left_point, right_point=find_matching_point(left_box, right_box, left_image, right_image)

        point3D = DLT(P1, P2, left_point, right_point)
        point3D_orig=DLT(P1_orig, P2_orig, left_point, right_point)

        print(f"point3D: ({point3D[0]:.2f}, {point3D[1]:.2f}, {point3D[2]:.2f})  Left: ({left_point[0]:.2f}, {left_point[1]:.2f})  Right: ({right_point[0]:.2f}, {right_point[1]:.2f})")
        print(f"point3D_orig: ({point3D_orig[0]:.2f}, {point3D_orig[1]:.2f}, {point3D_orig[2]:.2f})  Left: ({left_point[0]:.2f}, {left_point[1]:.2f})  Right: ({right_point[0]:.2f}, {right_point[1]:.2f})")

        pointRef = np.array([0, 0, 0])
        # Calculate the Euclidean distance between the two points
        spatial_distance = np.linalg.norm(point3D - pointRef)
        print(f"Spatial Distance from{pointRef} : {spatial_distance:.2f} units")

        spatial_distance_orig = np.linalg.norm(point3D_orig - pointRef)
        print(f"Spatial Distance from{pointRef} : {spatial_distance_orig:.2f} units")


    else:
        print("Cannot detect depth, because object is not recognized by both cameras")

        #angle = mask_angle(frame, box)
    if DEBUG:
        print("angle ", angle)
        #print("average_depth ", average_depth)



if __name__ == "__main__":
    device = load_model_owl()
    load_model_depth()

    world_calibrate('/Users/ms/code/random/saved/ref.jpg')

    texts = [["corner of a cube", "a green surface"]]

    wk = 5000

    # texts = [["a red block", "a green block"]]
    #texts = [["a green block", "a green square"]]

    #texts = [["face", "a human face"]];capture_webcam_and_display(texts)
    texts = [["a yellow wooden cube", "a green cube"]];capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid_green_3.mov")
    #texts = [["a green wooden block"]];
    #capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid_green.mov")
    #capture_webcam_and_display(texts, 0)


    img = cv2.imread("/Users/ms/code/random/saved/saved_image_ang_in_13.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)

    img = cv2.imread("/Users/ms/code/random/saved/saved_image_ang_in_14.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)



    img = cv2.imread("/Users/ms/Downloads/b.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)

    img = cv2.imread("/Users/ms/Downloads/g.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)


    texts = [["books", "a book"]]
    img = cv2.imread("/Users/ms/Downloads/b1.jpeg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)
    img = cv2.imread("/Users/ms/Downloads/b2.jpeg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)
    img = cv2.imread("/Users/ms/Downloads/b3.jpeg")
    read_image_and_display(img, texts)
    cv2.waitKey(wk)
    img = cv2.imread("/Users/ms/Downloads/a1.jpg")
    read_image_and_display(img, texts)

    img = cv2.imread("/Users/ms/code/random/saved/ref.jpg")
    read_image_and_display(img, texts)
