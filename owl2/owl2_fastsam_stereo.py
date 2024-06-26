import cv2
import time
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image

import os
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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

from FastSAM.fastsam import FastSAM, FastSAMPrompt


fast_same_model = FastSAM('FastSAM/weights/FastSAM-x.pt')


# export PYTORCH_ENABLE_MPS_FALLBACK=1
perform_depth = 1
DEBUG = False  # Set to True for debug mode, False for normal mode
perform_owl = 1

SHOW_OBJECT_DETECTION = False
SHOW_DEPTH_FRAME_FULL = False
SHOW_ANGLE_FRAME = True
SHOW_DEPTH_FRAME_BOXED = True



# Function to calculate length of a contour


def calculate_contour_length(contour):
    return cv2.arcLength(contour, True)

# Function to check if a contour is outside the padding area and the box


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
    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on proximity to the edge of the frame and the rectangular box
    valid_contours = [cnt for cnt in contours if is_contour_inside_box(cnt, frame.shape, box)]

    # Calculate length for each contour
    lengths = [calculate_contour_length(contour) for contour in valid_contours]

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
        cv2.line(frame, tuple(longest_side[0]), tuple(longest_side[1]), (0, 255, 255), thickness=5)  # Yellow thick line

        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)  # Green box

        cv2.drawContours(frame, valid_contours, -1, (0, 0, 255), thickness=2)

        # Calculate the angle of the yellow line
        angle = calculate_angle(longest_side[0], longest_side[1])

        if SHOW_ANGLE_FRAME:
            cv2.imshow('Angle Frame', frame)

        if DEBUG:
            print("Angle of the yellow line:", angle)
        return angle
    else:
        if DEBUG:
            print("No contours found.")
        return -1


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
        print("MPS is available. Using MPS as the default device.")
    else:
        # MPS is not available, so use the CPU
        torch.set_default_device("cpu")
        if DEBUG:
            print("MPS is not available. Using CPU as the default device.")
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
    device='mps'
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


def process_frame_with_fastsam(frame, largest_box, device='mps'):
    # Save the frame as an image temporarily
    temp_image_path = 'temp_frame.jpg'
    cv2.imwrite(temp_image_path, frame)

    input = Image.open(temp_image_path)
    input = input.convert("RGB")

    # Run FastSAM on the captured frame

    everything_results = fast_same_model(input, device=device, retina_masks=True, conf=0.1, iou=0.9)
    prompt_process = FastSAMPrompt(input, everything_results, device=device)

    #ann = prompt_process.everything_prompt()
    ann = prompt_process.box_prompt(bboxes=[largest_box])
    #ann = prompt_process.box_prompt(bboxes=[[1498.5, 2.1868, 1920.4, 132.91]])

    # Display the segmentation result
    output_image_path = 'output_frame_tmp.jpg'

    prompt_process.plot(annotations=ann, output_path=output_image_path,)

    # Read the output image and display it
    segmented_frame = cv2.imread(output_image_path)
    return segmented_frame


def read_image_and_display(frame, texts):

    P1 = get_projection_matrix(left_camera_id, base_path)
    P2 = get_projection_matrix(right_camera_id, base_path)

    height, width, _ = frame.shape

    # Divide the image into left and right halves (example division)
    left_image = frame[:, :width // 2]
    right_image = frame[:, width // 2:]

    conf=0.35
    box_size_diff_threshold=0.8
    box_center_diff_threshold=0.1
    left_box = owl2_max_conf(left_image.copy(), texts, conf)
    right_box = owl2_max_conf(right_image.copy(), texts, conf)



    left_image = process_frame_with_fastsam(left_image, left_box, device='mps')
    #cv2.imshow('Segmented Frame', left_image)

    right_image = process_frame_with_fastsam(right_image, right_box, device='mps')
    #cv2.imshow('Segmented Frame', right_image)

    rejoined_image = np.concatenate((left_image, right_image), axis=1)
    cv2.imshow('Rejoined Image', rejoined_image)


    #depth_frame = depth_anything(frame)

    if left_box is not None and right_box is not None:

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

        left_point_inside_right = (right_box[0] <= left_point[0] <= right_box[2] and
                                   right_box[1] <= left_point[1] <= right_box[3])

        # Check if right_point is inside left_box
        right_point_inside_left = (left_box[0] <= right_point[0] <= left_box[2] and
                                   left_box[1] <= right_point[1] <= left_box[3])


        # Compare with the box_size_diff_threshold
        if (normalized_difference < box_size_diff_threshold) : #and left_point_inside_right and right_point_inside_left:
            # Calculate the 3D points using DLT
            point3D = DLT(P1, P2, left_point, right_point)

            if DEBUG:
                print('p1,p2', P1, P2)
                print('pointL1,pointL2', left_point, right_point)

            print(f"point3D: ({point3D[0]:.2f}, {point3D[1]:.2f}, {point3D[2]:.2f})  Left: ({left_point[0]:.2f}, {left_point[1]:.2f})  Right: ({right_point[0]:.2f}, {right_point[1]:.2f})")

            pointRef = np.array([0, 0, 0])

            # Calculate the Euclidean distance between the two points
            spatial_distance = np.linalg.norm(point3D - pointRef)
            print(f"Spatial Distance: {spatial_distance:.2f} units")
        else:
            print("Cannot detect depth, because objects seem to be different sizes or positions")


    else:
        print("Cannot detect depth, because object is not recognized by both cameras")

        #angle = mask_angle(frame, box)
    if DEBUG:
        print("angle ", angle)
        #print("average_depth ", average_depth)



def capture_webcam_and_display(texts, videofile=None):
    # Open the video capture device (webcam 0)
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

        read_image_and_display(frame, texts)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_model_owl()

    base_path = '../stereo/permanent_calibration_ipad/'
    calibration_file = base_path+'calibration_settings.yaml'
    parse_calibration_settings_file(calibration_file)
    #img_to_calib = load_image('/Users/ms/code/random/saved/ref.jpg')
    img_to_calib = load_image('/Users/ms/code/random/saved/ref.jpg')
    R_W0, T_W0, R_W1, T_W1 = get_and_save_camera_extrinsics(base_path, img_to_calib)
    if DEBUG:
        print(R_W0, T_W0, R_W1, T_W1 )


    # texts = [["a red block", "a green block"]]
    texts = [["a green block", "a green square"]]
    #texts = [["a green circle", "a green square"]]

    #texts = [["face", "a human face"]];capture_webcam_and_display(texts)
    #texts = [["a yellow wooden cube", "a green cube"]];capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid1.mov")
    texts = [["a green wooden block"]];capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid_green.mov")
    #capture_webcam_and_display(texts, 0)

    #img = cv2.imread("/Users/ms/code/random/saved/saved_image_ang_in_13.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)

    img = cv2.imread("/Users/ms/code/random/saved/saved_image_ang_in_14.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)

    exit()

    img = cv2.imread("/Users/ms/Downloads/b.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)

    img = cv2.imread("/Users/ms/Downloads/g.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)

    texts = [["books", "a book"]]
    img = cv2.imread("/Users/ms/Downloads/b1.jpeg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)
    img = cv2.imread("/Users/ms/Downloads/b2.jpeg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)
    img = cv2.imread("/Users/ms/Downloads/b3.jpeg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)
    img = cv2.imread("/Users/ms/Downloads/a1.jpg")
    read_image_and_display(img, texts)

    img = cv2.imread("/Users/ms/Downloads/b1.jpg")
    read_image_and_display(img, texts)
