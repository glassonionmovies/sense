import cv2
import time
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt
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

from FastSAM.fastsam import FastSAM, FastSAMPrompt
fast_same_model = FastSAM('FastSAM/weights/FastSAM-x.pt')

from stereo.stereo_utils import *
from stereo.stereo_utils import _convert_to_homogeneous


# export PYTORCH_ENABLE_MPS_FALLBACK=1
perform_depth = 1
DEBUG = False  # Set to True for debug mode, False for normal mode
perform_owl = 1


SHOW_OBJECT_DETECTION = False
SHOW_DEPTH_FRAME_FULL = False
SHOW_ANGLE_FRAME = False
SHOW_DEPTH_FRAME_BOXED = False

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
        if ret==False:
            exit()
        # Increment frame count
        if(frame_count  % speed == 0):
            read_image_and_display(frame, texts)

        frame_count += 1


        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # Release the video capture device and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


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


def owl2_max_conf(frame, texts, conf):
    device = 'mps'
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

    return max_conf_box


SHOW_ANGLE_FRAME=0

def is_contour_at_frame_edge(contour, frame_shape):
    # Check if the contour is at the edge of the frame
    x, y, w, h = cv2.boundingRect(contour)
    height, width = frame_shape[:2]
    return (x == 0 or y == 0 or x + w == width or y + h == height)

def mask_angle(frame, box, side):
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
        epsilon = 0.005 * cv2.arcLength(longest_contour, True)
        approx = cv2.approxPolyDP(longest_contour, epsilon, True)

        # Find the longest side of the polygon
        longest_side = find_longest_side(approx[:, 0])

        # Draw the longest side of the polygon in yellow
        angle=-1
        if(longest_side is not None):
            cv2.line(cropped_frame, tuple(longest_side[0]), tuple(longest_side[1]), (0, 255, 255), thickness=5)  # Yellow thick line
            angle = calculate_angle(longest_side[0], longest_side[1])

        #cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)  # Green box

        # Draw rotated rectangles around each valid contour
        # Draw rotated rectangle around the largest contour

        cv2.drawContours(cropped_frame, valid_contours, -1, (0, 0, 255), thickness=1)

        # Calculate the angle of the yellow line

        if SHOW_ANGLE_FRAME:
            cv2.imshow(side, cropped_frame)

        if DEBUG:
            print("Angle of the yellow line:", angle)
        return angle
    else:
        if DEBUG:
            print("No contours found.")
        return -1



def find_matching_point(left_box, right_box, left_image, right_image):


    box_size_diff_threshold=0.8

    mask_success_left, left_image_sammed, left_point = process_frame_with_fastsam(left_image, left_box.copy(), device='mps')
    mask_success_right, right_image_sammed, right_point = process_frame_with_fastsam(right_image, right_box.copy(), device='mps')
    new_height = min(left_image_sammed.shape[0], right_image_sammed.shape[0])
    # Resize both images to the new height while maintaining aspect ratio
    left_image_resized = cv2.resize(left_image_sammed, (
    int(left_image_sammed.shape[1] * new_height / left_image_sammed.shape[0]), new_height))
    right_image_resized = cv2.resize(right_image_sammed, (
    int(right_image_sammed.shape[1] * new_height / right_image_sammed.shape[0]), new_height))

    # Concatenate images horizontally
    combined_image = np.concatenate((left_image_resized, right_image_resized), axis=1)

    # Display the concatenated image using OpenCV
    cv2.imshow('combined_masked_image', combined_image)

    if (mask_success_left is False or mask_success_right is False):
        print("USING CENTER POINT")
        left_point = [int((left_box[0] + left_box[2]) / 2), int((left_box[1] + left_box[3]) / 2)]
        right_point = [int((right_box[0] + right_box[2]) / 2), int((right_box[1] + right_box[3]) / 2)]


    cv2.circle(left_image, left_point, 10, (255, 0, 0), -1)
    cv2.circle(right_image, right_point, 10, (255, 0, 0), -1)

    left_box = [int(round(coord)) for coord in left_box]
    right_box = [int(round(coord)) for coord in right_box]


    cv2.rectangle(left_image, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 255, 0), 2)
    cv2.rectangle(right_image, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 255, 0), 2)

    rejoined_image = np.concatenate((left_image, right_image), axis=1)
    cv2.imshow('rejoined_image', rejoined_image)

    left_box_size = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
    right_box_size = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])


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

        return True, left_point, right_point
    else:
        return False, left_point, right_point




import numpy as np







def read_image_and_display(frame, texts):
    height, width, _ = frame.shape
    left_image = frame[:, :width // 2]
    right_image = frame[:, width // 2:]

    conf=0.35
    box_size_diff_threshold=0.8

    #left_image, right_image=rectify_images(left_image, right_image, base_path, '')

    left_box = owl2_max_conf(left_image, texts, conf)
    right_box = owl2_max_conf(right_image, texts, conf)
#    left_box = owl2_max_conf(left_image.copy(), texts, conf)
#    right_box = owl2_max_conf(right_image.copy(), texts, conf)

    if left_box is not None and right_box is not None:
        #mask_angle(left_image, left_box,'left')
        #mask_angle(right_image, right_box,'right')

        match, left_point, right_point=find_matching_point(left_box, right_box, left_image, right_image)

        if match:
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
            print("Cannot detect depth, because object sizes differs by more than threshold")

    else:
        print("Cannot detect depth, because object is not recognized by both cameras")

        #angle = mask_angle(frame, box)
    if DEBUG:
        print("angle ", angle)
        #print("average_depth ", average_depth)





def extract_mask_vertices(segmentation_mask):
    # Get unique object identifiers, excluding the background (usually 0)
    unique_objects = np.unique(segmentation_mask)
    unique_objects = unique_objects[unique_objects != 0]

    object_vertices = {}

    for obj_id in unique_objects:
        # Create a binary mask for the current object
        binary_mask = (segmentation_mask == obj_id)

        # Extract row and column coordinates where the binary mask is True
        rows, cols = np.where(binary_mask)

        # Store the coordinates in a dictionary
        object_vertices[obj_id] = (rows, cols)

    return object_vertices
def visualize_mask_with_vertices(segmentation_mask):
    vertices=extract_mask_vertices(segmentation_mask)
    plt.figure(figsize=(10, 10))
    plt.imshow(segmentation_mask, cmap='jet', interpolation='nearest')
    plt.colorbar()

    for obj_id, (rows, cols) in vertices.items():
        plt.scatter(cols, rows, edgecolor='black', facecolor='none', s=100, label=f'Object {obj_id}')

    plt.title("Segmentation Mask with Vertices")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.show()

def apply_binary_masks_to_frame(frame, binary_masks):
    masked_frame = frame.copy()

    # Define transparent purple color (BGR format)
    purple_color = np.array([255, 0, 255], dtype=np.uint8)

    # Iterate over each binary mask and apply it to the frame
    for binary_mask in binary_masks:
        # Ensure binary mask is of type np.uint8
        binary_mask = binary_mask.astype(np.uint8)

        # Create an image with transparent purple color
        purple_mask = np.zeros_like(masked_frame, dtype=np.uint8)
        purple_mask[:] = purple_color

        # Apply the binary mask to the purple mask
        masked_purple = cv2.bitwise_and(purple_mask, purple_mask, mask=binary_mask)

        # Blend the masked purple with the frame using bitwise_or
        masked_frame = cv2.addWeighted(masked_frame, 1, masked_purple, 0.5, 0)

    return masked_frame


##function below does the following
# 1. enlarges box (mask is better)
# 2. Create binary mask with ann[0] - Experiment what happens when ann has ann[1], ann[2] and so on - means multiple masks
# 3.
def process_frame_with_fastsam(full_frame, box, device='mps'):
    print("box at beginning of fast sam", box)
    x1, y1, x2, y2 = map(int, box)
    mask_success = False
    center_x = (box[2] + box[0]) / 2
    center_y = (box[3] + box[1]) / 2

    enlargement_factor_sam = 0.5

    bigger_box = [
        center_x - (center_x - box[0]) * (1 + enlargement_factor_sam),  # x_min
        center_y - (center_y - box[1]) * (1 + enlargement_factor_sam),  # y_min
        center_x + (box[2] - center_x) * (1 + enlargement_factor_sam),  # x_max
        center_y + (box[3] - center_y) * (1 + enlargement_factor_sam)  # y_max
    ]

    #box = bigger_box

    x_min, y_min, x_max, y_max = map(int, bigger_box)
    cropped_frame = full_frame[y_min:y_max, x_min:x_max]
    input = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

    # Run FastSAM on the captured frame
    everything_results = fast_same_model(input, device=device, retina_masks=True, conf=0.2, iou=0.9)
    prompt_process = FastSAMPrompt(input, everything_results, device=device)

    #ann = prompt_process.everything_prompt()
    ann = prompt_process.box_prompt(bboxes=[box])
    if (len(ann)) >= 1:
        print('MULTIPLE MASKS ', len(ann))
    if(len(ann))==0:
        return mask_success, cropped_frame, (-1, -1)

    binary_masks = create_binary_masks(ann[0])

    segmented_frame = apply_binary_masks_to_frame(cropped_frame, binary_masks)

    # Find topmost and rightmost point in the segmented frame
    topmost_rightmost_point = None
    for i, binary_mask in enumerate(binary_masks):
        # Find the non-zero points in the binary mask
        points = np.argwhere(binary_mask > 0)
        if len(points) > 0:
            # Find the topmost points
            topmost_points = points[points[:, 0] == np.min(points[:, 0])]
            # Find the rightmost point among the topmost points
            rightmost_point = topmost_points[np.argmax(topmost_points[:, 1])]
            # Translate point to coordinates in the original full frame
            rightmost_point = (rightmost_point[1] + x_min, rightmost_point[0] + y_min)

            # Update the topmost_rightmost_point if not set or if this one is further right or higher up
            if (topmost_rightmost_point is None or
                    rightmost_point[1] > topmost_rightmost_point[1] or
                    (rightmost_point[1] == topmost_rightmost_point[1] and rightmost_point[0] < topmost_rightmost_point[
                        0])):
                topmost_rightmost_point = rightmost_point


    if (topmost_rightmost_point is not None):

        (x_full,y_full) = topmost_rightmost_point
        ##################
        x=x_full-x_min
        y=y_full-y_min

        if(x1 <= x_full <= x2 and y1 <= y_full <= y2):
            mask_success=True
            cv2.circle(segmented_frame, (x, y), 10, (255, 0, 0), -1)
        else:
            return False, segmented_frame, (-1, -1)
    else:
        return False, segmented_frame, (-1, -1)


        #print("drawing circle on segmented frame at ", (x, y))

    #visualize_binary_masks(binary_masks)
    #print("before returning (x_full, y_full)", (x_full, y_full))
    return mask_success, segmented_frame, (x_full, y_full)


def create_binary_masks(segmentation_mask):
    # Get unique object identifiers, excluding the background (usually 0)
    unique_objects = np.unique(segmentation_mask)
    unique_objects = unique_objects[unique_objects != 0]

    binary_masks = []

    for obj_id in unique_objects:
        # Create a binary mask for the current object
        binary_mask = (segmentation_mask == obj_id).astype(np.uint8)
        binary_masks.append(binary_mask)

    return binary_masks


def visualize_binary_masks(binary_masks):

    num_masks = len(binary_masks)
    print(num_masks)
    plt.figure(figsize=(10, 10))

    for i, binary_mask in enumerate(binary_masks):
        plt.subplot(1, num_masks, i + 1)
        plt.imshow(binary_mask.squeeze(), cmap='gray')  # Ensure the mask is 2D
        plt.title(f'Object {i + 1}')
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    load_model_owl()


    world_calibrate('/Users/ms/code/random/saved/ref.jpg')

    texts = [["corner of a cube", "a green surface"]]

    wk = 5000

    # texts = [["a red block", "a green block"]]
    #texts = [["a green block", "a green square"]]

    #texts = [["face", "a human face"]];capture_webcam_and_display(texts)
    texts = [["a green wooden block", "a green cube"]];capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid_green_3s.mov")
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

