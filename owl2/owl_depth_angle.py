import cv2
import time
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
global processor_owl
global model_owl
global perform_owl
global processor_depth
global model_depth
global perform_depth
global box_to_mask

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

def owl2(frame, texts):
    inputs = processor_owl(text=texts, images=frame, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device
    outputs = model_owl(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([frame.shape[:2]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.5)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    largest_box = None
    max_area = 0

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if DEBUG:
            print(f" {text[label]}  confidence {round(score.item(), 3)} at  {box}")

        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > max_area:
            max_area = area
            largest_box = box

        # Draw bounding box on the frame
        box = [int(b) for b in box]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{text[label]}: {round(score.item(), 3)}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if SHOW_OBJECT_DETECTION:
            cv2.imshow('Object Detection', frame)

    return largest_box

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

def read_image_and_display(frame, texts):
    box = owl2(frame, texts)
    depth_frame = depth_anything(frame)

    if (box is not None):
        depth_image_box =  depth_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        min_depth = 0  # Example minimum depth value
        max_depth = 1500000  # Example maximum depth value
        average_depth = find_average_depth(depth_image_box, min_depth, max_depth)
        print(f"Average Depth of the Object: {average_depth}")
        angle = mask_angle(frame, box)
    if DEBUG:
        print("angle ", angle)
        print("average_depth ", average_depth)



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
    device = load_model_owl()
    load_model_depth()
    # texts = [["a red block", "a green block"]]
    texts = [["a green block", "a wooden block"]]

    #texts = [["face", "a human face"]];capture_webcam_and_display(texts)
    #texts = [["a yellow wooden cube", "a green cube"]];capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid1.mov")
    texts = [["a green wooden block"]];capture_webcam_and_display(texts, "/Users/ms/Downloads/stevid_green.mov")

    img = cv2.imread("/Users/ms/Downloads/w.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)

    img = cv2.imread("/Users/ms/Downloads/a.jpg")
    read_image_and_display(img, texts)
    cv2.waitKey(9000)

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
