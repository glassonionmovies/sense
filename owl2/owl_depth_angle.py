import cv2
import time
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image
import requests
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"



global processor_owl
global model_owl
global perform_owl
perform_owl = 1

global processor_depth
global model_depth
global perform_depth
perform_depth = 1

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
        print("MPS is available. Using MPS as the default device.")
    else:
        # MPS is not available, so use the CPU
        torch.set_default_device("cpu")
        print("MPS is not available. Using CPU as the default device.")

    model_owl.to(
        torch.device("mps" if torch.backends.mps.is_available() else "cpu"))  # Move the model to the default device
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def mask_angle(frame, box_to_mask):
    x, y, x1, y1 = [int(coord) for coord in box_to_mask]

    # Print the coordinates to debug
    print(f"Box coordinates: x={x}, y={y}, x1={x1}, y1={y1}")

    # Check if the box coordinates are valid
    if x >= 0 and y >= 0 and x1 > x and y1 > y:
        mask_frame = frame[y:y1, x:x1]  # Extract the region within the box

        # Print the shape of the extracted region
        print(f"Mask frame shape: {mask_frame.shape}")

        mask_frame_pil = Image.fromarray(mask_frame)  # Convert to PIL Image

        # Find the angle of the mask
        mask_cv = np.array(mask_frame_pil)

        # Find the Contours
        contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the Convex Hull of the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            # Find the Minimum Area Rectangle
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Use np.intp instead of np.int0

            # Calculate the Angle
            angle = rect[-1]

            # Adjust angle to be within the range [-90, 90]
            if angle < -45:
                angle += 90

            print(f"The angle of the mask is: {angle:.2f} degrees")

            # Optionally, draw the box on the mask image for visualization
            mask_with_box = cv2.cvtColor(mask_cv, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(mask_with_box, [box], 0, (0, 255, 0), 2)
            cv2.imshow(f"./mask_with_box.png", mask_with_box)

            return mask_with_box  # Return the modified frame

        else:
            print("No contours found in the mask.")
            return frame  # Return the original frame if no contours are found

    else:
        print("Invalid box coordinates.")
        return frame  # Return the original frame




def owl2(frame, texts):

    inputs = processor_owl(text=texts, images=frame, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device
    outputs = model_owl(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([frame.shape[:2]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    box_to_mask = [0, 0, 0, 0]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f" {text[label]}  confidence {round(score.item(), 3)} at  {box}")

        # Draw bounding box on the frame
        box = [int(b) for b in box]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{text[label]}: {round(score.item(), 3)}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        box_to_mask = box

    return box_to_mask

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
    return np.array(depth)

def capture_webcam_and_display(texts):

    # Open the video capture device (webcam 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
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


        box_to_mask = [0.12, 445.85, 180.11, 883.16]

        # Perform object detection

        if(perform_owl == 1):
            box_to_mask=owl2(frame, texts)
            print(box_to_mask)
\
        if(perform_depth == 1):
            frame = depth_anything(frame)

        box=box_to_mask
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)


        frame=mask_angle(frame,box_to_mask)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = load_model_owl()
    load_model_depth()
    texts = [["a person face", "eyes"]]

    capture_webcam_and_display(texts)
