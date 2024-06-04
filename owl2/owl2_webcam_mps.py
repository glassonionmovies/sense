import cv2
import time
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

global processor
global model
global perform_owl
perform_owl = 0


def load_model():
    global processor
    global model
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Check if MPS is available
    if torch.backends.mps.is_available():
        # Set the default device to MPS
        torch.set_default_device("mps")
        print("MPS is available. Using MPS as the default device.")
    else:
        # MPS is not available, so use the CPU
        torch.set_default_device("cpu")
        print("MPS is not available. Using CPU as the default device.")

    model.to(
        torch.device("mps" if torch.backends.mps.is_available() else "cpu"))  # Move the model to the default device
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def owl2(frame, texts):
    inputs = processor(text=texts, images=frame, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([frame.shape[:2]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
    return results


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

            results=owl2(frame, texts)
            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                print(f" {text[label]}  confidence {round(score.item(), 3)} at  {box}")

                # Draw bounding box on the frame
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{text[label]}: {round(score.item(), 3)}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                box_to_mask=box



        print(box_to_mask)

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
    device = load_model()
    texts = [["a person face", "eyes"]]

    capture_webcam_and_display(texts)
