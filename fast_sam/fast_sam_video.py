import cv2
from fastsam import FastSAM, FastSAMPrompt

# Initialize the model
model = FastSAM('/home/ms/code/sense/fast_sam/../../FastSAM/weights/FastSAM-x.pt')
DEVICE = 'mps'

# Function to process and display each frame
def process_frame(frame):
    # Save the frame as an image temporarily

    height, width = frame.shape[:2]
    leftFrame = frame[:, :width // 2]
    rightFrame = frame[:, width // 2:]
    frame=leftFrame


    temp_image_path = 'temp_frame.jpg'
    cv2.imwrite(temp_image_path, frame)
    
    # Run the model on the captured frame
    everything_results = model(temp_image_path, device=DEVICE, retina_masks=True, conf=0.1, iou=0.9,)
    prompt_process = FastSAMPrompt(temp_image_path, everything_results, device=DEVICE)
    
    # Everything prompt
    #ann = prompt_process.everything_prompt()
    ann = prompt_process.text_prompt(text='small red cube in the foreground at the center')
    
    # Display the segmentation result
    output_image_path = 'output_frame.jpg'
    prompt_process.plot(annotations=ann, output_path=output_image_path)
    
    # Read the output image and display it
    segmented_frame = cv2.imread(output_image_path)
    return segmented_frame

# Capture video from the webcam
#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('/Users/mudit.e.srivastava/Downloads/d.mov')

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Process and get the segmented frame
    segmented_frame = process_frame(frame)
    
    # Display the segmented frame
    cv2.imshow('Segmented Frame', segmented_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
