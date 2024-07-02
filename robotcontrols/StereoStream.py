import cv2
from flask import Flask, Response

app = Flask(__name__)


def get_available_webcam_ids():
    available_ids = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_ids.append(index)
        cap.release()
        index += 1
    return available_ids


def generate_frames(video_id, width=2560, height=720):
    cap = cv2.VideoCapture(video_id)  # Use 0 for webcam or provide a video file path

    # Check if the video capture is successful
    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_id}")
        return None  # Return None if capture is not successful

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize the frame to the desired resolution (if needed)
            frame = cv2.resize(frame, (width, height))

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            # Convert the frame to bytes
            frame = buffer.tobytes()

            # Yield the frame in a multipart MIME response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    available_webcam_ids = get_available_webcam_ids()
    print(f"Available Webcam IDs: {available_webcam_ids}")

    video_id = 0  # Default to webcam ID 0, change as needed

    if video_id not in available_webcam_ids:
        return "Error: Selected webcam ID not available."

    frame_generator = generate_frames(video_id)
    if frame_generator is None:
        return "Error: Cannot start video stream."

    return Response(frame_generator, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Check if the selected video source can be opened before starting the app
    available_webcam_ids = get_available_webcam_ids()
    video_id = 0  # Default to webcam ID 0, change as needed
    if video_id in available_webcam_ids:
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print(f"Error: Selected webcam ID {video_id} not available. Cannot start application.")
