import cv2
from flask import Flask, Response

app = Flask(__name__)

def generate_frames(video_id, width=2560, height=720):
    cap = cv2.VideoCapture(video_id)  # Use 0 for webcam or provide a video file path

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
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    video_id = 0
        #'/Users/ms/Downloads/stevid.mov'
    return Response(generate_frames(video_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Added debug=True for development mode
