from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

cap = cv2.VideoCapture(0)  # Using webcam device ID 0

current_frame = None
lock = threading.Lock()

def read_frames():
    global current_frame
    while True:
        success, frame = cap.read()
        if not success:
            continue
        with lock:
            current_frame = frame.copy()  # Make a copy to avoid conflicts
        cv2.waitKey(1)

def get_frame():
    with lock:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            return frame
        else:
            return None

def generate_video_feed():
    while True:
        frame = get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n\r\n')

@app.route('/frame_feed')
def frame_feed():
    frame = get_frame()
    if frame:
        return Response(frame, mimetype='image/jpeg')
    else:
        return Response(status=204)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    thread = threading.Thread(target=read_frames)
    thread.daemon = True
    thread.start()
    app.run(host='192.168.1.129', port=5002)
