from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

video_file_path = '/Users/ms/Downloads/stevid_green.mov'
cap = cv2.VideoCapture(video_file_path)

current_frame = None
lock = threading.Lock()

def read_frames():
    global current_frame
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            continue
        with lock:
            current_frame = frame
        cv2.waitKey(1)

def get_frame():
    with lock:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            return None

@app.route('/frame_feed')
def frame_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    thread = threading.Thread(target=read_frames)
    thread.daemon = True
    thread.start()
    app.run(host='192.168.1.129', port=5002)
