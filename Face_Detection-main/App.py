from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import time

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Photo formats

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load cascade classifiers once at startup
cascades = {
    "smile": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml"),
    "face": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    "face_alt": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
    "face_alt2": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
    "profile": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
    "eye": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml"),
}

# Verify all classifiers loaded properly
for name, classifier in cascades.items():
    if classifier.empty():
        print(f"Warning: {name} classifier failed to load")

# Global variables
webcam_active = False
smile_counter = 0

def allowed_file(filename):
    """Check if the uploaded file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html", uploaded_image=None, processed_image=None, face_count=None, smile_count=None)

@app.route("/detect", methods=["POST"])
def detect():
    """Handle image upload and face/smile detection."""
    if "file" not in request.files:
        flash("No file part in the request")
        return redirect(url_for("index"))
        
    file = request.files["file"]
    
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))
        
    if not allowed_file(file.filename):
        flash("File type not allowed. Please use jpg, jpeg, png, or gif")
        return redirect(url_for("index"))
    
    # Secure the filename to prevent path traversal attacks
    secure_fname = secure_filename(file.filename)
    timestamp = int(time.time())
    filename = f"{timestamp}_{secure_fname}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        file.save(filepath)  # Save the uploaded file
        processed_filepath, face_count, smile_count = detect_faces(filepath, filename)
        
        # Generate URLs for the uploaded and processed images
        uploaded_file_url = url_for("static", filename=f"uploads/{filename}")
        processed_file_url = (
            url_for("static", filename=f"processed/{filename}") if face_count > 0 else None
        )
        
        return render_template(
            "index.html",
            uploaded_image=uploaded_file_url,
            processed_image=processed_file_url,
            face_count=face_count,
            smile_count=smile_count,
        )
    except Exception as e:
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for("index"))

def detect_faces(image_path, filename):
    """Detect faces and smiles in an uploaded image."""
    image = cv2.imread(image_path)
    if image is None:
        flash("Error loading image")
        return None, 0, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast

    faces = cascades["face"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        faces = cascades["face_alt"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        faces = cascades["face_alt2"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    smile_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        roi_y = y + int(h * 0.4)
        roi_h = int(h * 0.6)
        roi_y = max(0, roi_y)
        roi_h = min(roi_h, image.shape[0] - roi_y)

        if roi_h <= 0 or w <= 0:
            continue

        roi_gray = gray[roi_y:roi_y + roi_h, x:x + w]
        smiles = cascades["smile"].detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=15,
            minSize=(int(w / 6), int(h / 10))
        )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                image,
                (x + sx, roi_y + sy),
                (x + sx + sw, roi_y + sy + sh),
                (255, 0, 0),
                2,
            )
            smile_count += 1

    processed_filepath = os.path.join(app.config["PROCESSED_FOLDER"], filename)
    cv2.imwrite(processed_filepath, image)

    return processed_filepath, len(faces), smile_count

@app.route("/start_webcam", methods=["POST"])
def start_webcam():
    """Activate the webcam and reset the smile counter."""
    global webcam_active, smile_counter
    webcam_active = True
    smile_counter = 0  # Reset the counter
    return "", 204

@app.route("/stop_webcam", methods=["POST"])
def stop_webcam():
    """Deactivate the webcam."""
    global webcam_active
    webcam_active = False
    return "", 204

@app.route("/smile_counter_updates")
def smile_counter_updates():
    """Provide real-time updates of the smile counter using SSE."""
    def generate():
        global smile_counter
        while webcam_active:
            yield f"data: {smile_counter}\n\n"
            time.sleep(1)  # Update every second
    return Response(generate(), mimetype="text/event-stream")

@app.route("/video_feed")
def video_feed():
    """Route for streaming webcam video with face and smile detection."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    """Generate webcam frames with face and smile detection."""
    global smile_counter
    camera = cv2.VideoCapture(0)
    smile_detected = False

    if not camera.isOpened():
        print("Error: Could not open webcam")
        return

    while webcam_active:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, face_count, smile_count = detect_faces_frame(frame)

            if smile_count > 0 and not smile_detected:
                timestamp = int(time.time())
                filename = f"{timestamp}_smile.jpg"
                filepath = os.path.join(app.config["PROCESSED_FOLDER"], filename)
                cv2.imwrite(filepath, frame)
                smile_detected = True
                smile_counter += 1  # Increment the smile counter
            elif smile_count == 0:
                smile_detected = False

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

def detect_faces_frame(frame):
    """Detect faces and smiles in a webcam frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascades["face"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    smile_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        roi_gray = gray[y:y + h, x:x + w]
        smiles = cascades["smile"].detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 0, 0), 2)
            smile_count += 1

    return frame, len(faces), smile_count

if __name__ == "__main__":
    app.run(debug=True)