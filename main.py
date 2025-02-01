from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from dotenv import load_dotenv
from deepface import DeepFace
from datetime import datetime, timedelta
import numpy as np
import threading
import time
import os
import cv2
import pytz
import pandas as pd


app = Flask(__name__)

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["ipcameraAttendanceSystem"]

EMPLOYEE_FOLDER = "employee_faces/"
os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)

# Global variables
camera_thread_running = True
last_detection_times = {}
DETECTION_COOLDOWN = 30


# Time zone setup
pkt = pytz.timezone('Asia/Karachi')

# Utility function to get the current time in 12-hour format (AM/PM)
def get_current_time_pkt_12hr():
    return datetime.now(pytz.utc).astimezone(pkt).strftime('%I:%M:%S %p')  # 12-hour format with AM/PM

# Utility function to get the current date in pkt
def get_current_date_pkt():
     return datetime.now(pytz.utc).astimezone(pkt).strftime('%d-%m-%y')  # Correct date format (dd-mm-yy)

# Utility function to convert a given datetime to pkt
def convert_to_pkt(dt):
    return dt.astimezone(pkt)


def log_status(message):
    """Print timestamped status messages"""
    timestamp = get_current_time_pkt_12hr()
    print(f"[{timestamp}] {message}")

def is_already_checked_in(employee_id):
    """Check if employee has already checked in for the current date"""
    try:
        # Get current datetime in pkt, then convert to UTC for comparison
        current_date_pk = get_current_date_pkt()
        
        
        existing_record = db.attendance.find_one({
            "employee_id": employee_id,
            "date": current_date_pk,
        })
        if existing_record:
            if existing_record.get('inTime'):
                return True , existing_record
        else:
            return False , None
       
    except Exception as e:
        log_status(f"Error checking check-in status: {str(e)}")
        return False, None

def verify_face(current_face_img, stored_face_path):
    """Verify if two face images belong to the same person"""
    try:
        temp_path = "temp_current_face.jpg"
        cv2.imwrite(temp_path, current_face_img)
        
        result = DeepFace.verify(
            img1_path=temp_path,
            img2_path=stored_face_path,
            model_name="VGG-Face",
            enforce_detection=False
        )
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        log_status(f"Face verification result: {result['verified']} with distance {result['distance']}")
        return result['verified'], result['distance']
    except Exception as e:
        log_status(f"Error in face verification: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, 1.0

def process_attendance(frame, face_cascade):
    """Process a frame for attendance"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw text showing if faces were found
        face_count = len(faces)
        cv2.putText(frame, f"Faces detected: {face_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
                
            for filename in os.listdir(EMPLOYEE_FOLDER):
                if not filename.endswith('.jpg'):
                    continue
                    
                stored_face_path = os.path.join(EMPLOYEE_FOLDER, filename)
                is_match, distance = verify_face(face_img, stored_face_path)
                
                if is_match and distance < 0.4:
                    employee_name, employee_id = filename.split('.')[0].split('_')
                    
                    current_time = time.time()
                    if employee_id in last_detection_times:
                        if current_time - last_detection_times[employee_id] < DETECTION_COOLDOWN:
                            continue
                    
                    last_detection_times[employee_id] = current_time
                    
                    already_checked_in, existing_record = is_already_checked_in(employee_id)
                    current_date_pk = get_current_date_pkt()
                    current_time_pk = get_current_time_pkt_12hr()

                    if already_checked_in == False:
                        # Create new attendance record
                        attendance = {
                            "employee_id": employee_id,
                            "employee_name": employee_name,
                            "date": current_date_pk,  # Store full datetime object
                            "inTime": current_time_pk,
                            "outTime": None,
                            "status": "present",
                        }
                        
                        try:
                            db.attendance.insert_one(attendance)
                            log_status(f"✓ Check-in recorded for {employee_name}")
                        except Exception as e:
                            log_status(f"Database error during check-in: {str(e)}")
                        
                    elif existing_record and not existing_record.get('outTime'):
                        # Convert to datetime object
                        hour=current_time_pk.split(":")[0]
                        if int(hour) >= 7:
                            try:
                                db.attendance.update_one(
                                    {"_id": existing_record["_id"]},
                                    {"$set": {"outTime": current_time_pk}}
                                )
                                log_status(f"→ Check-out recorded for {employee_name}")
                            except Exception as e:
                                log_status(f"Database error during check-out: {str(e)}")
                    
                    # Visual feedback
                    status = "Checked In" if not already_checked_in else "Already Recorded"
                    cv2.putText(frame, f"{employee_name}: {status}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.9, (0, 255, 0), 2)
                    
        return frame
    except Exception as e:
        log_status(f"Error in process_attendance: {str(e)}")
        return frame

def continuous_camera_monitoring():
    """Background thread for continuous camera monitoring"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_status("Error: Could not open camera")
            return
            
        log_status("Camera opened successfully")
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            log_status("Error: Could not load face cascade classifier")
            return
            
        log_status("Face cascade classifier loaded successfully")
        
        while camera_thread_running:
            ret, frame = cap.read()
            if not ret:
                log_status("Error: Failed to grab frame")
                time.sleep(0.1)  # Small delay before retry
                continue
            
            processed_frame = process_attendance(frame, face_cascade)
            
            # Show frame with status overlay
            cv2.imshow('Attendance Monitoring', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        log_status("Camera monitoring stopped")
    except Exception as e:
        log_status(f"Critical error in camera monitoring: {str(e)}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/start_monitoring')
def start_monitoring():
    global camera_thread_running
    camera_thread_running = True
    thread = threading.Thread(target=continuous_camera_monitoring, daemon=True)
    thread.start()
    log_status("Monitoring started")
    return jsonify({"message": "Monitoring started"})

@app.route('/stop_monitoring')
def stop_monitoring():
    global camera_thread_running
    camera_thread_running = False
    log_status("Monitoring stopped")
    return jsonify({"message": "Monitoring stopped"})

@app.route('/export_attendance', methods=['GET'])
def export_attendance():
    
    employee_id = request.args.get('employee_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = {}
    if employee_id:
        query['employee_id'] = employee_id
    if start_date and end_date:
        query['date'] = {
            '$gte': datetime.strptime(start_date, '%Y-%m-%d'),
            '$lte': datetime.strptime(end_date, '%Y-%m-%d')
        }
    
    data = list(db.attendance.find(query, {'_id': 0}))
    df = pd.DataFrame(data)
    
    # if not df.empty:
    #     df['duration'] = df.apply(
    #         lambda x: str(x['outTime'] - x['inTime']) if x['outTime'] else "N/A", 
    #         axis=1
    #     )
    if start_date and end_date:
        filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        filename = f"attendance_report_{employee_id}.csv"
    df.to_csv(filename, index=False)
    return jsonify({"message": "CSV generated", "file": filename})


@app.route('/register', methods=['POST'])
def register_employee():
    data = request.json
    employee_name = data.get("name")
    employee_id = data.get("id")
    
    if not employee_name or not employee_id:
        return jsonify({"error": "Missing employee name or ID"}), 400

    # Start camera and capture face
    # cap = cv2.VideoCapture("http://192.168.100.42:4747")  # Use the mobile camera (DroidCam users set URL here)
    cap = cv2.VideoCapture(0)  # Use the mobile camera (DroidCam users set URL here)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow("Camera Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            filename = f"{EMPLOYEE_FOLDER}{employee_name}_{employee_id}.jpg"
            cv2.imwrite(filename, face)
            cap.release()
            cv2.destroyAllWindows()
            
            # Save to database
            db.employees.insert_one({
                "name": employee_name,
                "id": employee_id,
                "image_path": filename
            })
            
            return jsonify({"message": "Employee registered successfully", "image": filename})
        
        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"error": "No face detected"}), 400

if __name__ == "__main__":
    app.run(debug=True)