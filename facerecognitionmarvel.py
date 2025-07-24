import cv2
import dlib
import numpy as np
import os

# Load face detector
detector = dlib.get_frontal_face_detector()

# Load shape predictor and face recognition model
shape_predictor_path = "D:/ASUS/Download/faceRecognition/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "D:/ASUS/Download/faceRecognition/dlib_face_recognition_resnet_model_v1.dat"

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor file not found at {shape_predictor_path}. Please download it.")
if not os.path.exists(face_rec_model_path):
    raise FileNotFoundError(f"Face recognition model not found at {face_rec_model_path}. Please download it.")

sp = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Fungsi untuk mendapatkan encoding wajah
def get_face_encoding(image, face_rect):
    if image is None or face_rect is None:
        return None
    shape = sp(image, face_rect)
    encoding = np.array(face_rec_model.compute_face_descriptor(image, shape))
    return encoding

# Fungsi untuk memuat wajah referensi dari folder
def load_reference_faces(folder="D:/ASUS/Download/faceRecognition/faceRecognitionMarvel/wajah/"):
    reference_encodings = {}
    
    if not os.path.exists(folder):
        print(f"Warning: Reference folder {folder} not found. Creating empty reference.")
        return reference_encodings

    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)

        if not os.path.isdir(person_path):
            continue

        encodings = []
        for filename in os.listdir(person_path):
            filepath = os.path.join(person_path, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"Warning: Could not load image {filepath}")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                encoding = get_face_encoding(image, faces[0])
                if encoding is not None:
                    encodings.append(encoding)

        if encodings:
            reference_encodings[person_name] = np.mean(encodings, axis=0)

    return reference_encodings

# Load wajah referensi dari folder
reference_faces = load_reference_faces()

# Buka video sebagai input
video_path = "D:/ASUS/Download/faceRecognition/video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Warning: Could not open video {video_path}. Switching to webcam.")
    cap = cv2.VideoCapture(0)  # Fallback ke webcam

threshold = 0.75  # Ambang batas pengenalan wajah

# Simpan video hasil deteksi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_path = "D:/ASUS/Download/faceRecognition/output.avi"
out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
if not out.isOpened():
    print(f"Warning: Could not open output video {out_path}. Output will be disabled.")
    out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_image = frame[y1:y2, x1:x2]

        if face_image.shape[0] == 0 or face_image.shape[1] == 0:
            continue

        encoding = get_face_encoding(frame, face)
        if encoding is None:
            continue

        label = "Unknown"
        min_distance = float("inf")

        # Bandingkan dengan wajah referensi
        for name, ref_encoding in reference_faces.items():
            distance = np.linalg.norm(ref_encoding - encoding)
            if distance < threshold and distance < min_distance:
                min_distance = distance
                label = name

        # Gambar kotak dan teks
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition Marvel", frame)
    if out is not None:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
