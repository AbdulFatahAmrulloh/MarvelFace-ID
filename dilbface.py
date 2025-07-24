import cv2
import dlib
import os
import sys

def main():
    # Konfigurasi path
    MODEL_NAME = "shape_predictor_68_face_landmarks.dat"
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)

    # Periksa keberadaan file model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model '{MODEL_NAME}' tidak ditemukan!")
        print("Pastikan Anda telah:")
        print(f"1. Mendownload file dari: http://dlib.net/files/{MODEL_NAME}.bz2")
        print("2. Mengekstrak file .bz2 menggunakan WinRAR/7-Zip")
        print(f"3. Menempatkan file '{MODEL_NAME}' di folder: {CURRENT_DIR}")
        sys.exit(1)

    # Inisialisasi detector
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {str(e)}")
        sys.exit(1)

    # Fungsi untuk menggambar landmark
    def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), radius, color, -1)

    # Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat mengakses kamera")
        sys.exit(1)

    print("Program deteksi wajah berjalan...")
    print("Petunjuk:")
    print("- Tekan 'q' untuk keluar")
    print("- Pastikan pencahayaan cukup")
    print("- Posisikan wajah menghadap kamera")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Gagal menangkap frame")
            continue

        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = detector(gray)
        
        for face in faces:
            try:
                # Deteksi landmark
                landmarks = predictor(gray, face)
                
                # Gambar landmark (titik hijau)
                draw_landmarks(frame, landmarks)
                
                # Gambar bounding box (opsional)
                # x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                
            except Exception as e:
                print(f"[WARNING] Gagal memproses wajah: {str(e)}")
                continue
        
        # Tampilkan hasil
        cv2.imshow('Face Landmark Detection', frame)
        
        # Keluar dengan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai")

if __name__ == "__main__":
    main()