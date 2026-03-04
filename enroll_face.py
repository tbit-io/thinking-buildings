#!/usr/bin/env python3
"""Enroll faces from webcam or image files."""

import sys

import cv2

from thinking_buildings.config import load_config
from thinking_buildings.face_recognizer import FaceRecognizer


def enroll_webcam(name: str, face_rec: FaceRecognizer):
    """Capture frames from webcam and enroll. ESPACIO=capturar, X=salir."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam")
        return

    enrolled = 0
    print(f"Enrolling '{name}' — ESPACIO=capturar, X=salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"{name} [{enrolled}] ESPACIO=capturar X=salir",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll Face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            break
        if key == ord(" "):
            if face_rec.enroll_from_frame(name, frame):
                enrolled += 1
                print(f"  [{enrolled}] cara registrada")
            else:
                print("  [!] no se detecto cara, intenta de nuevo")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal: {enrolled} cara(s) registradas para '{name}'")


def enroll_images(name: str, face_rec: FaceRecognizer, paths: list[str]):
    """Enroll from image files."""
    enrolled = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [!] no se pudo leer {path}")
            continue
        if face_rec.enroll_from_frame(name, img):
            enrolled += 1
            print(f"  [{enrolled}] registrada desde {path}")
        else:
            print(f"  [!] no se detecto cara en {path}")

    print(f"\nTotal: {enrolled} caras registradas para '{name}'")


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python enroll_face.py javier              # webcam")
        print("  python enroll_face.py javier foto1.jpg foto2.jpg  # imagenes")
        sys.exit(1)

    name = sys.argv[1]
    images = sys.argv[2:]

    cfg = load_config()
    face_rec = FaceRecognizer(cfg.face_recognition)
    face_rec.enroll_from_directory(cfg.face_recognition.faces_dir)

    if images:
        enroll_images(name, face_rec, images)
    else:
        enroll_webcam(name, face_rec)

    # Show all enrolled faces
    persons = face_rec.db.list_persons()
    print("\nCaras registradas:")
    for person_name, count in persons:
        print(f"  {person_name}: {count} embeddings")

    face_rec.close()


if __name__ == "__main__":
    main()
