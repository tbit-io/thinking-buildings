#!/usr/bin/env python3
"""CLI for face enrollment management."""

import argparse
import sys

import cv2

from thinking_buildings.config import load_config
from thinking_buildings.logger_setup import setup_logging
from thinking_buildings.face_recognizer import FaceRecognizer


def capture_and_enroll(recognizer: FaceRecognizer, name: str, camera_source: int) -> None:
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_source}")
        sys.exit(1)

    print(f"Camera open. Press SPACE to capture a frame for '{name}', 'q' to cancel.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Enroll — press SPACE to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if recognizer.enroll_from_frame(name, frame):
                    print(f"Successfully enrolled '{name}'")
                else:
                    print("No face detected in frame. Try again.")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Face enrollment management")
    parser.add_argument("--name", type=str, help="Enroll a person by capturing from camera")
    parser.add_argument("--from-dir", type=str, help="Bulk enroll from a directory of face images")
    parser.add_argument("--list", action="store_true", help="List all enrolled persons")
    parser.add_argument("--remove", type=str, help="Remove an enrolled person")
    args = parser.parse_args()

    if not any([args.name, args.from_dir, args.list, args.remove]):
        parser.print_help()
        sys.exit(1)

    cfg = load_config()
    setup_logging(cfg.logging)
    recognizer = FaceRecognizer(cfg.face_recognition)

    try:
        if args.list:
            persons = recognizer.db.list_persons()
            if not persons:
                print("No enrolled persons.")
            else:
                print(f"{'Name':<20} {'Embeddings':>10}")
                print("-" * 32)
                for name, count in persons:
                    print(f"{name:<20} {count:>10}")

        elif args.remove:
            if recognizer.db.remove_person(args.remove):
                print(f"Removed '{args.remove}'")
            else:
                print(f"Person '{args.remove}' not found")

        elif args.from_dir:
            count = recognizer.enroll_from_directory(args.from_dir)
            print(f"Enrolled {count} face(s) from '{args.from_dir}'")

        elif args.name:
            capture_and_enroll(recognizer, args.name, cfg.camera.source)
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()
