#!/usr/bin/env python3
"""
OpenCV ile kamera, MediaPipe Tasks (HandLandmarker) ile el tespiti.
El varsa yeşil "El tespit edildi!", yoksa kirmizi "El yok" yazar.
Q ile kapat.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model dosyası (yoksa indirilecek)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")


def ensure_model():
    """Model yoksa indir."""
    if not os.path.isfile(MODEL_PATH):
        print("Model indiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model indirildi.")


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # HandLandmarker (mediapipe.tasks)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            # VIDEO modu için timestamp (ms)
            frame_timestamp_ms = int(frame_count * 1000 / 30)
            frame_count += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                text = "El tespit edildi!"
                color = (0, 255, 0)  # Yeşil (BGR)
            else:
                text = "El yok"
                color = (0, 0, 255)  # Kırmızı (BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x, y = 20, 50
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (x + 5, y), font, font_scale, color, thickness)

            cv2.imshow("El Tespiti", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
