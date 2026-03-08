#!/usr/bin/env python3
"""
OpenCV ile kamera, MediaPipe Tasks (HandLandmarker) ile el tespiti.
El varsa yeşil "El tespit edildi!", yoksa kirmizi "El yok" yazar.
Q ile kapat.
"""

import os
import urllib.request
import time

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model dosyası (yoksa indirilecek)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

ALPHA = 0.2  # EMA katsayısı


def ensure_model():
    """Model yoksa indir."""
    if not os.path.isfile(MODEL_PATH):
        print("Model indiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model indirildi.")


def main():
    ensure_model()

    # Mouse bilgisi
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0)
    # FPS ve performans için çözünürlüğü düşür
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # HandLandmarker (mediapipe.tasks)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    frame_count = 0
    ema_x = None
    ema_y = None
    last_mouse_x = None
    last_mouse_y = None
    last_click_time = 0.0

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

            # Sadece işaret parmağı açıksa mouse'u hareket ettir
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                def finger_open(hand_landmarks, tip_id, pip_id):
                    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y

                index_open = finger_open(hand, 8, 6)
                middle_open = finger_open(hand, 12, 10)
                ring_open = finger_open(hand, 16, 14)
                pinky_open = finger_open(hand, 20, 18)

                if index_open and not (middle_open or ring_open or pinky_open):
                    tip = hand[8]
                    raw_x = tip.x
                    raw_y = tip.y

                    # EMA filtresi
                    if ema_x is None or ema_y is None:
                        ema_x, ema_y = raw_x, raw_y
                    else:
                        ema_x = ALPHA * raw_x + (1 - ALPHA) * ema_x
                        ema_y = ALPHA * raw_y + (1 - ALPHA) * ema_y

                    # 0.08-0.92 (X), 0.08-0.80 (Y) aralıklarını tüm ekrana ölçekle
                    def clamp_x(v, vmin=0.08, vmax=0.92):
                        return max(vmin, min(vmax, v))

                    def clamp_y(v, vmin=0.08, vmax=0.80):
                        return max(vmin, min(vmax, v))

                    norm_x = (clamp_x(ema_x) - 0.08) / 0.84
                    norm_y = (clamp_y(ema_y) - 0.08) / 0.72

                    # Normalized koordinatları ekran pikseline çevir (X ekseni aynalı)
                    mouse_x = int((1.0 - norm_x) * screen_w)
                    mouse_y = int(norm_y * screen_h)

                    # Küçük titremeleri yok say (3 pikselden azsa hareket etme)
                    if (
                        last_mouse_x is not None
                        and last_mouse_y is not None
                        and abs(mouse_x - last_mouse_x) < 3
                        and abs(mouse_y - last_mouse_y) < 3
                    ):
                        pass
                    else:
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0)
                        last_mouse_x, last_mouse_y = mouse_x, mouse_y

                # Sol tık: işaret ve orta parmak ikisi birden açıksa, 0.5 sn debounce ile
                now = time.monotonic()
                if index_open and middle_open and now - last_click_time >= 0.5:
                    pyautogui.click(button="left")
                    last_click_time = now

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
