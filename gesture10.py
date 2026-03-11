#!/usr/bin/env python3
"""
gesture10 - Tkinter arayuzlu el tespit & mouse kontrol programi.
Kamera goruntusu, durum bilgisi, Baslat/Durdur ve Cikis butonu icerir.
"""

import os
import urllib.request
import time
import threading
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

ALPHA = 0.2
PREVIEW_W = 640
PREVIEW_H = 400


def ensure_model():
    if not os.path.isfile(MODEL_PATH):
        print("Model indiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model indirildi.")


# ─────────────────────────────────────────
#  Renk & stil sabitleri
# ─────────────────────────────────────────
BG        = "#0d0d0f"
PANEL     = "#16161a"
ACCENT    = "#00e5a0"
ACCENT2   = "#00b87a"
DANGER    = "#ff4a6e"
TEXT      = "#e8e8f0"
SUBTEXT   = "#7a7a9a"
BORDER    = "#2a2a3a"


class GestureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("gesture10")
        self.configure(bg=BG)
        self.resizable(False, False)

        # Durum
        self.running = False
        self.cap = None
        self.landmarker = None
        self._thread = None
        self._stop_event = threading.Event()

        # Mouse
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.screen_w, self.screen_h = pyautogui.size()
        self.ema_x = self.ema_y = None
        self.last_mouse_x = self.last_mouse_y = None
        self.last_click_time = 0.0
        self.frame_count = 0

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ──────────────────────────────────
    def _build_ui(self):
        # Başlık çubuğu
        header = tk.Frame(self, bg=BG, pady=14)
        header.pack(fill="x", padx=24)

        tk.Label(
            header, text="gesture", font=("Courier New", 22, "bold"),
            fg=ACCENT, bg=BG
        ).pack(side="left")
        tk.Label(
            header, text="10", font=("Courier New", 22),
            fg=SUBTEXT, bg=BG
        ).pack(side="left")
        tk.Label(
            header, text="el -> mouse", font=("Courier New", 11),
            fg=SUBTEXT, bg=BG
        ).pack(side="left", padx=(10, 0), pady=(6, 0))

        # ince ayirici cizgi
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=0)

        # Ana içerik
        body = tk.Frame(self, bg=BG)
        body.pack(padx=20, pady=16)

        # Kamera önizleme alanı
        cam_frame = tk.Frame(body, bg=BORDER, bd=0)
        cam_frame.pack()
        self.cam_canvas = tk.Canvas(
            cam_frame, width=PREVIEW_W, height=PREVIEW_H,
            bg="#080810", highlightthickness=0
        )
        self.cam_canvas.pack()
        self._draw_placeholder()

        # Durum bandi
        status_bar = tk.Frame(self, bg=PANEL, pady=10)
        status_bar.pack(fill="x", padx=0)

        tk.Label(status_bar, text="DURUM", font=("Courier New", 9),
                 fg=SUBTEXT, bg=PANEL).pack(side="left", padx=(20, 8))

        self.dot = tk.Label(status_bar, text="●", font=("Courier New", 14),
                            fg=SUBTEXT, bg=PANEL)
        self.dot.pack(side="left")

        self.status_label = tk.Label(
            status_bar, text="Bekleniyor",
            font=("Courier New", 13, "bold"),
            fg=SUBTEXT, bg=PANEL
        )
        self.status_label.pack(side="left", padx=(6, 0))

        # Alt buton alani
        footer = tk.Frame(self, bg=BG, pady=16)
        footer.pack()

        btn_row = tk.Frame(footer, bg=BG)
        btn_row.pack()

        self.btn = tk.Button(
            btn_row,
            text="▶  BASLAT",
            font=("Courier New", 13, "bold"),
            fg=BG, bg=ACCENT,
            activeforeground=BG, activebackground=ACCENT2,
            relief="flat", bd=0, padx=30, pady=10, cursor="hand2",
            command=self._toggle
        )
        self.btn.pack(side="left", padx=(0, 10))

        tk.Button(
            btn_row,
            text="✕  CIKIS",
            font=("Courier New", 13, "bold"),
            fg=TEXT, bg="#2a2a3a",
            activeforeground=TEXT, activebackground="#3a3a4a",
            relief="flat", bd=0, padx=22, pady=10, cursor="hand2",
            command=self._on_close
        ).pack(side="left")

    def _draw_placeholder(self):
        self.cam_canvas.delete("all")
        cx, cy = PREVIEW_W // 2, PREVIEW_H // 2
        # izgara
        for x in range(0, PREVIEW_W, 40):
            self.cam_canvas.create_line(x, 0, x, PREVIEW_H, fill="#111118", width=1)
        for y in range(0, PREVIEW_H, 40):
            self.cam_canvas.create_line(0, y, PREVIEW_W, y, fill="#111118", width=1)
        self.cam_canvas.create_text(
            cx, cy, text="[ kamera kapali ]",
            font=("Courier New", 14), fill=BORDER
        )

    # ── Başlat / Durdur ─────────────────────
    def _toggle(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        ensure_model()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self._set_status("Kamera acilamadi!", DANGER)
            return

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.running = True
        self.ema_x = self.ema_y = None
        self.last_mouse_x = self.last_mouse_y = None
        self.last_click_time = 0.0
        self.frame_count = 0
        self._stop_event.clear()

        self.btn.config(text="■  DURDUR", bg=DANGER, activebackground="#cc3a58")
        self._set_status("Calisıyor - el bekleniyor", ACCENT)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self._stop_event.set()
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
        self.btn.config(text="▶  BASLAT", bg=ACCENT, activebackground=ACCENT2)
        self._set_status("Durduruldu", SUBTEXT)
        self._draw_placeholder()

    # ── Kamera döngüsü (ayrı thread) ────────
    def _loop(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            ts = int(self.frame_count * 1000 / 30)
            self.frame_count += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect_for_video(mp_image, ts)

            status_text = "El yok"
            status_color = DANGER

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                def finger_open(lm, tip, pip):
                    return lm[tip].y < lm[pip].y

                idx  = finger_open(hand, 8, 6)
                mid  = finger_open(hand, 12, 10)
                ring = finger_open(hand, 16, 14)
                pink = finger_open(hand, 20, 18)

                only_index       = idx and not (mid or ring or pink)
                index_and_middle = idx and mid and not (ring or pink)

                if only_index:
                    status_text  = "Mouse modu aktif"
                    status_color = ACCENT

                    tip   = hand[8]
                    raw_x, raw_y = tip.x, tip.y

                    if self.ema_x is None:
                        self.ema_x, self.ema_y = raw_x, raw_y
                    else:
                        self.ema_x = ALPHA * raw_x + (1 - ALPHA) * self.ema_x
                        self.ema_y = ALPHA * raw_y + (1 - ALPHA) * self.ema_y

                    def clamp(v, lo, hi): return max(lo, min(hi, v))
                    norm_x = (clamp(self.ema_x, 0.08, 0.92) - 0.08) / 0.84
                    norm_y = (clamp(self.ema_y, 0.08, 0.80) - 0.08) / 0.72

                    mx = int((1.0 - norm_x) * self.screen_w)
                    my = int(norm_y * self.screen_h)

                    if not (self.last_mouse_x is not None
                            and abs(mx - self.last_mouse_x) < 3
                            and abs(my - self.last_mouse_y) < 3):
                        pyautogui.moveTo(mx, my, duration=0)
                        self.last_mouse_x, self.last_mouse_y = mx, my

                elif index_and_middle:
                    status_text  = "Sol tik!"
                    status_color = "#ffd166"
                    now = time.monotonic()
                    if now - self.last_click_time >= 0.5:
                        pyautogui.click(button="left")
                        self.last_click_time = now
                else:
                    status_text  = "El tespit edildi"
                    status_color = ACCENT

            # Kamera görüntüsüne overlay
            overlay_color_bgr = (
                (0, 229, 160) if status_color == ACCENT else
                (102, 209, 255) if status_color == "#ffd166" else
                (110, 74, 255)
            )
            h_fr, w_fr = frame.shape[:2]
            cv2.rectangle(frame, (0, h_fr - 36), (w_fr, h_fr), (13, 13, 15), -1)
            cv2.putText(frame, status_text, (12, h_fr - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, overlay_color_bgr, 2)

            # Tkinter'a gönder
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((PREVIEW_W, PREVIEW_H))
            img_tk  = ImageTk.PhotoImage(img_pil)

            self.after(0, self._update_frame, img_tk, status_text, status_color)

    def _update_frame(self, img_tk, status_text, status_color):
        # PhotoImage referansını tut (GC'den korunmak için)
        self.cam_canvas._img = img_tk
        self.cam_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self._set_status(status_text, status_color)

    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)
        self.dot.config(fg=color)

    # ── Kapat ───────────────────────────────
    def _on_close(self):
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = GestureApp()
    app.mainloop()
