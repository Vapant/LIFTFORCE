import cv2
import re
import logging
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO
from paddleocr import PaddleOCR
import warnings

# Настройки логов
logging.getLogger("ppocr").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==========================================
# КЛАСС OCR
# ==========================================
class LicensePlateOCR:
    def __init__(self):
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    def preprocess_image(self, img):
        if img is None or img.size == 0: return None
        try:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            return img
        except Exception:
            return None

    def validate_text(self, text):
        clean = re.sub(r'[^ABEKMHOPCTYX0123456789]', '', text.upper())
        if 8 <= len(clean) <= 9:
            return clean
        return None

    def recognize(self, plate_img):
        if plate_img is None or plate_img.size == 0: return None
        if plate_img.shape[0] < 10 or plate_img.shape[1] < 10: return None

        processed_img = self.preprocess_image(plate_img)
        if processed_img is None: return None

        try:
            result = self.ocr.ocr(processed_img)
            if not result: return None

            data = result[0]
            if isinstance(data, dict):
                rec_texts = data.get('rec_texts', [])
                rec_scores = data.get('rec_scores', [])
                rec_boxes = data.get('rec_boxes', [])

                if not rec_texts: return None

                blocks = []
                for i, text in enumerate(rec_texts):
                    score = rec_scores[i]
                    box = rec_boxes[i]
                    try:
                        if hasattr(box[0], '__len__'):
                            x_coord = box[0][0]
                        else:
                            x_coord = box[0]
                    except:
                        x_coord = 0

                    if score > 0.4:
                        blocks.append((text, x_coord))

                blocks.sort(key=lambda b: b[1])
                full_text_raw = "".join([b[0] for b in blocks])

                return self.validate_text(full_text_raw)
            return None
        except Exception:
            return None


# ==========================================
# КЛАСС НАБЛЮДАТЕЛЬ
# ==========================================
class PlateObserver:
    def __init__(self, confidence_threshold=2):
        self.history = {}
        self.confirmed = {}
        self.THRESHOLD = confidence_threshold

    def add_reading(self, track_id, text):
        if track_id in self.confirmed:
            return None

        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append(text)

        if len(self.history[track_id]) > 10:
            self.history[track_id].pop(0)

        return self.check_winner(track_id)

    def check_winner(self, track_id):
        readings = self.history[track_id]
        if not readings: return None

        counts = Counter(readings)
        most_common, count = counts.most_common(1)[0]

        if count >= self.THRESHOLD:
            self.confirmed[track_id] = most_common
            return most_common
        return None


# ==========================================
# MAIN
# ==========================================
def main():
    print("🚀 Запуск без сохранения видео...")

    VIDEO_SOURCE = 'test_video_7.mp4'
    YOLO_MODEL = 'best.pt'

    # --- НАСТРОЙКИ ФИЛЬТРАЦИИ ---
    MIN_PLATE_WIDTH = 30

    # --- НАСТРОЙКИ ЗОНЫ (В процентах от экрана) ---
    ROI_TOP_CUT = 0.40  # Отрезать верхние 40%
    ROI_LEFT_CUT = 0.15  # Отрезать левые 15%
    ROI_RIGHT_CUT = 0.15  # Отрезать правые 15%

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Ошибка открытия видео")
        return

    try:
        detector = YOLO(YOLO_MODEL)
        ocr_reader = LicensePlateOCR()
        observer = PlateObserver(confidence_threshold=2)
    except Exception as e:
        print(f"Ошибка моделей: {e}")
        return

    frame_count = 0
    SKIP_FRAMES = 2
    system_cooldown_until = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Видео закончилось.")
            break

        frame_count += 1
        h_full, w_full, _ = frame.shape

        # --- ВЫЧИСЛЯЕМ КООРДИНАТЫ ЗОНЫ ---
        roi_y_start = int(h_full * ROI_TOP_CUT)
        roi_x_start = int(w_full * ROI_LEFT_CUT)
        roi_x_end = int(w_full * (1 - ROI_RIGHT_CUT))

        # Вырезаем зону
        roi_frame = frame[roi_y_start:, roi_x_start:roi_x_end]

        # Трекинг
        results = detector.track(roi_frame, persist=True, verbose=False, conf=0.3, tracker="bytetrack.yaml")

        # Отрисовка линий зоны
        cv2.line(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_start), (255, 0, 0), 2)
        cv2.line(frame, (roi_x_start, roi_y_start), (roi_x_start, h_full), (255, 0, 0), 2)
        cv2.line(frame, (roi_x_end, roi_y_start), (roi_x_end, h_full), (255, 0, 0), 2)
        cv2.putText(frame, "ACTIVE ZONE", (roi_x_start + 10, roi_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

        current_time = time.time()
        is_system_paused = current_time < system_cooldown_until

        if is_system_paused:
            remaining = int(system_cooldown_until - current_time)
            cv2.putText(frame, f"GATE OPEN! WAIT {remaining}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                        3)

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                roi_x1, roi_y1, roi_x2, roi_y2 = box

                # Пересчет координат
                x1 = roi_x1 + roi_x_start
                x2 = roi_x2 + roi_x_start
                y1 = roi_y1 + roi_y_start
                y2 = roi_y2 + roi_y_start

                box_width = x2 - x1

                if box_width < MIN_PLATE_WIDTH:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    continue

                color = (0, 255, 255)  # Желтый

                if track_id in observer.confirmed:
                    final_text = observer.confirmed[track_id]
                    color = (0, 255, 0)  # Зеленый
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{track_id} {final_text} (OPEN)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    continue

                if is_system_paused:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    continue

                if frame_count % SKIP_FRAMES == 0:
                    pad = 10
                    crop_x1 = max(0, x1 - pad)
                    crop_y1 = max(0, y1 - pad)
                    crop_x2 = min(w_full, x2 + pad)
                    crop_y2 = min(h_full, y2 + pad)

                    plate_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    text = ocr_reader.recognize(plate_img)

                    if text:
                        print(f"✅ Чтение: {text}")
                        winner = observer.add_reading(track_id, text)

                        if winner:
                            print(f"🔥🔥🔥 ВОРОТА ОТКРЫТЫ ДЛЯ: {winner}")
                            system_cooldown_until = time.time() + 5.0

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} Analyzing...", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Smart Checkpoint', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Работа завершена.")


if __name__ == '__main__':
    main()