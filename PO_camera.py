import cv2
import re
import logging
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import warnings

logging.getLogger("ppocr").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class LicensePlateOCR:
    def __init__(self):
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    def preprocess_image(self, img):
        if img is None or img.size == 0: return None
        try:
            img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
            return img
        except Exception:
            return None

    def validate_text(self, text):
        # Убираем всё лишнее
        clean = re.sub(r'[^ABEKMHOPCTYX0123456789]', '', text.upper())

        # Длина полного номера РФ (с регионом) - 8 или 9 знаков.
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

                    # --- ИСПРАВЛЕНИЕ ОШИБКИ ЗДЕСЬ ---
                    try:
                        # Проверяем, является ли box[0] списком/массивом (полигон) или числом (bbox)
                        if hasattr(box[0], '__len__'):
                            x_coord = box[0][0]  # Формат [[x,y], [x,y]...]
                        else:
                            x_coord = box[0]  # Формат [x, y, x2, y2]
                    except Exception:
                        x_coord = 0  # На случай сбоя ставим в начало
                    # --------------------------------

                    if score > 0.4:
                        blocks.append((text, x_coord))

                # Сортируем слева направо
                blocks.sort(key=lambda b: b[1])

                # Склеиваем
                full_text_raw = "".join([b[0] for b in blocks])

                print(f"👁️ СКЛЕЕНО: '{full_text_raw}'")

                valid = self.validate_text(full_text_raw)
                if valid:
                    return valid

            return None

        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return None


def main():
    print("🚀 ЗАПУСК ДИАГНОСТИКИ...")

    VIDEO_SOURCE = 0
    YOLO_MODEL = 'best.pt'

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Ошибка камеры")
        return

    try:
        detector = YOLO(YOLO_MODEL)
        ocr_reader = LicensePlateOCR()
    except Exception as e:
        print(f"Ошибка моделей: {e}")
        return

    frame_count = 0

    # ВАЖНО: Убрали пропуск кадров, чтобы видеть результат сразу
    SKIP_FRAMES = 1

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        h_frame, w_frame, _ = frame.shape

        results = detector(frame, stream=True, verbose=False, conf=0.3)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Отступы 5 пикселей
                pad = 5
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w_frame, x2 + pad)
                y2 = min(h_frame, y2 + pad)

                # Рисуем синюю рамку (она у вас есть)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                plate_img = frame[y1:y2, x1:x2]

                # Пытаемся распознать
                text = ocr_reader.recognize(plate_img)
                if text:
                    print(f"✅✅✅ НОМЕР: {text}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow('Smart Checkpoint', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()