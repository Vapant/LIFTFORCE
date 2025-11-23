import cv2
import pytesseract
import numpy as np
import re
import time
from ultralytics import YOLO  # Библиотека, которую ты использовал для обучения

# ==========================================
# НАСТРОЙКИ
# ==========================================

# 1. Путь к Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Путь к твоей обученной модели
# В скрипте обучения она сохраняется в папке runs/detect/nomeroff_yolov8n/weights/best.pt
# Скопируй best.pt в папку с этим скриптом или укажи полный путь
MODEL_PATH = 'best.pt'

# 3. Источник видео
VIDEO_SOURCE = r'E:\Учёба\Языки программирования\LIFTFORCE\test_video.mp4'


# VIDEO_SOURCE = 0  # Раскомментируй для веб-камеры

# ==========================================
# КЛАСС OCR (Остался без изменений)
# ==========================================

class LicensePlateOCR:
    def __init__(self):
        self.whitelist = 'ABEKMHOPCTYX0123456789'
        self.config = f'--oem 3 --psm 7 -c tessedit_char_whitelist={self.whitelist}'

    def preprocess(self, img):
        scale_percent = 200
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def validate_text(self, text):
        clean = re.sub(r'[^ABEKMHOPCTYX0123456789]', '', text.upper())
        if 6 <= len(clean) <= 9:
            return clean
        return None

    def recognize(self, plate_img):
        try:
            processed = self.preprocess(plate_img)
            text = pytesseract.image_to_string(processed, lang='eng', config=self.config)
            valid_text = self.validate_text(text)
            return valid_text, processed
        except Exception as e:
            print(f"Ошибка OCR: {e}")
            return None, None


# ==========================================
# ОСНОВНАЯ ЛОГИКА (Адаптировано под YOLOv8)
# ==========================================

def main():
    print(f"Загрузка модели из {MODEL_PATH}...")
    try:
        # Загружаем модель YOLOv8
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Убедитесь, что файл best.pt находится в папке или укажите правильный путь.")
        return

    ocr = LicensePlateOCR()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Ошибка: Невозможно открыть видео.")
        return

    cv2.namedWindow('Smart Checkpoint', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # --- ДЕТЕКЦИЯ (YOLOv8) ---
        # stream=True делает работу быстрее, возвращая генератор
        results = model(frame, stream=True, verbose=False, conf=0.4)

        # YOLOv8 возвращает список результатов (по кадрам), у нас 1 кадр
        for result in results:
            boxes = result.boxes  # Получаем объект с рамками

            for box in boxes:
                # Координаты (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Уверенность (confidence)
                conf = float(box.conf[0])

                # Класс (если у тебя 1 класс "license_plate", это будет 0)
                cls = int(box.cls[0])

                # --- Вырезаем номер и распознаем ---
                h, w, _ = frame.shape
                pad = 5  # Отступ
                y1_c = max(0, y1 - pad)
                y2_c = min(h, y2 + pad)
                x1_c = max(0, x1 - pad)
                x2_c = min(w, x2 + pad)

                plate_roi = frame[y1_c:y2_c, x1_c:x2_c]

                if plate_roi.size == 0:
                    continue

                # OCR
                text, debug_img = ocr.recognize(plate_roi)

                # Рисуем
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if text:
                    label = f"{text} ({conf:.2f})"
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 15, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    print(f"НОМЕР: {text}")

                    if debug_img is not None:
                        cv2.imshow(f"Debug OCR", debug_img)
                else:
                    cv2.putText(frame, "Detecting...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Smart Checkpoint', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()