import cv2
import pytesseract
import numpy as np
import re
import time
from ultralytics import YOLO

# ==========================================
# НАСТРОЙКИ
# ==========================================

# 1. Пути (Проверь их!)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
MODEL_PATH = 'best.pt'  # Твой обученный файл
VIDEO_SOURCE = r'test_video.mp4'
# VIDEO_SOURCE = 0 # Для веб-камеры

# 2. Настройки оптимизации
SKIP_FRAMES = 5  # Обрабатывать только каждый 5-й кадр
MIN_PLATE_WIDTH = 60  # Игнорировать номера меньше 60px шириной
COOLDOWN_SECONDS = 10  # Не отправлять один и тот же номер чаще чем раз в 10 сек


# ==========================================
# КЛАСС OCR
# ==========================================

class LicensePlateOCR:
    def __init__(self):
        self.whitelist = 'ABEKMHOPCTYX0123456789'
        self.config = f'--oem 3 --psm 7 -c tessedit_char_whitelist={self.whitelist}'

    def preprocess(self, img):
        # Увеличение
        scale = 2
        w = int(img.shape[1] * scale)
        h = int(img.shape[0] * scale)
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        # ЧБ + Размытие
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Бинаризация (получаем черно-белую маску)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def validate_text(self, text):
        # Оставляем только допустимые символы
        clean = re.sub(r'[^ABEKMHOPCTYX0123456789]', '', text.upper())
        # Простая проверка длины (РФ номера обычно 8-9 символов)
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
# MAIN
# ==========================================

def main():
    print("Загрузка модели...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Не найдена модель {MODEL_PATH}. Проверь путь.")
        return

    ocr = LicensePlateOCR()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    # Словари для логики
    sent_plates = {}  # {номер: время_последней_отправки}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ОПТИМИЗАЦИЯ 1: Пропускаем кадры для ускорения
        # (показываем видео всегда, но нейросети запускаем редко)
        if frame_count % SKIP_FRAMES != 0:
            cv2.imshow('Smart Checkpoint', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        start_time = time.time()

        # --- ДЕТЕКЦИЯ ---
        results = model(frame, stream=True, verbose=False, conf=0.4)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # ОПТИМИЗАЦИЯ 2: Фильтр по размеру
                width = x2 - x1
                if width < MIN_PLATE_WIDTH:
                    continue

                # Рисуем рамку на основном видео
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Вырезаем номер с отступом
                pad = 5
                h_img, w_img, _ = frame.shape
                plate_roi = frame[max(0, y1 - pad):min(h_img, y2 + pad), max(0, x1 - pad):min(w_img, x2 + pad)]

                if plate_roi.size == 0:
                    continue

                # --- РАСПОЗНАВАНИЕ ---
                text, debug_img = ocr.recognize(plate_roi)

                if debug_img is not None:
                    # Чтобы нарисовать ЦВЕТНОЙ текст на ЧБ картинке, нужно перевести её обратно в BGR
                    debug_color = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

                    if text:
                        # Рисуем текст прямо на вырезанном номере (Debug окно)
                        # Координаты (10, 30) - примерное место начала текста
                        cv2.putText(debug_color, text, (10, debug_color.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Рисуем текст на основном окне
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # ЛОГИКА БИЗНЕСА (Кулдаун)
                        current_time = time.time()
                        last_sent = sent_plates.get(text, 0)
                        if (current_time - last_sent) > COOLDOWN_SECONDS:
                            print(f">>> ОТКРЫТЬ ШЛАГБАУМ: {text} <<<")
                            sent_plates[text] = current_time

                    # Показываем окно с вырезанным номером и нарисованным текстом
                    cv2.imshow("Debug Plate", debug_color)

        # Отображение
        cv2.imshow('Smart Checkpoint', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()