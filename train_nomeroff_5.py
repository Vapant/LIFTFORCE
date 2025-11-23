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
VIDEO_SOURCE = r'test_video_2.mp4'


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

    # ... (импорты и инициализация модели остаются те же) ...

    # Словарь для защиты от повторных срабатываний: { "A123BC77": timestamp }
    sent_plates = {}

    # Буфер для "голосования" (упрощенный): собираем последние распознавания
    # Чтобы считать номер подтвержденным
    plate_buffer = []

    frame_count = 0
    SKIP_FRAMES = 3  # Обрабатывать только каждый 5-й кадр (ускорение в 5 раз)
    MIN_PLATE_WIDTH = 30  # Минимальная ширина номера в пикселях для OCR
    COOLDOWN_SECONDS = 1  # Не отправлять один и тот же номер чаще чем раз в 15 сек

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Просто отображаем видео (без обработки) для плавности,
        # если это пропускаемый кадр
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            cv2.imshow('Smart Checkpoint', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # Пропускаем тяжелую логику

        start_time = time.time()

        # --- 1. ДЕТЕКЦИЯ (YOLO) ---
        results = model(frame, stream=True, verbose=False, conf=0.4)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                # Вычисляем ширину номера
                plate_w = x2 - x1
                plate_h = y2 - y1

                # Рисуем рамку всегда, чтобы видеть, что YOLO работает
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- 2. ФИЛЬТР: Если номер слишком мелкий или слишком далеко ---
                # Это экономит ресурсы, не запуская Tesseract на мусоре
                if plate_w < MIN_PLATE_WIDTH:
                    continue

                # --- 3. ПОДГОТОВКА К OCR ---
                pad = 5
                h_img, w_img, _ = frame.shape
                plate_roi = frame[max(0, y1 - pad):min(h_img, y2 + pad), max(0, x1 - pad):min(w_img, x2 + pad)]

                # --- 4. ТЯЖЕЛАЯ ОПЕРАЦИЯ: OCR ---
                # Запускаем только если прошли фильтры
                text, _ = ocr.recognize(plate_roi)

                if text:
                    current_time = time.time()

                    # Визуализация
                    label = f"{text}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Вижу номер: {text}")

                    # --- 5. ЛОГИКА БИЗНЕС-ПРОЦЕССА (Защита сервера) ---
                    # Проверяем, отправляли ли мы этот номер недавно
                    last_sent = sent_plates.get(text, 0)

                    if (current_time - last_sent) > COOLDOWN_SECONDS:
                        # === ЗДЕСЬ КОД ОТПРАВКИ НА СЕРВЕР ===
                        print(f">>> ЗАПРОС В БД: Открываем шлагбаум для {text} <<<")
                        # requests.post(...)

                        # Обновляем время последней отправки
                        sent_plates[text] = current_time
                    else:
                        print(f"Дубликат. Игнорирую {text}")

        # FPS будет реальным (с учетом пропусков)
        fps = 1.0 / (time.time() - start_time) * SKIP_FRAMES
        cv2.putText(frame, f"FPS (Logic): {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Smart Checkpoint', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()