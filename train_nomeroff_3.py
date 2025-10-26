from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

def load_model():
    """Загрузка модели по конкретному пути"""
    print("🔍 ЗАГРУЗКА МОДЕЛИ")
    print("=" * 30)
    
    # Конкретный путь к вашей модели
    model_path = Path(r"H:\Учёба\Хакатоны\Цифра\2025-26\Камера\Цифра\runs\detect\nomeroff_yolov8n\weights\best.pt")
    
    if model_path.exists():
        print(f"✅ Модель найдена: {model_path}")
        
        # Загружаем модель
        model = YOLO(str(model_path))
        print(f"✅ Модель загружена")
        print(f"📊 Классы: {model.names}")
        
        return model
    else:
        print(f"❌ Модель не найдена по пути: {model_path}")
        return None

def preprocess_license_plate(plate_image):
    """Предобработка изображения номерного знака для OCR"""
    # Конвертируем в серый
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Увеличиваем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    
    # Бинаризация
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Убираем шум
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def recognize_license_plate_text(plate_image):
    """Распознавание текста на номерном знаке"""
    try:
        # Предобработка
        processed_plate = preprocess_license_plate(plate_image)
        
        # Настройки для Tesseract (русские буквы + цифры)
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Распознавание текста
        text = pytesseract.image_to_string(processed_plate, config=custom_config)
        
        # Очистка текста
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        
        return cleaned_text if cleaned_text else "Не распознано"
        
    except Exception as e:
        print(f"❌ Ошибка OCR: {e}")
        return "Ошибка распознавания"

def process_image_with_model(model, image_path):
    """Обработка одного изображения моделью с OCR"""
    print(f"\n🔍 Обработка изображения: {image_path}")
    
    # Загружаем изображение
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return
    
    # Детекция с помощью модели
    results = model(image)
    
    # Копируем изображение для рисования
    output_image = image.copy()
    
    # Обрабатываем результаты
    detections_count = 0
    for r in results:
        if len(r.boxes) > 0:
            for i, box in enumerate(r.boxes):
                # Получаем координаты bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Вырезаем область номерного знака
                plate_roi = image[y1:y2, x1:x2]
                
                # Распознаем текст на номерном знаке
                plate_text = recognize_license_plate_text(plate_roi)
                
                # Рисуем bounding box
                color = (0, 255, 0)  # Зеленый
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
                
                # Добавляем текст с номером и уверенностью
                label = f"{plate_text} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Фон для текста
                cv2.rectangle(output_image, (x1, y1 - label_size[1] - 15), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Текст
                cv2.putText(output_image, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detections_count += 1
                print(f"   ✅ Номер {i+1}: {plate_text} (уверенность: {confidence:.3f})")
                
                # Сохраняем вырезанный номерной знак (для отладки)
                plate_filename = f"plate_{image_path.stem}_{i+1}.jpg"
                cv2.imwrite(plate_filename, plate_roi)
                print(f"   💾 Номер сохранен: {plate_filename}")
    
    if detections_count == 0:
        print("   ❌ Номера не обнаружены")
    
    # Показываем результат
    window_name = f"Результат: {image_path.name}"
    cv2.imshow(window_name, output_image)
    print("🖼️ Нажмите любую клавишу для продолжения...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Сохраняем результат
    output_path = f"detected_{image_path.name}"
    cv2.imwrite(output_path, output_image)
    print(f"💾 Результат сохранен: {output_path}")

def process_video_with_model(model, video_path):
    """Обработка видео моделью с OCR"""
    print(f"\n🎥 Обработка видео: {video_path}")
    
    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Информация о видео:")
    print(f"   FPS: {fps}")
    print(f"   Всего кадров: {total_frames}")
    
    print("\n🎬 Воспроизведение видео с детекцией и OCR...")
    print("   Нажмите 'q' для выхода, 'p' для паузы")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ Видео завершено")
                break
            
            frame_count += 1
            
            # Обрабатываем каждый N-ый кадр для производительности
            if frame_count % 5 == 0:
                # Детекция на текущем кадре
                results = model(frame)
                
                # Рисуем bounding boxes и распознаем текст
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        
                        # Вырезаем номерной знак
                        plate_roi = frame[y1:y2, x1:x2]
                        if plate_roi.size > 0:
                            # Распознаем текст
                            plate_text = recognize_license_plate_text(plate_roi)
                            
                            # Рисуем прямоугольник
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Добавляем текст
                            label = f"{plate_text} ({confidence:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Показываем кадр
        cv2.imshow(f"Видео: {video_path.name}", frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸️ Пауза" if paused else "▶️ Продолжение")
    
    cap.release()
    cv2.destroyAllWindows()

def check_tesseract_installation():
    """Проверка установки Tesseract"""
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR установлен")
        return True
    except:
        print("❌ Tesseract OCR не установлен")
        print("📥 Установите Tesseract:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Linux: sudo apt-get install tesseract-ocr")
        print("   Mac: brew install tesseract")
        return False

def main():
    """Основная функция"""
    # Проверяем установку Tesseract
    if not check_tesseract_installation():
        return
    
    # Загружаем модель по конкретному пути
    model = load_model()
    if not model:
        print("❌ Не удалось загрузить модель")
        return
    
    print("\n🎯 ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
    print("1. Обработка изображений из папки")
    print("2. Обработка видео файла")
    print("3. Обработка веб-камеры")
    
    choice = input("Введите номер (1-3): ").strip()
    
    if choice == "1":
        # Обработка изображений из папки
        folder_path = input("Введите путь к папке с изображениями (или нажмите Enter для папки 'test_images'): ").strip()
        if not folder_path:
            folder_path = "test_images"
        
        image_folder = Path(folder_path)
        if not image_folder.exists():
            print(f"❌ Папка {folder_path} не найдена")
            return
        
        # Ищем все изображения
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(ext))
            image_files.extend(image_folder.glob(ext.upper()))
        
        if not image_files:
            print(f"❌ Изображения не найдены в папке {folder_path}")
            return
        
        print(f"📁 Найдено изображений: {len(image_files)}")
        
        # Обрабатываем каждое изображение
        for img_path in sorted(image_files):
            process_image_with_model(model, img_path)
    
    elif choice == "2":
        # Обработка видео файла
        video_path = input("Введите путь к видео файлу: ").strip()
        if not video_path:
            print("❌ Путь к видео не указан")
            return
        
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"❌ Видео файл не найден: {video_path}")
            return
        
        process_video_with_model(model, video_file)
    
    elif choice == "3":
        # Обработка веб-камеры (упрощенная версия)
        print("📹 Запуск веб-камеры с OCR...")
        print("   Нажмите 'q' для выхода")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Не удалось открыть веб-камеру")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем каждый 10-ый кадр для производительности
            if frame_count % 10 == 0:
                results = model(frame)
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        
                        plate_roi = frame[y1:y2, x1:x2]
                        if plate_roi.size > 0:
                            plate_text = recognize_license_plate_text(plate_roi)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{plate_text} ({confidence:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Веб-камера - детекция номеров с OCR", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    main()