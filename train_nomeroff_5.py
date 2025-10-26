from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import re

class LicensePlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['ru', 'en'])
    
    def preprocess_plate(self, plate_image):
        lab = cv2.cvtColor(plate_image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def recognize_text(self, plate_image):
        try:
            processed_plate = self.preprocess_plate(plate_image)
            rgb_image = cv2.cvtColor(processed_plate, cv2.COLOR_BGR2RGB)
            results = self.reader.readtext(rgb_image)
            
            detected_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.6:
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if 3 <= len(clean_text) <= 12:
                        detected_text.append((clean_text, confidence))
            
            if detected_text:
                detected_text.sort(key=lambda x: x[1], reverse=True)
                return detected_text[0][0]
            else:
                return "Не распознано"
                
        except Exception as e:
            return "Ошибка"

def load_model():
    model_path = Path(r"H:\Учёба\Хакатоны\Цифра\2025-26\Камера\Цифра\runs\detect\nomeroff_yolov8n\weights\best.pt")
    
    if model_path.exists():
        model = YOLO(str(model_path))
        return model
    else:
        return None

def process_image_with_model(model, ocr_engine, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None
    
    results = model(image)
    output_image = image.copy()
    
    detected_numbers = []  # Переменная для хранения распознанных номеров
    
    for r in results:
        if len(r.boxes) > 0:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                plate_roi = image[y1:y2, x1:x2]
                plate_text = ocr_engine.recognize_text(plate_roi)
                
                # Сохраняем номер в переменную
                detected_numbers.append({
                    'number': plate_text,
                    'confidence': float(confidence),
                    'position': (x1, y1, x2, y2)
                })
                
                color = (0, 255, 0)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
                
                label = f"{plate_text} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                cv2.rectangle(output_image, (x1, y1 - label_size[1] - 15), 
                            (x1 + label_size[0], y1), color, -1)
                
                cv2.putText(output_image, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                plate_filename = f"ans/plate_{image_path.stem}_{i+1}.jpg"
                cv2.imwrite(plate_filename, plate_roi)
    
    # Сохраняем результат
    output_path = f"ans/detected_{image_path.name}"
    cv2.imwrite(output_path, output_image)
    
    return detected_numbers

def process_video_with_model(model, ocr_engine, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return
    
    all_detected_numbers = []  # Переменная для хранения всех номеров из видео
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 5 == 0:
                results = model(frame)
                
                frame_numbers = []  # Номера на текущем кадре
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        
                        plate_roi = frame[y1:y2, x1:x2]
                        if plate_roi.size > 0:
                            plate_text = ocr_engine.recognize_text(plate_roi)
                            
                            # Сохраняем номер
                            frame_numbers.append({
                                'number': plate_text,
                                'confidence': float(confidence),
                                'frame': frame_count,
                                'position': (x1, y1, x2, y2)
                            })
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{plate_text} ({confidence:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Выводим номера с текущего кадра
                if frame_numbers:
                    print(f"\nКадр {frame_count}:")
                    for num_data in frame_numbers:
                        print(f"  Номер: {num_data['number']} (уверенность: {num_data['confidence']:.3f})")
                    all_detected_numbers.extend(frame_numbers)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    return all_detected_numbers

def main():
    ocr_engine = LicensePlateOCR()
    model = load_model()
    
    if not model:
        print("Не удалось загрузить модель")
        return
    
    print("1. Обработка изображений из папки")
    print("2. Обработка видео файла")
    
    choice = input("Введите номер (1-2): ").strip()
    
    if choice == "1":
        folder_path = input("Введите путь к папке с изображениями: ").strip()
        if not folder_path:
            folder_path = "test_images"
        
        image_folder = Path(folder_path)
        if not image_folder.exists():
            print(f"Папка {folder_path} не найдена")
            return
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(ext))
            image_files.extend(image_folder.glob(ext.upper()))
        
        if not image_files:
            print(f"Изображения не найдены в папке {folder_path}")
            return
        
        print(f"Найдено изображений: {len(image_files)}")
        
        all_detected_numbers = []  # Переменная для всех номеров
        
        for img_path in sorted(image_files):
            print(f"\n--- Обработка: {img_path.name} ---")
            detected_numbers = process_image_with_model(model, ocr_engine, img_path)
            
            if detected_numbers:
                all_detected_numbers.extend(detected_numbers)
                # Выводим распознанные номера для этого изображения
                for num_data in detected_numbers:
                    print(f"Распознанный номер: {num_data['number']} (уверенность: {num_data['confidence']:.3f})")
            else:
                print("Номера не обнаружены")
        
        # Итоговый вывод всех распознанных номеров
        print(f"\n{'='*50}")
        print("ИТОГОВЫЙ ОТЧЕТ:")
        print(f"{'='*50}")
        if all_detected_numbers:
            for i, num_data in enumerate(all_detected_numbers, 1):
                print(f"{i}. Номер: {num_data['number']} (уверенность: {num_data['confidence']:.3f})")
        else:
            print("Номера не обнаружены ни на одном изображении")
    
    elif choice == "2":
        video_path = input("Введите путь к видео файлу: ").strip()
        if not video_path:
            print("Путь к видео не указан")
            return
        
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"Видео файл не найден: {video_path}")
            return
        
        print("Обработка видео...")
        all_detected_numbers = process_video_with_model(model, ocr_engine, video_file)
        
        # Итоговый вывод всех распознанных номеров из видео
        print(f"\n{'='*50}")
        print("ИТОГОВЫЙ ОТЧЕТ ПО ВИДЕО:")
        print(f"{'='*50}")
        if all_detected_numbers:
            # Группируем по номерам для устранения дубликатов
            unique_numbers = {}
            for num_data in all_detected_numbers:
                number = num_data['number']
                if number not in unique_numbers or num_data['confidence'] > unique_numbers[number]['confidence']:
                    unique_numbers[number] = num_data
            
            for i, (number, num_data) in enumerate(unique_numbers.items(), 1):
                print(f"{i}. Номер: {number} (макс. уверенность: {num_data['confidence']:.3f})")
        else:
            print("Номера не обнаружены в видео")
    
    else:
        print("Неверный выбор")

if __name__ == "__main__":
    main()