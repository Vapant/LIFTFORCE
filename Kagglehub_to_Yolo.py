import kagglehub
import cv2
import os
import shutil
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET


class CarPlateDatasetProcessorXML:
    def __init__(self):
        self.dataset_path = None
        self.yolo_path = Path('./car_plate_yolo_dataset')

    def download_dataset(self):
        """Скачивание датасета car-plate-detection"""
        print("📥 СКАЧИВАНИЕ ДАТАСЕТА CAR-PLATE-DETECTION")
        print("=" * 50)

        try:
            self.dataset_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
            print(f"✅ Датасет скачан: {self.dataset_path}")
            return True
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            return False

    def explore_dataset(self):
        """Исследование структуры датасета"""
        print("\n🔍 ИССЛЕДОВАНИЕ СТРУКТУРЫ ДАТАСЕТА")
        print("=" * 45)

        if not self.dataset_path:
            print("❌ Датасет не скачан")
            return False

        dataset_dir = Path(self.dataset_path)

        # Показываем структуру
        for item in dataset_dir.iterdir():
            if item.is_dir():
                print(f"📁 {item.name}/")
                files = list(item.iterdir())
                for file in files[:5]:
                    print(f"   📄 {file.name}")
                if len(files) > 5:
                    print(f"   ... и еще {len(files) - 5} файлов")

        # Проверяем наличие папок
        annotations_dir = dataset_dir / 'annotations'
        images_dir = dataset_dir / 'images'

        if annotations_dir.exists() and images_dir.exists():
            xml_files = list(annotations_dir.glob('*.xml'))
            image_files = list(images_dir.glob('*.*'))
            print(f"\n📊 Статистика:")
            print(f"   XML аннотаций: {len(xml_files)}")
            print(f"   Изображений: {len(image_files)}")
            return True
        else:
            print("❌ Папки annotations или images не найдены")
            return False

    def parse_xml_annotation(self, xml_path):
        """Парсинг XML аннотации"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Информация об изображении
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Ищем объекты (номера)
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')

                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                objects.append({
                    'class': name,
                    'bbox': [xmin, ymin, xmax, ymax]
                })

            return {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            }

        except Exception as e:
            print(f"❌ Ошибка парсинга {xml_path}: {e}")
            return None

    def process_annotations(self):
        """Обработка XML аннотаций и создание YOLO формата"""
        print("\n🔄 ОБРАБОТКА XML АННОТАЦИЙ")
        print("=" * 40)

        dataset_dir = Path(self.dataset_path)
        annotations_dir = dataset_dir / 'annotations'
        images_dir = dataset_dir / 'images'

        if not annotations_dir.exists() or not images_dir.exists():
            print("❌ Папки annotations или images не найдены")
            return False

        # Создаем YOLO структуру
        self.create_yolo_structure()

        # Собираем все XML файлы
        xml_files = list(annotations_dir.glob('*.xml'))
        print(f"📊 Найдено XML файлов: {len(xml_files)}")

        # Парсим все аннотации
        images_data = {}
        valid_count = 0

        for xml_file in tqdm(xml_files, desc="Парсинг XML"):
            annotation = self.parse_xml_annotation(xml_file)
            if annotation and annotation['objects']:
                images_data[annotation['filename']] = annotation
                valid_count += 1

        print(f"✅ Успешно распарсено: {valid_count}/{len(xml_files)}")

        if valid_count == 0:
            print("❌ Не найдено валидных аннотаций")
            return False

        # Разделяем на train/val/test
        image_names = list(images_data.keys())
        train_names, temp_names = train_test_split(image_names, test_size=0.3, random_state=42)
        val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=42)

        print(f"🎯 Разделение: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")

        # Обрабатываем каждое разделение
        self.process_split('train', train_names, images_data, images_dir)
        self.process_split('val', val_names, images_data, images_dir)
        self.process_split('test', test_names, images_data, images_dir)

        # Создаем YAML конфиг
        self.create_yaml_config()

        return True

    def create_yolo_structure(self):
        """Создание структуры папок YOLO"""
        print("📁 СОЗДАНИЕ СТРУКТУРЫ YOLO")

        folders = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test',
            'images_processed/train', 'images_processed/val', 'images_processed/test'
        ]

        for folder in folders:
            (self.yolo_path / folder).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Создана: {folder}")

    def process_split(self, split_name, image_names, images_data, images_dir):
        """Обработка одного разделения (train/val/test)"""
        print(f"\n🔧 Обработка {split_name}...")

        processed_count = 0
        for image_name in tqdm(image_names, desc=f"  {split_name}"):
            # Ищем файл изображения
            image_path = images_dir / image_name

            # Пробуем разные расширения если оригинальное не найдено
            if not image_path.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                    potential_path = images_dir / f"{Path(image_name).stem}{ext}"
                    if potential_path.exists():
                        image_path = potential_path
                        break

            if not image_path.exists():
                continue

            # Загружаем изображение
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            annotation = images_data[image_name]
            img_height, img_width = annotation['height'], annotation['width']

            # Создаем YOLO аннотации
            yolo_annotations = []
            for obj in annotation['objects']:
                xmin, ymin, xmax, ymax = obj['bbox']

                # Конвертация в YOLO формат
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # Проверка валидности
                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 <= width <= 1 and 0 <= height <= 1):
                    # Класс 0 для номерного знака
                    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if not yolo_annotations:
                continue

            # Копируем оригинальное изображение
            dest_image_path = self.yolo_path / 'images' / split_name / image_path.name
            shutil.copy2(image_path, dest_image_path)

            # Обрабатываем изображение (серый + оптимизация)
            processed_image = self.process_image_for_training(image)
            processed_image_path = self.yolo_path / 'images_processed' / split_name / image_path.name
            cv2.imwrite(str(processed_image_path), processed_image)

            # Сохраняем аннотации
            annotation_path = self.yolo_path / 'labels' / split_name / f"{image_path.stem}.txt"
            with open(annotation_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_annotations))

            processed_count += 1

        print(f"   ✅ Обработано: {processed_count} изображений")

    def process_image_for_training(self, image):
        """
        Улучшенная обработка изображения для обучения
        Более мягкие методы, сохраняющие читаемость номеров
        """
        # 1. Конвертация в серый (сохраняем оригинальные оттенки)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. Мягкое шумоподавление
        denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)

        # 3. Умеренное улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Уменьшен clipLimit
        contrast_enhanced = clahe.apply(denoised)

        # 4. Возвращаем улучшенное серое изображение без бинаризации
        return contrast_enhanced

    def create_yaml_config(self):
        """Создание YAML конфигурационного файла"""
        config = {
            'path': str(self.yolo_path.resolve()),
            'train': 'images_processed/train',  # Используем обработанные изображения
            'val': 'images_processed/val',
            'test': 'images_processed/test',
            'nc': 1,
            'names': ['license_plate']
        }

        with open('car_plate_dataset.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"\n✅ YAML конфиг создан: car_plate_dataset.yaml")

    def show_processing_example(self):
        """Показ примера обработки изображения"""
        print("\n🎨 ПРИМЕР ОБРАБОТКИ ИЗОБРАЖЕНИЯ")
        print("=" * 40)

        # Находим любое изображение для демонстрации
        train_images = list((self.yolo_path / 'images' / 'train').glob('*.*'))
        if not train_images:
            print("❌ Изображения для демонстрации не найдены")
            return

        example_image_path = train_images[0]
        original = cv2.imread(str(example_image_path))
        processed = self.process_image_for_training(original)

        print(f"📐 Оригинал: {original.shape}")
        print(f"📐 Обработанное: {processed.shape}")
        print(f"🎯 Тип: {'Серое' if len(processed.shape) == 2 else 'Цветное'}")

        # Сохраняем пример для наглядности
        cv2.imwrite('example_original.jpg', original)
        cv2.imwrite('example_processed.jpg', processed)
        print("💡 Примеры сохранены: example_original.jpg, example_processed.jpg")

    def analyze_dataset_quality(self):
        """Анализ качества датасета"""
        print("\n📊 АНАЛИЗ КАЧЕСТВА ДАТАСЕТА")
        print("=" * 35)

        dataset_dir = Path(self.dataset_path)
        annotations_dir = dataset_dir / 'annotations'

        if not annotations_dir.exists():
            print("❌ Папка annotations не найдена")
            return

        xml_files = list(annotations_dir.glob('*.xml'))

        # Анализируем несколько XML файлов
        sizes = []
        objects_per_image = []

        for xml_file in xml_files[:100]:  # Анализируем первые 100
            annotation = self.parse_xml_annotation(xml_file)
            if annotation:
                sizes.append((annotation['width'], annotation['height']))
                objects_per_image.append(len(annotation['objects']))

        if sizes:
            avg_width = np.mean([s[0] for s in sizes])
            avg_height = np.mean([s[1] for s in sizes])
            avg_objects = np.mean(objects_per_image)

            print(f"📐 Средний размер изображения: {avg_width:.0f}x{avg_height:.0f}")
            print(f"🎯 Среднее количество номеров на изображение: {avg_objects:.2f}")
            print(f"📊 Размеры выборки: {len(sizes)} изображений")


def main():
    """Основная функция подготовки датасета"""
    processor = CarPlateDatasetProcessorXML()

    # 1. Скачивание датасета
    if not processor.download_dataset():
        return

    # 2. Исследование структуры
    if not processor.explore_dataset():
        return

    # 3. Анализ качества
    processor.analyze_dataset_quality()

    # 4. Обработка аннотаций и создание YOLO формата
    if not processor.process_annotations():
        return

    # 5. Показ примера обработки
    processor.show_processing_example()

    print(f"\n🎉 ДАТАСЕТ ПОДГОТОВЛЕН!")
    print(f"📁 Путь: {processor.yolo_path}")
    print(f"🚀 Для обучения используйте: car_plate_dataset.yaml")


if __name__ == "__main__":
    main()