import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import sys
import os
import importlib.util
import io

# Установка UTF-8 кодировки для вывода
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Установка интерактивного режима для matplotlib
plt.ion()

def load_module_from_file(file_path, module_name):
    """Динамически загружает модуль из файла"""
    try:
        # ✅ Проверка существования файла перед загрузкой
        if not os.path.exists(file_path):
            print(f"Файл модуля не найден: {file_path}")
            return None
            
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Ошибка загрузки модуля {file_path}: {e}")
        return None

# Загружаем модули с проверкой путей
pic2txt_module = load_module_from_file('cv-1-30/pic2txt.py', 'pic2txt')
text_detection_module = load_module_from_file('cw-2-35/text_detection.py', 'text_detection')

# Проверяем загрузку модулей
if pic2txt_module is None:
    print("Не удалось загрузить модуль pic2txt.py")
if text_detection_module is None:
    print("Не удалось загрузить модуль text_detection.py")

def detect_plate_by_shape(img):
    """Детекция номерного знака по форме"""
    # ✅ Проверка входных данных
    if img is None:
        return None, None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
    edges = cv2.Canny(img_filter, 30, 200)

    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    pos = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            pos = approx
            break

    return pos, edges, gray

def detect_plate_by_color(img):
    """Детекция номерного знака по цвету"""
    # ✅ Проверка входных данных
    if img is None:
        return None
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # ❌ Хардкод цветовых диапазонов - можно вынести в параметры
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    color_mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    return color_mask

def extract_plate_region(img, pos):
    """Извлечение региона номерного знака"""
    # ✅ Проверка входных данных
    if img is None or pos is None:
        return None, None, None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [pos], 0, 255, -1)
    
    bitwise_img = cv2.bitwise_and(img, img, mask=mask)
    
    (x, y) = np.where(mask == 255)
    if len(x) == 0 or len(y) == 0:
        return None, None, None, None
    
    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)
    
    # ✅ Проверка валидности координат
    if x2 <= x1 or y2 <= y1:
        return None, None, None, None
        
    cropped_color = img[x1:x2, y1:y2]
    cropped_gray = gray[x1:x2, y1:y2]
    
    return cropped_color, cropped_gray, bitwise_img, (x1, y1, x2, y2)

def run_east_detection(image_path):
    """Правильный запуск EAST с имитацией ввода"""
    try:
        if text_detection_module is None:
            print("Модуль text_detection не загружен")
            return None
        # ✅ Проверка существования файла изображения
        if not os.path.exists(image_path):
            print(f"Файл изображения не найден: {image_path}")
            return None
        print("Запуск EAST детекции...")
        # Сохраняем оригинальный stdin
        original_stdin = sys.stdin
        # Создаем mock для ввода
        from io import StringIO
        input_data = f"{image_path} east_result.jpg\n"
        sys.stdin = StringIO(input_data)
        try:
            # Запускаем EAST детекцию
            text_detection_module.main()
        except Exception as e:
            print(f"Ошибка при выполнении EAST: {e}")
            return None
        finally:
            # Восстанавливаем stdin
            sys.stdin = original_stdin
        # Загружаем результат EAST
        if os.path.exists('east_result.jpg'):
            east_result = cv2.imread('east_result.jpg')
            print("EAST детекция завершена")
            return east_result
        else:
            print("EAST не смог обработать изображение")
            return None
    except Exception as e:
        print(f"Ошибка при работе EAST: {e}")
        return None

def recognize_plate_text(plate_image):
    """Распознавание текста с номером используя ваш OCR модуль"""
    try:
        if pic2txt_module is None:
            print("Модуль pic2txt не загружен")
            return ""
        # ✅ Проверка входного изображения
        if plate_image is None or plate_image.size == 0:
            print("Пустое изображение для OCR")
            return ""
        print("Запуск OCR...")
        text = pic2txt_module.perform_ocr(plate_image, mode="document", lang="eng", psm=8, scale=1.5)
        cleaned_text = ''.join(e for e in text if e.isalnum())
        return cleaned_text
    except Exception as e:
        print(f"Ошибка OCR: {e}")
        return ""

def preprocess_plate(plate_image):
    """Предобработка изображения для OCR"""
    try:
        if pic2txt_module is None:
            print("Модуль pic2txt не загружен")
            return plate_image
        # ✅ Проверка входного изображения
        if plate_image is None:
            return None
        return pic2txt_module.preprocess(plate_image, mode="document", scale=1.5)
    except Exception as e:
        print(f"Ошибка предобработки: {e}")
        return plate_image

def save_visualization(original, edges, color_mask, cropped_plate, processed_ocr, plate_text, plate_coords, east_result):
    """Сохранение визуализации в файл вместо показа"""
    try:
        # Создание папки, если она не существует
        result_dir = 'cv-3-03/res'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        fig = plt.figure(figsize=(20, 12))
        
        # Исходное изображение
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Исходное изображение')
        plt.axis('off')
        
        # Границы
        plt.subplot(2, 4, 2)
        if edges is not None:
            plt.imshow(edges, cmap='gray')
        plt.title('Границы (форма)')
        plt.axis('off')
        
        # Цветовая маска
        plt.subplot(2, 4, 3)
        if color_mask is not None:
            plt.imshow(color_mask, cmap='gray')
        plt.title('Цветовая маска')
        plt.axis('off')
        
        # Результат детекции
        result_img = original.copy()
        if plate_coords:
            x1, y1, x2, y2 = plate_coords
            cv2.rectangle(result_img, (y1, x1), (y2, x2), (0, 255, 0), 3)
            if plate_text:
                cv2.putText(result_img, plate_text, (y1, x1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        plt.subplot(2, 4, 4)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Результат: {plate_text}')
        plt.axis('off')
        
        # Вырезанный номер
        if cropped_plate is not None:
            plt.subplot(2, 4, 5)
            if len(cropped_plate.shape) == 3:
                plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(cropped_plate, cmap='gray')
            plt.title('Вырезанный номер')
            plt.axis('off')
        
        # Обработанный для OCR
        if processed_ocr is not None:
            plt.subplot(2, 4, 6)
            plt.imshow(processed_ocr, cmap='gray')
            plt.title('Для OCR')
            plt.axis('off')
        
        # EAST результат если есть
        if east_result is not None:
            plt.subplot(2, 4, 7)
            plt.imshow(cv2.cvtColor(east_result, cv2.COLOR_BGR2RGB))
            plt.title('EAST детекция')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Сохраняем вместо показа
        plt.savefig(os.path.join(result_dir, 'detection_results.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)  # Закрываем figure чтобы освободить память
        
        print("Результаты сохранены в detection_results.png")
        
    except Exception as e:
        print(f"Ошибка при сохранении визуализации: {e}")

def main():
    """Основная функция"""
    # ❌ Хардкод пути - можно сделать параметром
    image_path = 'cv-3-03/photo.jpg'
    
    # ✅ Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Ошибка: файл изображения не найден: {image_path}")
        print("Проверьте путь к файлу")
        return
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Ошибка: не удалось загрузить изображение")
        print("Возможные причины: неправильный формат или поврежденный файл")
        return
    
    original = img.copy()
    
    print("Начало обработки...")
    
    # 1. Детекция по форме
    pos, edges, gray = detect_plate_by_shape(img)
    
    # 2. Детекция по цвету
    color_mask = detect_plate_by_color(img)
    
    plate_detected = False
    cropped_plate = None
    plate_coords = None
    processed_ocr = None
    plate_text = ""
    
    # Приоритет: сначала по форме, потом по цвету
    if pos is not None:
        cropped_plate, cropped_gray, bitwise_img, plate_coords = extract_plate_region(img, pos)
        if cropped_plate is not None:
            plate_detected = True
            print("Номер обнаружен по форме")
    
    if not plate_detected:
        print("Поиск по цвету...")
        if color_mask is not None:
            color_contours = cv2.findContours(color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color_contours = imutils.grab_contours(color_contours)
            
            for contour in color_contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 2.0 < aspect_ratio < 5.0:
                        cropped_plate = original[y:y+h, x:x+w]
                        plate_coords = (y, x, y+h, x+w)
                        plate_detected = True
                        print("Номер обнаружен по цвету")
                        break
    
    # 3. Распознавание текста
    if plate_detected and cropped_plate is not None:
        # ✅ Проверка размера вырезанной области
        if cropped_plate.size > 0:
            plate_text = recognize_plate_text(cropped_plate)
            print(f"Распознанный текст: {plate_text}")
            
            # Сохранение вырезанного номера
            try:
                cv2.imwrite(os.path.join('cv-3-03', 'res', 'detected_plate.jpg'), cropped_plate)
                print("Вырезанный номер сохранен как detected_plate.jpg")
            except Exception as e:
                print(f"Ошибка сохранения номера: {e}")
            
            # Предобработка для OCR и сохранение
            processed_ocr = preprocess_plate(cropped_plate)
            if processed_ocr is not None:
                try:
                    cv2.imwrite('plate_for_ocr.jpg', processed_ocr)
                    print("Обработанный номер сохранен как plate_for_ocr.jpg")
                except Exception as e:
                    print(f"Ошибка сохранения обработанного номера: {e}")
        else:
            print("Вырезанная область пустая")
            plate_detected = False
    
    # 4. EAST детекция (опционально) - пропускаем если тормозит
    east_result = None
    try:
         east_result = run_east_detection(image_path)
    except Exception as e:
         print(f"EAST детекция пропущена: {e}")
    
    # Сохранение визуализации
    save_visualization(original, edges, color_mask, cropped_plate, processed_ocr, plate_text, plate_coords, east_result)
    
    # Вывод итоговой информации
    print("\n" + "="*50)
    if plate_detected:
        print("ИТОГ: Номерной знак обнаружен и сохранен")
        print(f"Распознанный текст: {plate_text}")
        print("Созданные файлы:") 
        print("  - detected_plate.jpg - вырезанный номер")
        print("  - plate_for_ocr.jpg - обработанный для OCR")
        print("  - detection_results.png - визуализация результатов")
        if east_result is not None:
            print("  - east_result.jpg - результат EAST детекции")
    else:
        print("ИТОГ: Номерной знак не обнаружен")
        print("Возможные причины:") 
        print("  - Номер не виден на изображении")
        print("  - Плохое качество изображения")
        print("  - Нестандартный размер/цвет номера")
    print("="*50)

if __name__ == "__main__":
    main()
