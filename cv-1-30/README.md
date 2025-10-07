# Simple-OCR
Простая программа распознавания текста с помощью tesseract

# Dependencies
```
pip install -r requirements.txt
```
> [!NOTE]
> На Windows вы должны добавить Tesseract в переменные среды или указывать путь до tesseract.exe через аргумент `--tesseract-cmd`

# Usage
```
python pic2txt.py path/to/image.jpg
```
```
python pic2txt.py path/to/image.jpg --show --psm 6 --output path/to/text.txt
```

## Optional arguments

- `--show` - Показать предобработанное изображение
- `--psm` - Page Segmentation Mode для Tesseract *(по умолчанию 1)*
- `--output`, `-o` - Путь для сохранения результат в файл
- `--lang` - Языки для Tesseract (по умолчанию eng+rus)
- `--scale`, `-s` - Коэффицент изменения размера *(по умолчанию 1.5)* 
- `--tesseract-cmd` - Путь к исполняемому файлу tesseract
- `--mode` - Режимы предобработки: "default", "document", "newspaper", "inverted", "comic" *(по умолчанию default)* 


### Modes
- **Document** - Универсальная бинаризация документов с легкой чисткой
- **Newspaper** - Режим для газет, агрессивный контраст, удаляем декоративные элементы
- **Inverted** - тот же document, но инвертирует изображение перед обработкой
- **Comic** - Текст с декорациями, агрессивная адаптивная бинаризация. 
- **Default** - выбирает между **document** и **inverted**

# Examples
<img width="1524" height="524" alt="1" src="https://github.com/user-attachments/assets/22f26da1-9471-4a5b-9a2f-74ddc62a9d6f" />

<img width="1106" height="526" alt="2" src="https://github.com/user-attachments/assets/7ab935a5-17c2-497d-9e0e-9d84effdde8f" />

<img width="1188" height="493" alt="3" src="https://github.com/user-attachments/assets/9df4d662-1b94-43d7-bea3-6ced3151c442" />



