# Анализатор тональности отзывов IMDB

## Тема
Проект выполняет автоматический анализ тональности (сентимент-анализ) текстовых отзывов на фильмы из базы IMDb. На основе текста отзыва система определяет, является ли он положительным (🟢) или отрицательным (🔴).

## Цель
Цель работы — построить модель машинного обучения, способную с высокой точностью классифицировать отзывы на две полярности, и предоставить её через удобный веб-интерфейс для конечного пользователя.

## Источник данных
Исходный набор данных взят из **IMDb Dataset of 50K Movie Reviews** (открытый набор данных Kaggle). В проекте используется файл `imdb_dataset.csv`, содержащий две колонки:
- `review` — текст отзыва (на английском языке).
- `sentiment` — метка класса (`positive` / `negative`).

Фрагмент датасета выглядит следующим образом:
| review | sentiment |
|--------|-----------|
| "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked..." | positive |
| "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet..." | negative |

Объём набора данных в проекте: ~10 000 записей (исходный фрагмент содержит 495 отзывов).

## Структура проекта
IMDB-research/
├── data/
│ └── imdb_dataset.csv # Исходные данные
├── models/
│ └── model.pkl # Обученная модель и векторизатор
├── src/
│ ├── train.py # Скрипт обучения и сохранения модели
│ └── app.py # Веб-интерфейс на Gradio
├── tests/
│ └── test_basic.py # Модульные тесты
├── results/
│ ├── confusion_matrix.png # Матрица ошибок
│ └── log.csv # Лог запросов
├── requirements.txt # Зависимости Python
├── .gitignore # Игнорируемые файлы
└── README.md # Данный файл

## Используемые технологии
- **Python 3.9+**
- **pandas, numpy** – работа с данными
- **scikit-learn** – векторизация (`TfidfVectorizer`), логистическая регрессия, оценка качества
- **matplotlib, seaborn** – визуализация
- **gradio** – веб-интерфейс
- **pytest** – автоматическое тестирование

## Установка и запуск

### 1. Клонирование репозитория

git clone <URL репозитория>
cd IMDB-research
2. Установка зависимостей
pip install -r requirements.txt
3. Обучение модели
cd src
python train.py
4. Запуск веб-интерфейса
cd src
python app.py
5. Запуск тестов
cd tests
pytest test_basic.py -v
