import os
import sys

# Добавляем корневую папку проекта в пути, чтобы тесты "видели" файл app.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)

from app import predict_sentiment, iface


def test_model_predict_success():
    """Тест 1: Проверяет, что основная функция загружается и работает на корректном примере."""
    # Проверяем, что файл модели на месте
    assert os.path.exists('model.pkl'), "Файл model.pkl не найден в корне проекта"

    # Проверяем саму функцию
    result = predict_sentiment("This is a great test movie")
    assert result is not None, "Функция вернула пустое значение"


def test_prediction_format():
    """Тест 2: Проверяет, что функция возвращает ответ в правильном текстовом формате."""
    result = predict_sentiment("Awful experience")

    # Проверяем тип данных
    assert isinstance(result, str), "Ответ должен быть строкой"
    # Проверяем наличие ключевого слова из нашего интерфейса
    assert "отзыв" in result.lower(), "Строка ответа не содержит слово 'отзыв'"


def test_app_initialization():
    """Тест 3: Проверяет, что веб-приложение может запуститься без ошибок."""
    # Проверяем, что объект интерфейса Gradio успешно создан
    assert iface is not None, "Интерфейс не был инициализирован"
    # Проверяем, что подтянулся правильный заголовок
    assert iface.title == "🎬 Анализатор тональности отзывов о фильмах", "Заголовок приложения не совпадает"
