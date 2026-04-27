import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


print("Загрузка и подготовка данных...")
df = pd.read_csv('imdb_dataset.csv')  # Ожидаются колонки 'review' и 'sentiment'

# Какие столбцы — это признаки (X), а какой — правильный ответ (y):
# X (признаки) — это столбец 'review', содержащий текст отзыва.
# y (целевая переменная) — это столбец 'sentiment', содержащий класс (позитив/негатив).

# Превращаем категориальные признаки в числа
sentiment_dict = {'positive': 1, 'negative': 0}
df['sentiment_num'] = df['sentiment'].map(sentiment_dict)

X = df['review']
y = df['sentiment_num']

# Разделяем данные на две части.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование/преобразование признаков: для текста мы используем векторизацию (TF-IDF),
# что заменяет стандартное масштабирование числовых признаков, переводя слова в частотные веса.
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


print("\nОбучение моделей...")


# 1) Простое правило, основанное на одном признаке (собственная реализация)
# Алгоритм: подсчет соотношения "хороших" и "плохих" слов.
def simple_lexicon_predict(texts):
    good_words = ['good', 'great', 'excellent', 'amazing', 'best', 'love']
    bad_words = ['bad', 'terrible', 'awful', 'worst', 'boring', 'hate']
    predictions = []
    for text in texts:
        text_lower = text.lower()
        good_count = sum(text_lower.count(w) for w in good_words)
        bad_count = sum(text_lower.count(w) for w in bad_words)
        predictions.append(1 if good_count >= bad_count else 0)
    return np.array(predictions)


# 2) Готовый сложный алгоритм из библиотеки (Логистическая регрессия)
model_lr = LogisticRegression(max_iter=1000)

# Кросс-валидация для сложного алгоритма
cv_scores = cross_val_score(model_lr, X_train_vec, y_train, cv=5, scoring='accuracy')
print(f"Кросс-валидация (Логистическая регрессия) - Средняя точность: {cv_scores.mean():.4f}")

# Обучаем финальную версию сложной модели на всем train
model_lr.fit(X_train_vec, y_train)


print("\nДиагностика на тестовых (отложенных) данных...")

# Оценка простой модели
y_pred_simple = simple_lexicon_predict(X_test)
print("--- Отчет по простой модели (Lexicon) ---")
print(classification_report(y_test, y_pred_simple, target_names=['Negative', 'Positive']))

# Оценка сложной модели
y_pred_lr = model_lr.predict(X_test_vec)
print("--- Отчет по сложной модели (Логистическая регрессия) ---")
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

# Визуализация главной ошибки сложной модели (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Матрица ошибок (Логистическая регрессия)')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("График матрицы ошибок сохранен как 'confusion_matrix.png'.")

# Выбираем модель с лучшей и наиболее стабильной метрикой (Логистическая регрессия).
# Сохраняем модель и векторизатор в файл, чтобы не обучать заново.
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model_lr, 'vectorizer': vectorizer}, f)

# Финальный консольный отчет
print("\n=======================================================")
print(f"Лучшая модель — Логистическая регрессия с TF-IDF векторизацией.")
print(f"Её ключевая метрика (Accuracy) на новых данных — {accuracy_score(y_test, y_pred_lr):.4f}.")
# Анализ матрицы ошибок для вывода:
false_positives = cm[0, 1]
false_negatives = cm[1, 0]
if false_positives > false_negatives:
    print(f"Чаще всего она путает Негативные отзывы с Позитивными (ложноположительные).")
else:
    print(f"Чаще всего она путает Позитивные отзывы с Негативными (ложноотрицательные).")
print("=======================================================")
