import gradio as gr
import pickle

# 1. Загрузка сохраненной модели и векторизатора
print("Загрузка модели...")
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    vectorizer = data['vectorizer']


# 2. Функция предсказания, которая будет работать "под капотом" интерфейса
def predict_sentiment(review_text):
    if not review_text.strip():
        return "Пожалуйста, введите текст отзыва."

    # Преобразуем введенный текст в числа тем же способом, что и при обучении
    text_vectorized = vectorizer.transform([review_text])

    # Делаем предсказание (0 - негатив, 1 - позитив)
    prediction = model.predict(text_vectorized)[0]

    # Возвращаем понятный результат
    if prediction == 1:
        return "🟢 Позитивный отзыв"
    else:
        return "🔴 Негативный отзыв"


# 3. Настройка веб-интерфейса Gradio
iface = gr.Interface(
    fn=predict_sentiment,  # Функция, которая обрабатывает ввод
    inputs=gr.Textbox(lines=5,
                      placeholder="Напишите отзыв на фильм здесь (желательно на английском, так как модель обучалась на нем)...",
                      label="Текст отзыва"),
    outputs=gr.Text(label="Результат анализа:"),
    title="🎬 Анализатор тональности отзывов о фильмах",
    description="Введите отзыв на фильм, и нейросеть определит, положительный он или отрицательный.",
    theme="default"
)

# 4. Запуск приложения
if __name__ == "__main__":
    print("Запуск веб-интерфейса...")
    iface.launch()