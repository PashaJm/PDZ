import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)

# Создание модели
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Сохранение модели
model.save('mnist_model.h5')


import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import foolbox as fb
import matplotlib.pyplot as plt

# 1. Загрузка обученной модели
model_path = 'mnist_model.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели '{model_path}' не найден. Пожалуйста, убедитесь, что путь указан правильно.")

model = load_model(model_path)
print("Модель успешно загружена.")

# 2. Загрузка и предобработка данных
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # Форма: (num_samples, 28, 28, 1)

# Преобразование меток из uint8 в int64
y_test = y_test.astype(np.int64)

# Преобразование в TensorFlow Tensors
x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int64)

# 3. Инициализация модели Foolbox
fmodel = fb.TensorFlowModel(model, bounds=(0, 1))
print("Foolbox модель инициализирована.")

# 4. Оценка точности на чистых данных
def evaluate_clean_accuracy(model, x, y):
    predictions = model.predict(x)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == y.numpy())
    return accuracy

clean_accuracy = evaluate_clean_accuracy(model, x_test_tf, y_test_tf)
print(f"Точность модели на чистых данных: {clean_accuracy * 100:.2f}%")

# 5. Создание атаки PGD
attack = fb.attacks.LinfPGD()
epsilon = 0.3  # Уровень шума для атаки

# 6. Генерация противоречивых примеров
print("Генерация противоречивых примеров с помощью PGD атаки...")
try:
    adversarials, clipped, is_adv = attack(fmodel, x_test_tf, y_test_tf, epsilons=epsilon)
    print("Генерация завершена.")
except tf.errors.InvalidArgumentError as e:
    print("Произошла ошибка при генерации атак:", e)
    print("Убедитесь, что метки имеют тип int32 или int64.")
    raise

# Пример визуализации:
def visualize_adversarials(x, adversarials, y, is_adv, num=5):
    plt.figure(figsize=(10, 4))
    successful_indices = np.where(is_adv.numpy())[0]
    for i in range(min(num, len(successful_indices))):
        idx = successful_indices[i]

        # Оригинальное изображение
        plt.subplot(2, num, i + 1)
        plt.imshow(x[idx].numpy().squeeze(), cmap="gray")
        plt.title(f"Оригинал: {y[idx].numpy()}")
        plt.axis('off')

        # Атакованное изображение
        plt.subplot(2, num, num + i + 1)
        plt.imshow(adversarials[idx].numpy().squeeze(), cmap="gray")
        plt.title("Атаковано")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_adversarial_accuracy(model, adversarials, y, is_adv):
    # Используем только атакованные примеры
    adversarial_examples = adversarials[is_adv]
    adversarial_labels = y[is_adv]
    predictions = model.predict(adversarial_examples)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == adversarial_labels)
    return accuracy

adversarial_accuracy = evaluate_adversarial_accuracy(model, adversarials, y_test, is_adv)
print(f"Точность модели на атакованных данных: {adversarial_accuracy * 100:.2f}%")
# Визуализируем первые 5 успешных атак
num_successful = np.sum(is_adv.numpy())
if num_successful >= 5:
    visualize_adversarials(x_test_tf, adversarials, y_test_tf, is_adv, num=5)
else:
    print("Недостаточно успешных атак для визуализации.")
print("\nДополнительная статистика:")
print(f"Общая точность на чистых данных: {clean_accuracy * 100:.2f}%")
print(f"Точность на атакованных данных: {adversarial_accuracy * 100:.2f}%")