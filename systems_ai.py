import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_wine

# Загрузим датасет
wine = load_wine()

# Преобразуем данные в DataFrame для удобства работы
df = pd.DataFrame(data=wine['data'], columns=wine['feature_names'])

# Добавляем целевую переменную в DataFrame
df['target'] = wine['target']

# Разделим данные на признаки (X) и целевую переменную (y)
X = df.drop('target', axis=1).values
y = df['target'].values

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделим данные на обучающие и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Определяем модель нейронной сети
model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    verbose=1)

# Оценка точности модели на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность модели на тестовом наборе: {accuracy * 100:.2f}%")
