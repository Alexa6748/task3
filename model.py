import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных из csv-файла
df = pd.read_csv('data.csv')

# Обработка текста
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Текст'])
y = df['sentiment']

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Обучение модели
model = MultinomialNB()
model.fit(X_train, y_train)

# Оценка точности модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
